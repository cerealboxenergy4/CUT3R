import os
import os.path as osp
import sys

import numpy as np

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


def _parse_calib_file(path):
    calib = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(":", 1)
            data = np.fromstring(value, sep=" ", dtype=np.float64)
            if data.size == 12:
                calib[key] = data.reshape(3, 4)
            else:
                calib[key] = data
    return calib


def _to_homogeneous(matrix_3x4):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = matrix_3x4
    return matrix


def _resolve_sequences_root(root):
    candidates = [
        root,
        osp.join(root, "sequences"),
        osp.join(root, "dataset", "sequences"),
    ]
    for candidate in candidates:
        if osp.isdir(candidate) and osp.isdir(osp.join(candidate, "00")):
            return candidate
    return candidates[-1]


def _resolve_poses_root(root, sequences_root):
    candidates = [
        osp.join(root, "poses"),
        osp.join(root, "dataset", "poses"),
        osp.join(osp.dirname(sequences_root), "poses"),
        osp.join(osp.dirname(osp.dirname(sequences_root)), "poses"),
    ]
    for candidate in candidates:
        if osp.isdir(candidate):
            return candidate
    return candidates[0]


class KITTIOdometry_Multi(BaseMultiViewDataset):
    def __init__(
        self,
        ROOT,
        *args,
        sequences=None,
        camera="image_2",
        max_interval=8,
        pose_root=None,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.camera = camera
        self.video = True
        self.is_metric = True
        self.max_interval = max_interval
        self.sequence_filter = None if sequences is None else {str(seq) for seq in sequences}
        super().__init__(*args, **kwargs)

        self.sequences_root = _resolve_sequences_root(self.ROOT)
        self.poses_root = pose_root or _resolve_poses_root(self.ROOT, self.sequences_root)
        self._load_data()

    def _load_data(self):
        seq_dirs = sorted(
            d for d in os.listdir(self.sequences_root) if osp.isdir(osp.join(self.sequences_root, d))
        )
        if self.sequence_filter is not None:
            seq_dirs = [d for d in seq_dirs if d in self.sequence_filter]

        offset = 0
        self.scenes = []
        self.sceneids = []
        self.images = []
        self.start_img_ids = []
        self.scene_img_list = []
        self.calibs = {}
        self.poses = {}

        for seq in seq_dirs:
            seq_dir = osp.join(self.sequences_root, seq)
            image_dir = osp.join(seq_dir, self.camera)
            lidar_dir = osp.join(seq_dir, "velodyne")
            calib_path = osp.join(seq_dir, "calib.txt")
            pose_path = osp.join(self.poses_root, f"{seq}.txt")
            if not (osp.isdir(image_dir) and osp.isdir(lidar_dir) and osp.isfile(calib_path) and osp.isfile(pose_path)):
                continue

            frame_names = sorted(
                osp.splitext(name)[0]
                for name in os.listdir(image_dir)
                if name.endswith(".png")
            )
            frame_names = [
                frame_name
                for frame_name in frame_names
                if osp.isfile(osp.join(lidar_dir, f"{frame_name}.bin"))
            ]
            cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            if len(frame_names) < cut_off:
                continue

            calib = _parse_calib_file(calib_path)
            projection = calib[self.camera.replace("image_", "P")]
            intrinsics = projection[:, :3].astype(np.float32)
            cam2_from_cam0 = np.eye(4, dtype=np.float64)
            cam2_from_cam0[:3, 3] = np.linalg.inv(intrinsics.astype(np.float64)) @ projection[:, 3]
            cam0_from_cam2 = np.linalg.inv(cam2_from_cam0)

            velo_key = None
            for candidate in ("Tr", "Tr_velo_to_cam", "Tr_velo_cam"):
                if candidate in calib:
                    velo_key = candidate
                    break
            if velo_key is None:
                continue

            cam0_from_velo = _to_homogeneous(calib[velo_key])
            cam2_from_velo = cam2_from_cam0 @ cam0_from_velo

            poses = np.loadtxt(pose_path, dtype=np.float64).reshape(-1, 3, 4)
            poses = np.stack([_to_homogeneous(pose) for pose in poses], axis=0)
            poses = poses @ cam0_from_cam2[None]
            frame_names = [frame_name for frame_name in frame_names if int(frame_name) < len(poses)]
            if len(frame_names) < cut_off:
                continue

            self.calibs[seq] = {
                "intrinsics": intrinsics,
                "cam2_from_velo": cam2_from_velo.astype(np.float64),
            }
            self.poses[seq] = poses.astype(np.float32)

            img_ids = list(np.arange(len(frame_names)) + offset)
            scene_idx = len(self.scenes)
            self.scenes.append(seq)
            self.scene_img_list.append(img_ids)
            self.sceneids.extend([scene_idx] * len(frame_names))
            self.images.extend(frame_names)
            self.start_img_ids.extend(img_ids[: len(frame_names) - cut_off + 1])
            offset += len(frame_names)

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def get_stats(self):
        return f"{len(self)} groups of views across {len(self.scenes)} KITTI odometry sequences"

    def _project_lidar_to_depth(self, lidar_path, image_shape, intrinsics, cam2_from_velo):
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        xyz1 = np.concatenate([points[:, :3], np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        cam_points = (cam2_from_velo @ xyz1.T).T[:, :3]
        valid = cam_points[:, 2] > 1e-3
        cam_points = cam_points[valid]

        pixels = (intrinsics @ cam_points.T).T
        zs = pixels[:, 2]
        us = np.round(pixels[:, 0] / zs).astype(np.int32)
        vs = np.round(pixels[:, 1] / zs).astype(np.int32)

        height, width = image_shape[:2]
        in_bounds = (us >= 0) & (us < width) & (vs >= 0) & (vs < height)
        us = us[in_bounds]
        vs = vs[in_bounds]
        zs = zs[in_bounds].astype(np.float32)

        depthmap = np.zeros((height, width), dtype=np.float32)
        flat_idx = vs * width + us
        flat_depth = np.full(height * width, np.inf, dtype=np.float32)
        np.minimum.at(flat_depth, flat_idx, zs)
        valid_depth = np.isfinite(flat_depth)
        depthmap.reshape(-1)[valid_depth] = flat_depth[valid_depth]
        return depthmap

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        seq = self.scenes[scene_id]
        all_image_ids = self.scene_img_list[scene_id]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            video_prob=1.0,
            fix_interval_prob=0.9,
        )
        image_idxs = np.array(all_image_ids)[pos]

        seq_dir = osp.join(self.sequences_root, seq)
        image_dir = osp.join(seq_dir, self.camera)
        lidar_dir = osp.join(seq_dir, "velodyne")
        calib = self.calibs[seq]
        poses = self.poses[seq]

        views = []
        for v, view_idx in enumerate(image_idxs):
            frame_idx = int(self.images[view_idx])
            frame_name = self.images[view_idx]
            image_path = osp.join(image_dir, f"{frame_name}.png")
            lidar_path = osp.join(lidar_dir, f"{frame_name}.bin")

            image = imread_cv2(image_path)
            depthmap = self._project_lidar_to_depth(
                lidar_path,
                image.shape,
                calib["intrinsics"],
                calib["cam2_from_velo"],
            )
            intrinsics = calib["intrinsics"].copy()
            camera_pose = poses[frame_idx].copy()

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq, frame_name)
            )
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.85, 0.1, 0.05]
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="KITTIOdometry",
                    label=seq,
                    is_metric=self.is_metric,
                    instance=image_path,
                    is_video=ordered_video,
                    quantile=np.array(0.99, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )
        return views
