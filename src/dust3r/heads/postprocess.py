# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

import torch
import torch.nn.functional as F


def _sanitize_tensor(x, default=0.0):
    return torch.nan_to_num(x, nan=default, posinf=default, neginf=default)


def _sanitize_quaternion(quats):
    quats = _sanitize_tensor(quats, default=0.0)
    invalid = quats.norm(dim=-1, keepdim=True) < 1e-8
    identity = torch.zeros_like(quats)
    identity[..., 0] = 1.0
    return torch.where(invalid, identity, quats)


def postprocess(out, depth_mode, conf_mode, pos_z=False):
    """
    extract 3D points/confidence from prediction head output
    """
    fmap = _sanitize_tensor(out).permute(0, 2, 3, 1)  # B,H,W,3
    res = dict(pts3d=reg_dense_depth(fmap[:, :, :, 0:3], mode=depth_mode, pos_z=pos_z))

    if conf_mode is not None:
        res["conf"] = reg_dense_conf(fmap[:, :, :, 3], mode=conf_mode)
    return res


def postprocess_rgb(out, eps=1e-6):
    fmap = _sanitize_tensor(out).permute(0, 2, 3, 1)  # B,H,W,3
    res = torch.sigmoid(fmap) * (1 - 2 * eps) + eps
    res = (res - 0.5) * 2
    return dict(rgb=_sanitize_tensor(res))


def postprocess_pose(out, mode, inverse=False):
    """
    extract pose from prediction head output
    """
    mode, vmin, vmax = mode

    no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))
    assert no_bounds
    out = _sanitize_tensor(out)
    trans = out[..., 0:3]
    quats = _sanitize_quaternion(out[..., 3:7])

    if mode == "linear":
        if no_bounds:
            return trans  # [-inf, +inf]
        return trans.clip(min=vmin, max=vmax)

    d = trans.norm(dim=-1, keepdim=True)

    if mode == "square":
        if inverse:
            scale = d / d.square().clip(min=1e-8)
        else:
            scale = d.square() / d.clip(min=1e-8)

    if mode == "exp":
        if inverse:
            scale = d / torch.expm1(d).clip(min=1e-8)
        else:
            scale = torch.expm1(d) / d.clip(min=1e-8)

    trans = trans * scale
    quats = standardize_quaternion(quats)

    return _sanitize_tensor(torch.cat([trans, quats], dim=-1))


def postprocess_pose_conf(out):
    fmap = _sanitize_tensor(out).permute(0, 2, 3, 1)  # B,H,W,1
    return dict(pose_conf=_sanitize_tensor(torch.sigmoid(fmap)))


def postprocess_desc(out, depth_mode, conf_mode, desc_dim, double_channel=False):
    """
    extract 3D points/confidence from prediction head output
    """
    fmap = _sanitize_tensor(out).permute(0, 2, 3, 1)  # B,H,W,3
    res = dict(pts3d=reg_dense_depth(fmap[:, :, :, 0:3], mode=depth_mode))

    if conf_mode is not None:
        res["conf"] = reg_dense_conf(fmap[:, :, :, 3], mode=conf_mode)

    if double_channel:
        res["pts3d_self"] = reg_dense_depth(
            fmap[
                :, :, :, 3 + int(conf_mode is not None) : 6 + int(conf_mode is not None)
            ],
            mode=depth_mode,
        )
        if conf_mode is not None:
            res["conf_self"] = reg_dense_conf(
                fmap[:, :, :, 6 + int(conf_mode is not None)], mode=conf_mode
            )

    start = (
        3
        + int(conf_mode is not None)
        + int(double_channel) * (3 + int(conf_mode is not None))
    )
    res["desc"] = reg_desc(fmap[:, :, :, start : start + desc_dim], mode="norm")
    res["desc_conf"] = reg_dense_conf(fmap[:, :, :, start + desc_dim], mode=conf_mode)
    assert start + desc_dim + 1 == fmap.shape[-1]

    return res


def reg_desc(desc, mode="norm"):
    if "norm" in mode:
        desc = desc / desc.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    else:
        raise ValueError(f"Unknown desc mode {mode}")
    return _sanitize_tensor(desc)


def reg_dense_depth(xyz, mode, pos_z=False):
    """
    extract 3D points from prediction head output
    """
    mode, vmin, vmax = mode

    no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))
    assert no_bounds

    if mode == "linear":
        if no_bounds:
            return _sanitize_tensor(xyz)  # [-inf, +inf]
        return _sanitize_tensor(xyz.clip(min=vmin, max=vmax))

    if pos_z:
        sign = torch.sign(xyz[..., -1:])
        xyz *= sign
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)

    if mode == "square":
        return _sanitize_tensor(xyz * d.square())

    if mode == "exp":
        return _sanitize_tensor(xyz * torch.expm1(d))

    raise ValueError(f"bad {mode=}")


def reg_dense_conf(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == "exp":
        x = _sanitize_tensor(x)
        return _sanitize_tensor(vmin + x.exp().clip(max=vmax - vmin), default=vmin)
    if mode == "sigmoid":
        x = _sanitize_tensor(x)
        return _sanitize_tensor((vmax - vmin) * torch.sigmoid(x) + vmin, default=vmin)
    raise ValueError(f"bad {mode=}")


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    quaternions = F.normalize(quaternions, p=2, dim=-1, eps=1e-8)
    quaternions = _sanitize_quaternion(quaternions)
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)
