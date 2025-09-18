import matplotlib.pyplot as plt
import seaborn
import os, pandas, torch
import numpy as np
from PIL import Image, ImageOps
import cv2
import sys

def save_heatmap(
    arr,
    figsize=(6, 8),
    save_path=None,
    title=None,
    vmax=None,
    vmin=None,
    xlabel=None,
    ylabel=None,
    aspect="equal",
):
    """
    arr: ndarray or torch.Tensor
    """
    try:
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
    except ImportError:
        pass

    plt.figure(figsize=figsize)
    im = plt.imshow(
        arr,
        vmax=vmax,
        vmin=vmin,
        origin="lower",
        interpolation="nearest",
        aspect=aspect,
        cmap="magma",
    )
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    # plt.show()
    plt.close()

dataset = sys.argv[1]

# 0. choose and load attention tensor file (.pt)
filepath = f"/media/genchiprofac/Projects/CUT3R/experiments/attentions/{dataset}_attn_seq.pt"
dir = filepath[:-3]
os.makedirs(dir, exist_ok=True)


attns = torch.load(filepath, map_location="cpu")  # dict (key: img/state -> self/cross)
# print(len(attns))
# print(np.array(attns["state"]["cross"]).shape)  # (depth(12), batch(1), head(16), state dim (768), image feat + pose dim (576+1))


# 1. save CA heatmap for every n=1 image steps
def save_heatmap_by_image_step():
    for i in range(len(attns) // 1):
        step = i * 1
        attn = attns[step]
        state_cross_attns = np.array(attn["state"]["cross"]).squeeze(
            1
        )  # (depth, f_state, f_image)

        # mean over decoder depth dimension(12)
        save_heatmap(
            np.log(np.mean(state_cross_attns, axis=0)),
            save_path=dir + f"/ca_heatmap_step{step}.png",
            title=f"State Decoder Cross-Attention Map \n(state v img, mean over depth+head, step={step})",
            xlabel="pose + img dim",
            ylabel="state token dim",
        )


# 2. visualize revisit ratio
def visualize_revisit_ratio():
    ca_init = np.array(attns[1]["state"]["cross"]).squeeze(1)
    ca_revisit = np.array(attns[3]["state"]["cross"]).squeeze(1)

    change = np.divide(ca_revisit, ca_init)

    save_heatmap(
        np.mean(change, axis=0),
        save_path=dir + f"/revisit_change.png",
        title="CA Ratio (initial visit -> revisit)",
        vmax=2,
        vmin=0,
        xlabel="pose + img dim",
        ylabel="state token dim",
    )


# 3. visualize CA for chosen single state token vs image features over depth
def save_heatmap_for_single_state_token():
    token_num = 100
    attn = np.array(attns[1]["state"]["cross"]).squeeze(1)
    attn_by_depth = attn[
        :, token_num, :
    ]  # state token of choosing (token_num)_th, dims: (depth, f_image)

    save_heatmap(
        np.log(attn_by_depth),
        save_path=dir + f"/ca_by_head_token_{token_num}.png",
        title=f"Cross attention weight by head, state token #{token_num}",
        xlabel="pose + image dim",
        ylabel="head",
        figsize=(8, 8),
        aspect=attn_by_depth.shape[1] / attn_by_depth.shape[0],
    )


def image_to_state_ca_visualization(step):
    step = step
    attn = attns[step]
    from pathlib import Path

    image_dir = Path(f"/media/genchiprofac/Projects/assets/{dataset}")
    image_files = sorted(
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.jpeg"))
    )
    image = Image.open(image_files[step % len(image_files)]).convert("RGB")
    image = ImageOps.exif_transpose(image)
    img_arr = np.array(image)

    h, w, _ = img_arr.shape

    if h >= w:
        rw = 32 * w / h
        rh = 32
    else:
        rw = 32
        rh = 32 * h / w

    attn_state_img = np.array(attn["state"]["cross"]).squeeze(1)
    # print(attn_state_img.shape)
    # print(attn_state_img.mean(axis=1)[:,0])
    # print(attn_state_img.mean(axis=1)[:,1])
    # print(attn_state_img.mean(axis=1)[:, 2])

    how = sys.argv[2]

    if how == "last":
        camap_state_img = attn_state_img[-1, :,:].mean(axis=0)[1:].reshape(int(rh), int(rw))    # only use last decoder block
    elif how == "avg":
        camap_state_img = attn_state_img.mean(axis=(0, 1))[1:].reshape(int(rh), int(rw))      # take depthwise mean for all decoders
    elif how == "first":
        camap_state_img = (
            attn_state_img[0, :, :].mean(axis=0)[1:].reshape(int(rh), int(rw))
        )  # only use first decoder block

    else:
        ...
    camap_state_img[0][0] = 0.001                                                           # 1st entry dominant (100x) -- why ?
    # save_heatmap(
    #     np.log(camap_state_img),
    #     save_path=dir + f"/ca_s2i_step{step}.png",
    #     title=f"State to image CA",
    # )

    # visualization of heatmap on original image

    hm_normalized = (camap_state_img - np.min(camap_state_img)) / (
        np.max(camap_state_img) - np.min(camap_state_img)
    )
    hm_normalized = hm_normalized.astype(np.float32)

    heatmap_resized = cv2.resize(hm_normalized, (w, h), interpolation=cv2.INTER_NEAREST)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    alpha = 0.5  # 히트맵의 투명도 (0.0 ~ 1.0)
    # OpenCV는 BGR 순서를 사용하므로 원본 이미지를 RGB -> BGR로 변환
    original_image_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    overlayed_image = cv2.addWeighted(
        heatmap_color, alpha, original_image_bgr, 1 - alpha, 0
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # original image
    axes[0].imshow(img_arr)
    axes[0].set_title("Image Frame")
    axes[0].axis("off")

    # heatmap
    axes[1].imshow(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Attention Heatmap")
    axes[1].axis("off")

    # heatmap overlay
    axes[2].imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(dir + f"/heatmap_overlay_step{step}_{how}.png")
    plt.close()

    ...


if __name__ == "__main__":
    # save_heatmap_by_image_step()
    # visualize_revisit_ratio()
    # save_heatmap_for_single_state_token()
    for i in range(30):

        image_to_state_ca_visualization(i)
