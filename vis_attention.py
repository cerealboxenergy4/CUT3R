import matplotlib.pyplot as plt
import seaborn
import os, pandas, torch
import numpy as np


def save_heatmap(arr, save_path=None, title=None):
    """
    arr: ndarray or torch.Tensor
    """
    try:
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
    except ImportError:
        pass

    plt.figure(figsize=(8, 6))  # 정사각 화면
    im = plt.imshow(
        arr,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        # vmin=-2,
        # vmax=10,
        cmap="magma",
    )
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if title:
        plt.title(title)
    plt.xlabel("pose + img dim")
    plt.ylabel("state token dim")
    plt.title("Cross-Attention Map (state v img, mean over depth+head)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    # plt.show()
    plt.close()


filepath = (
    "/media/genchiprofac/Projects/CUT3R/experiments/attentions/round1f_20_attn_seq.pt"
)
dir = filepath[:-3]
os.makedirs(dir, exist_ok=True)


attns = torch.load(filepath, map_location="cpu") # dict (key: img/state -> self/cross)
print(len(attns))

# print(np.array(attns["state"]["cross"]).shape)  # (depth, batch(1), head(16), state dim (768), image feat + pose dim (576+1))

for i in range(len(attns) // 10):
    step = i * 10 
    attn = attns[10]
    state_cross_attns = np.array(attn["state"]["cross"]).squeeze(1) # (depth, f_state, f_image)

    # depth에 대한 mean해서 (states, img) 히트맵 시각화
    save_heatmap(np.log(np.mean(state_cross_attns, axis=0)), save_path=dir + f"/ca_heatmap_step{step}.png")
