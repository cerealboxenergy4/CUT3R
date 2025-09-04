import torch
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import os


def show_heatmap(arr, save_path=None, title=None):
    """
    arr: (768, 768) ndarray or torch.Tensor
    """
    # torch.Tensor면 NumPy로 변환
    try:
        import torch

        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
    except ImportError:
        pass

    assert arr.shape == (768, 768), f"Expected (768,768), got {arr.shape}"

    plt.figure(figsize=(6, 6))  # 정사각 화면
    im = plt.imshow(
        arr,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        vmin=-2,
        vmax=10,
        cmap="magma",
    )
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if title:
        plt.title(title)
    plt.xlabel("X (state encoder dim)")
    plt.ylabel("Y (state size (# tokens))")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    # plt.show()
    plt.close()


npypath = "/media/genchiprofac/Projects/CUT3R/experiments/state_per_frame/test_round1f_20.npy"
dir = npypath[:-4]
os.makedirs(dir, exist_ok=True)

arr = np.load(npypath)
# print(arr.shape)  # (60, 1, 768, 768)

states = arr.squeeze(1)  # (60, 768, 768)


for i in range(states.shape[0] // 10):
    step = 10 * i
    title = npypath[63:-4] + f"_step_{step}"
    show_heatmap(states[step], save_path=dir + f"/step_{step}.png", title=title)

# infos = np.sum(np.abs(states), axis=(1, 2))
# plt.plot(np.arange(infos.size), infos)
# plt.xlabel("image #")
# plt.ylabel("sum of state token entries")
# plt.tight_layout()
# plt.savefig(dir + "/states.png")
