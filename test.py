import torch
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


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
    im = plt.imshow(arr, origin="lower", interpolation="nearest", aspect="equal")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if title:
        plt.title(title)
    plt.xlabel("X (state encoder dim)")
    plt.ylabel("Y (state size (# tokens))")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


npypath = "/media/genchiprofac/Projects/CUT3R/experiments/state_per_frame/test_house_1x_2fps.npy"

arr = np.load(npypath)
# print(arr.shape)  # (60, 1, 768, 768)

states = arr.squeeze(1)  # (60, 768, 768)

for state in states:
    ...

step = 100
title = npypath[63:-4] + f"_step_{step}"
show_heatmap(states[step], save_path=npypath[:-4] + f"_step_{step}.png", title=title)
