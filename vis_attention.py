import matplotlib.pyplot as plt
import seaborn
import os, pandas, torch
import numpy as np


def save_heatmap(
    arr,
    figsize=(6,8),
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


# 0. choose and load attention tensor file (.pt)
filepath = (
    "/media/genchiprofac/Projects/CUT3R/experiments/attentions/chimera_attn_seq.pt"
)
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
    attn_by_depth = attn[:, token_num, :]  # state token of choosing (token_num)_th, dims: (depth, f_image)
    
    save_heatmap(
        np.log(attn_by_depth),
        save_path=dir + f"/ca_by_head_token_{token_num}.png",
        title=f"Cross attention weight by head, state token #{token_num}",
        xlabel= "pose + image dim",
        ylabel = "head",
        figsize=(8,8),
        aspect = attn_by_depth.shape[1]/attn_by_depth.shape[0],
    )

if __name__ == "__main__":
    # save_heatmap_by_image_step()
    # visualize_revisit_ratio()
    save_heatmap_for_single_state_token()
