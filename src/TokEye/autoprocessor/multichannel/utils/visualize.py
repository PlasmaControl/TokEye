import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import opening, disk, remove_small_objects

# from autotslabel.autosegment.multichannel.step_5_threshold import get_threshold

def plot_data(
    data,
    title=None,
    cmap='gist_heat',
    dpi=100,
    quantiles=True,
    adjust_threshold=0,
):
    if data.dtype == bool:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=dpi)
        im = axs[0].imshow(
            data, 
            cmap='gray', 
            origin='lower', 
            aspect='auto'
        )
        axs[0].axis('off')
        # axs[0].set_xticks([])
        # axs[0].set_yticks([])
    else:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=dpi)
        if quantiles:
            quantiles = np.quantile(data, [0.01, 0.99])
            print(f"quantiles: {quantiles}")
        else:
            quantiles = (data.min(), data.max())
        # threshold = get_threshold(data, adjust=adjust_threshold)
        im = axs[0].imshow(
            data, 
            cmap=cmap, 
            origin='lower', 
            aspect='auto', 
            vmin=quantiles[0], 
            vmax=quantiles[1]
        )
        axs[0].axis('off')
        binary = data > np.percentile(data, 97)#threshold
        binary = remove_small_objects(binary, min_size=5)
        axs[2].imshow(
            binary,
            cmap='gray',
            origin='lower',
            aspect='auto',
        )
        # axs[2].axis('off')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
    axs[1].plot(data.mean(axis=1))
    # axs[1].set_xticks([])
    # axs[1].set_yticks([])
    if title: plt.suptitle(title)
    plt.tight_layout()
    plt.show()