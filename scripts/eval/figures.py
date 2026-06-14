"""Composite figure: spec / GT / model prediction for each domain.

Loads a few example images from each of the four datasets, runs the model
zero-shot, and stacks one row per example into a single PNG.
"""
from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import RadDet
import TJII2021

from tokeye.extra.eval.silbidopy.data import AudioTonalDataset
from tokeye.extra.eval.silbidopy.eval import Metrics  # noqa: F401  (preload)

root_path = Path("/scratch/gpfs/nc1514/tokeye")
results_dir = root_path / "data" / "eval" / "results"
fig_dir = results_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)


def panel_row(ax_row, spec, mask, sigmoid, title_prefix="",
              display_lo_percentile=10):
    """Visualisation only. The lower percentile clip raises vmin so the
    noise floor saturates into the colormap minimum and signal contrast is
    visible. Metrics in the CSVs are computed on un-clipped sigmoid output.
    """
    vmin = float(np.percentile(spec, display_lo_percentile))
    vmax = float(spec.max())
    ax_row[0].imshow(spec, origin="lower", aspect="auto", cmap="gist_heat",
                     vmin=vmin, vmax=vmax)
    ax_row[0].set_title(f"{title_prefix} spec (preprocessed)")
    ax_row[1].imshow(mask, origin="lower", aspect="auto", cmap="gray",
                     vmin=0, vmax=1)
    ax_row[1].set_title("ground truth")
    ax_row[2].imshow(sigmoid, origin="lower", aspect="auto", cmap="gray",
                     vmin=0, vmax=1)
    n_pos = (sigmoid > RadDet.PRIMARY_THRESHOLD).sum()
    ax_row[2].set_title(f"prediction (>{RadDet.PRIMARY_THRESHOLD}: {n_pos} px)")


def _forward_sig(model, spec_t):
    return RadDet._forward_sig(model, spec_t)


def figure_RadDet(model, n=4, variant="RadDet40k128HW001Tv2"):
    parts = sorted(RadDet.data_path.glob(f"{variant}.tar.part-*"))
    proc = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    tf = tarfile.open(fileobj=proc.stdout, mode="r|")
    pi, pl, samples = {}, {}, []
    try:
        for m in tf:
            if not m.isfile():
                continue
            if "/images/test/" in m.name and m.name.endswith(".png"):
                s = Path(m.name).stem
                pi[s] = tf.extractfile(m).read()
            elif "/labels/test/" in m.name and m.name.endswith(".txt"):
                s = Path(m.name).stem
                pl[s] = tf.extractfile(m).read()
            common = pi.keys() & pl.keys()
            for s in list(common):
                lbl = pl.pop(s); png = pi.pop(s)
                nbox = sum(1 for line in lbl.decode().splitlines() if line.strip())
                if nbox >= 1:
                    samples.append((s, png, lbl))
                    if len(samples) >= n:
                        break
            if len(samples) >= n:
                break
    finally:
        proc.stdout.close(); proc.wait()

    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
    for i, (stem, png, lbl) in enumerate(samples):
        spec, mask, _ = RadDet.preprocess(png, lbl)
        spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(RadDet.device)
        sig = _forward_sig(model, spec_t)
        if RadDet.TTA_TIME_FLIP:
            sig_f = _forward_sig(model, spec_t.flip(-1)).flip(-1)
            sig = 0.5 * (sig + sig_f)
        sig = sig.cpu().numpy()[0, 0]
        panel_row(axes[i], spec, mask, sig, title_prefix=f"RadDet {stem}")
    fig.tight_layout()
    fig.savefig(fig_dir / "RadDet_examples.png", dpi=110)
    plt.close(fig)
    print("saved RadDet_examples.png")


def figure_TJII(model, n=4):
    inputs = sorted((TJII2021.data_path / "input").glob("spectrogram_*.png"))[:n]
    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
    for i, ip in enumerate(inputs):
        shotn = ip.stem.split("_")[1]
        spec = np.array(Image.open(ip).convert("L"))
        ann = np.array(Image.open(TJII2021.data_path / "gt" / ip.name).convert("L"))
        spec = np.flip(spec, 0).copy()
        ann = np.flip(ann, 0).copy()
        spec = (spec - TJII2021.MEAN) / TJII2021.STD
        ann = (ann // 255).astype(np.float32)
        spec_t = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0).to(RadDet.device)
        with torch.no_grad():
            out = model(spec_t)[0]
        sig = torch.sigmoid(out[:, 0:1]).cpu().numpy()[0, 0]
        panel_row(axes[i], spec, ann, sig, title_prefix=f"TJII shot {shotn}")
    fig.tight_layout()
    fig.savefig(fig_dir / "TJII2021_examples.png", dpi=110)
    plt.close(fig)
    print("saved TJII2021_examples.png")


def figure_DCLDE(model, n=4):
    species = "Delphinus capensis"
    stats = {"mean": 0.7080333218912784, "std": 0.051389602618240104}

    class P:
        def __init__(self, m, s):
            self.m, self.s = m, s
        def __call__(self, x):
            return (x - self.m) / self.s

    data_dir = root_path / "data" / "eval" / "DCLDE2011" / species
    ds = AudioTonalDataset(
        data_dir, data_dir,
        annotation_extension="ann",
        time_patch_frames=250, freq_patch_frames=250,
        post_processing_function=P(stats["mean"], stats["std"]),
    )
    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
    indices = np.linspace(0, len(ds) - 1, n).astype(int)
    for row_i, ds_i in enumerate(indices):
        spec, ann = ds[int(ds_i)]
        spec = np.flip(spec, 0).copy()
        ann = np.flip(ann, 0).copy()
        spec_t = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0).to(RadDet.device)
        with torch.no_grad():
            out = model(spec_t)[0]
        sig = torch.sigmoid(out[:, 0:1]).cpu().numpy()[0, 0]
        panel_row(axes[row_i], spec, ann, sig, title_prefix=f"DCLDE {species} #{ds_i}")
    fig.tight_layout()
    fig.savefig(fig_dir / "DCLDE2011_examples.png", dpi=110)
    plt.close(fig)
    print("saved DCLDE2011_examples.png")


def main():
    print(f"device: {RadDet.device}")
    model = RadDet.build_model(RadDet.device)
    print("model loaded")
    figure_RadDet(model)
    figure_TJII(model)
    figure_DCLDE(model)


if __name__ == "__main__":
    main()
