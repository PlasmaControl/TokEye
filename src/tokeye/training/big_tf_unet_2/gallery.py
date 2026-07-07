"""Visual checkpoints for every step boundary — the intern's decision tool.

All plots lazy-load at most ``n`` samples via per-sample HDF5 reads; nothing
here ever loads a whole step file. ``show(step, ...)`` dispatches to the
right view for each step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .utils.hdf5_io import read_sample

if TYPE_CHECKING:
    from .paths import RunPaths

_CMAP = "gist_heat"


def _mag(arr: np.ndarray) -> np.ndarray:
    """(C,F,T,2) -> (C,F,T) magnitude; passthrough otherwise."""
    if arr.ndim == 4 and arr.shape[-1] == 2:
        return np.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2)
    return np.abs(arr)


def _sample_keys(h5_path, n: int) -> list[int]:
    with h5py.File(h5_path, "r") as f:
        keys = sorted((int(k) for k in f["samples"]), key=int)
    if len(keys) <= n:
        return keys
    idx = np.linspace(0, len(keys) - 1, n).astype(int)
    return [keys[i] for i in idx]


def _imshow(ax, img: np.ndarray, title: str, log: bool = False) -> None:
    data = np.log1p(np.abs(img)) if log else img
    lo, hi = np.quantile(data, [0.01, 0.99])
    ax.imshow(data, aspect="auto", origin="lower", cmap=_CMAP, vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def timeseries_grid(paths: RunPaths, modality: str, n: int = 4) -> None:
    """step_0: raw windowed signals, one channel per row."""
    h5 = paths.step_h5("step_0", modality)
    keys = _sample_keys(h5, n)
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 2 * len(keys)), squeeze=False)
    for ax, k in zip(axes[:, 0], keys):
        arr = read_sample(h5, k)
        ax.plot(arr[0], lw=0.3)
        ax.set_title(f"{modality} window {k} ch0", fontsize=8)
    fig.tight_layout()
    plt.show()


def spectrogram_grid(
    paths: RunPaths, step: str, modality: str, n: int = 6, log: bool = True
) -> None:
    """Any spectrogram-shaped step: grid of channel-0 fields."""
    h5 = paths.step_h5(step, modality)
    keys = _sample_keys(h5, n)
    cols = min(3, len(keys))
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, k in zip(axes.flat, keys):
        arr = _mag(read_sample(h5, k))
        _imshow(ax, arr[0], f"{modality} {step} sample {k} ch0", log=log)
    for ax in axes.flat[len(keys) :]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def pair_grid(
    paths: RunPaths,
    step_a: str,
    step_b: str,
    modality: str,
    n: int = 4,
    log_a: bool = True,
    log_b: bool = False,
) -> None:
    """Before/after pairs (e.g. step_1 vs step_2, step_2 vs step_3)."""
    h5_a, h5_b = paths.step_h5(step_a, modality), paths.step_h5(step_b, modality)
    keys = _sample_keys(h5_b, n)
    fig, axes = plt.subplots(len(keys), 2, figsize=(9, 3 * len(keys)), squeeze=False)
    for row, k in zip(axes, keys):
        _imshow(row[0], _mag(read_sample(h5_a, k))[0], f"{step_a} s{k}", log=log_a)
        _imshow(row[1], _mag(read_sample(h5_b, k))[0], f"{step_b} s{k}", log=log_b)
    fig.tight_layout()
    plt.show()


def mask_overlay(paths: RunPaths, modality: str, n: int = 4) -> None:
    """step_4: denoised field with coherent/transient mask contours."""
    h5_den = paths.step_h5("step_3", modality)
    h5_mask = paths.step_h5("step_4", modality)
    keys = _sample_keys(h5_mask, n)
    fig, axes = plt.subplots(len(keys), 2, figsize=(9, 3 * len(keys)), squeeze=False)
    for row, k in zip(axes, keys):
        den = _mag(read_sample(h5_den, k))[0]
        mask = read_sample(h5_mask, k)  # (C, 2, F, T)
        for ax, ch, name in ((row[0], 0, "coherent"), (row[1], 1, "transient")):
            _imshow(ax, den, f"s{k} {name}")
            m = mask[0, ch].astype(float)
            if m.any():
                ax.contour(m, levels=[0.5], colors="cyan", linewidths=0.6)
            ax.set_title(f"s{k} {name} ({m.mean():.2%} px)", fontsize=8)
    fig.tight_layout()
    plt.show()


def knee_plot(paths: RunPaths, modality: str, max_curves: int = 8) -> None:
    """step_4: the thresholds.csv knee decisions, one curve per (shot, target)."""
    df = pd.read_csv(paths.thresholds_csv(modality))
    fig, ax = plt.subplots(figsize=(8, 4))
    shown = df.head(max_curves)
    ax.scatter(
        range(len(shown)),
        shown["threshold"],
        c=["tab:blue" if t == "coherent" else "tab:orange" for t in shown["target"]],
    )
    for i, (_, r) in enumerate(shown.iterrows()):
        marker = " (fallback)" if r["used_fallback"] else ""
        ax.annotate(
            f"{r['shotn']} ch{r['channel']} {r['target']}{marker}\n"
            f"{r['positive_fraction']:.2%} px",
            (i, r["threshold"]),
            fontsize=6,
            rotation=45,
        )
    ax.set_ylabel("threshold (robust sigma)")
    ax.set_xticks([])
    ax.set_title(f"{modality}: knee thresholds (blue=coherent, orange=transient)")
    fig.tight_layout()
    plt.show()
    n_fb = int(df["used_fallback"].sum())
    if n_fb:
        print(f"note: {n_fb}/{len(df)} thresholds used the quantile fallback")


def dataset_grid(paths: RunPaths, n: int = 6) -> None:
    """step_5: normalized training images with their masks."""
    h5 = paths.step_h5("step_5")
    with h5py.File(h5, "r") as f:
        total = int(f.attrs["n_samples"])
        idx = np.linspace(0, total - 1, min(n, total)).astype(int)
        fig, axes = plt.subplots(
            len(idx), 2, figsize=(9, 3 * len(idx)), squeeze=False
        )
        for row, i in zip(axes, idx):
            img = np.asarray(f["images"][str(i)])[0]
            mask = np.asarray(f["masks"][str(i)])
            mod = f["prov_modality"][i].decode()
            _imshow(row[0], img, f"{mod} sample {i} (N_a image)")
            _imshow(row[1], mask[0] + 2 * mask[1], "mask (coh=1, tra=2)")
    fig.tight_layout()
    plt.show()


def refine_triptych(paths: RunPaths, n: int = 4) -> None:
    """step_6: image / knee label / OOF probability / disagreement."""
    h5_data = paths.step_h5("step_5")
    h5_ref = paths.step_h5("step_6")
    with h5py.File(h5_data, "r") as fd, h5py.File(h5_ref, "r") as fr:
        total = fr["p_oof"].shape[0]
        # Most-disagreeing samples are the ones worth human eyes.
        means = [float(np.mean(fr["disagreement"][i])) for i in range(total)]
        idx = np.argsort(means)[::-1][:n]
        fig, axes = plt.subplots(
            len(idx), 4, figsize=(14, 3 * len(idx)), squeeze=False
        )
        for row, i in zip(axes, idx):
            img = np.asarray(fd["images"][str(i)])[0]
            y0 = np.asarray(fd["masks"][str(i)])[0]
            p = fr["p_oof"][i][0]
            d = fr["disagreement"][i][0]
            _imshow(row[0], img, f"s{i} image")
            _imshow(row[1], y0.astype(float), "knee label y0")
            _imshow(row[2], p, "OOF prob")
            _imshow(row[3], d, f"disagreement ({means[i]:.3f})")
    fig.tight_layout()
    plt.show()


def eval_curves(paths: RunPaths) -> None:
    """step_8: the TJ-II threshold sweep + best point vs deployed anchor."""
    df = pd.read_csv(paths.eval_csv)
    best = df.iloc[df["f1"].idxmax()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["threshold"], df["f1"], label="F1")
    ax.plot(df["threshold"], df["iou_per_image_mean"], label="IoU (per-image)")
    ax.axvline(best["threshold"], ls="--", lw=0.8, color="gray")
    ax.axhline(0.26, ls=":", lw=0.8, color="green", label="deployed IoU anchor 0.26")
    ax.set_xlabel("threshold")
    ax.legend()
    ax.set_title(
        f"TJ-II: IoU={best['iou_per_image_mean']:.3f}, F1={best['f1']:.3f} "
        f"@ {best['threshold']:.2f}"
    )
    fig.tight_layout()
    plt.show()


def show(
    paths: RunPaths,
    step: str,
    modality: str | None = None,
    modalities: list[str] | None = None,
    n: int = 6,
) -> None:
    """Dispatch to the right view for a step. Per-modality steps show every
    modality unless one is named."""
    mods = [modality] if modality else (modalities or [])
    if step == "step_0":
        for m in mods:
            timeseries_grid(paths, m, n=min(n, 4))
    elif step == "step_1":
        for m in mods:
            spectrogram_grid(paths, "step_1", m, n=n)
    elif step == "step_2":
        for m in mods:
            pair_grid(paths, "step_1", "step_2", m, n=min(n, 4))
    elif step == "step_3":
        for m in mods:
            pair_grid(paths, "step_2", "step_3", m, n=min(n, 4), log_a=False)
    elif step == "step_4":
        for m in mods:
            mask_overlay(paths, m, n=min(n, 4))
            knee_plot(paths, m)
    elif step == "step_5":
        dataset_grid(paths, n=n)
    elif step == "step_6":
        refine_triptych(paths, n=min(n, 4))
    elif step == "step_7":
        print(yaml.safe_dump(yaml.safe_load(paths.deploy_manifest.read_text())))
    elif step == "step_8":
        eval_curves(paths)
    else:
        raise KeyError(f"No gallery view for {step}")
