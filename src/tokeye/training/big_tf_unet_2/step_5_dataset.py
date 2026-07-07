"""step_5 — merge all modalities into the segmenter training dataset.

Replaces the TIF-based step_6a: one HDF5 with ``/images/{n}`` float32
``(1, F, T)`` and ``/masks/{n}`` uint8 ``(2, F, T)`` (each (window, channel)
pair is one training sample), plus provenance datasets so any sample maps
back to (modality, window, channel).

Image normalization is the pipeline's single convention at the segmenter
site: ``N_a(x) = a*asinh((x - median)/(a*scale))`` of ``log1p(|Z|)`` with
**per-modality global** robust stats (median/1.4826*MAD from sampled
windows) and a=3 — the smooth, invertible generalization of the old
``clip(z, -3, 3)``. The stats are stored in the file attrs and returned to
the runner's ledger; step_7 exports them in the deploy manifest, replacing
the old hardcoded ``NORMALIZATION_CONFIGS`` constants.
"""

from __future__ import annotations

import logging

import h5py
import numpy as np

from .utils.auto_params import compute_logmag_robust_stats, normalize_asinh
from .utils.hdf5_io import iter_samples, read_sample

logger = logging.getLogger(__name__)


def main(settings: dict) -> dict:
    a = settings["a"]
    out_path = settings["out_h5"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    in_step: dict[str, float] = {}
    prov_modality: list[str] = []
    prov_window: list[int] = []
    prov_channel: list[int] = []

    with h5py.File(out_path, "w") as out:
        out.attrs["a"] = a
        images = out.create_group("images")
        masks = out.create_group("masks")
        n = 0
        for mod, spec in settings["inputs"].items():
            med, scale = compute_logmag_robust_stats(
                spec["img_h5"], max_samples=settings["stats_windows"]
            )
            out.attrs[f"{mod}_median"] = med
            out.attrs[f"{mod}_scale"] = scale
            in_step[f"{mod}_median"] = med
            in_step[f"{mod}_scale"] = scale

            n_mod = 0
            with h5py.File(spec["mask_h5"], "r") as f_mask:
                for idx, img in iter_samples(spec["img_h5"]):
                    mag = np.sqrt(img[..., 0] ** 2 + img[..., 1] ** 2)  # (C, F, T)
                    norm = normalize_asinh(np.log1p(mag), a, med, scale).astype(
                        np.float32
                    )
                    mask = read_sample(f_mask, idx).astype(np.uint8)  # (C, 2, F, T)
                    for c in range(norm.shape[0]):
                        images.create_dataset(
                            str(n), data=norm[c][None], compression="lzf"
                        )
                        masks.create_dataset(
                            str(n), data=mask[c], compression="lzf"
                        )
                        prov_modality.append(mod)
                        prov_window.append(idx)
                        prov_channel.append(c)
                        n += 1
                        n_mod += 1
            logger.info(f"{mod}: {n_mod} samples (median={med:.4f}, scale={scale:.4f})")

        out.create_dataset(
            "prov_modality", data=np.array(prov_modality, dtype="S8")
        )
        out.create_dataset("prov_window", data=np.array(prov_window, dtype=np.int64))
        out.create_dataset("prov_channel", data=np.array(prov_channel, dtype=np.int64))
        out.attrs["n_samples"] = n
        logger.info(f"dataset complete: {n} samples -> {out_path}")

    return in_step
