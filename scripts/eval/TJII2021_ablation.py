"""TJ-II ablation evaluation with uncertainty.

For each ablation variant (full / mag / nobaseline / nodenoise) and each of its
CV-fold surrogates, evaluate zero-shot on the TJ-II 2021 test set and report:
  - recall at the default threshold 0.5,
  - F1 and per-image IoU at the per-fold F1-optimal threshold,
with mean +/- std and 95% CI ACROSS folds, plus image-level bootstrap CIs.

Fold models are loaded from their TorchScript export
``model/ablation/<variant>/fold_<k>/final.torchscript.pt`` (the raw U-Net:
input (1,1,H,W) -> output (1,2,H,W); channel 0 = coherent).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from tokeye.extra.eval.fold_stats import aggregate_across_folds
from tokeye.extra.eval.silbidopy.eval import Metrics
from tokeye.extra.eval.sweep import PRSweep
from tokeye.training.big_tf_unet_ablation.ablation_matrix import build_variants
from tokeye.training.big_tf_unet_ablation.utils.configuration import (
    load_pipeline_config,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

PRIMARY_THRESHOLD = 0.5
# Batch the (deterministic, eval-mode) forward pass to fill the GPU instead of
# running one image at a time. All 493 TJ-II images are 257x368, so they stack
# cleanly. Peak is ~80MB/image (a ~16GB decoder allocation dominates), so 256
# peaks ~23GB on a 32GB V100 with comfortable margin (416/493 OOM it; the GPU
# cache is also emptied between folds, see main). Raise EVAL_BATCH on a 40GB+ A100.
EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "256"))
PR_SWEEP_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05..0.95
MEAN = 17.84620821169868
STD = 25.016818830630463

ROOT = Path("/scratch/gpfs/nc1514/tokeye")
DATA = ROOT / "data" / "eval" / "TJII2021"
RESULTS = ROOT / "data" / "eval" / "results"


def _load_tjii():
    """Yield (spec_tensor (1,1,H,W), mask_tensor (1,1,H,W)) for each TJ-II shot."""
    shotns = [p.stem.split("_")[1] for p in DATA.glob("input/*.png")]
    for shotn in shotns:
        spec = np.array(Image.open(DATA / "input" / f"spectrogram_{shotn}.png").convert("L"))
        ann = np.array(Image.open(DATA / "gt" / f"spectrogram_{shotn}.png").convert("L"))
        spec = np.flip(spec, axis=0).copy()
        ann = np.flip(ann, axis=0).copy()
        spec = (spec - MEAN) / STD
        ann = (ann // 255).astype(np.float32)
        spec_t = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0)
        mask_t = torch.from_numpy(ann).unsqueeze(0).unsqueeze(0).float()
        yield spec_t, mask_t


def _pad32(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]
    ph, pw = (-h) % 32, (-w) % 32
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


@torch.no_grad()
def eval_fold(model, specs_pad, h, w, masks) -> dict:
    """Run one fold model on TJ-II; return per-fold recall@0.5, F1@opt, IoU@opt + sweep.

    ``specs_pad`` is the (N,1,H',W') pad-to-32 batch of every TJ-II spectrogram
    (identical for all folds); ``masks`` is the matching list of (1,1,H,W) GT
    tensors. The forward pass is batched -- the model is in eval mode (no dropout,
    BN uses running stats) so this is bit-equivalent to per-image -- but each
    metric update stays per-image.
    """
    pr = PRSweep(PR_SWEEP_THRESHOLDS, track_per_image=True)
    metrics_default = Metrics(device="cpu")
    n = specs_pad.shape[0]
    for start in range(0, n, EVAL_BATCH):
        out = model(specs_pad[start:start + EVAL_BATCH].to(device))
        if isinstance(out, (tuple, list)):
            out = out[0]
        sig = torch.sigmoid(out[:, 0:1])[..., :h, :w].cpu()
        for j in range(sig.shape[0]):
            mask_t = masks[start + j]
            sig_i = sig[j:j + 1]
            try:
                if sig_i.shape[-2:] != mask_t.shape[-2:]:
                    # rare label/spectrogram size mismatch in the public set (1 shot)
                    continue
                metrics_default.update(sig_i > PRIMARY_THRESHOLD, mask_t)
                pr.update(sig_i, mask_t)
            except Exception:  # noqa: BLE001  (skip the rare malformed shot)
                continue
    rows = list(pr.rows("TJII", key_field="dataset"))
    best_i = max(range(len(rows)), key=lambda j: rows[j]["f1"])
    opt = pr.metrics_at_threshold(best_i)
    default = metrics_default.compute()
    return {
        "recall": float(default["recall"]),  # recall at default 0.5
        "f1": float(rows[best_i]["f1"]),  # F1 at F1-opt
        "iou": float(opt.get("iou_per_image_mean", 0.0)),  # per-image IoU at F1-opt
        "f1_opt_threshold": float(rows[best_i]["threshold"]),
        "_pr": pr,
        "_best_i": best_i,
    }


def main(config_path: str | None = None) -> None:
    cfg = load_pipeline_config(config_path)
    model_root = Path(cfg["paths"]["model_dir"])
    variants = build_variants(cfg)
    RESULTS.mkdir(parents=True, exist_ok=True)
    samples = list(_load_tjii())
    print(f"device={device}  TJ-II images={len(samples)}  variants={[v.id for v in variants]}")
    # Pre-stack all specs into one padded batch, reused across every fold model;
    # batching the eval-mode forward fills the GPU vs one image at a time.
    specs_pad, H, W = _pad32(torch.cat([s for s, _ in samples], dim=0))
    masks = [m for _, m in samples]
    print(f"batched forward: EVAL_BATCH={EVAL_BATCH}, padded batch {tuple(specs_pad.shape)}")

    fold_rows: list[dict] = []
    summary_rows: list[dict] = []
    image_ci_rows: list[dict] = []

    for v in variants:
        vdir = model_root / v.id
        ckpts = sorted(vdir.glob("fold_*/final.torchscript.pt"))
        if not ckpts:
            print(f"WARNING: no fold checkpoints for variant {v.id} under {vdir}; skipping")
            continue
        per_fold: list[dict] = []
        for cpath in tqdm(ckpts, desc=f"variant {v.id}"):
            fold = cpath.parent.name
            model = torch.jit.load(str(cpath), map_location=device).eval()
            res = eval_fold(model, specs_pad, H, W, masks)
            per_fold.append({"recall": res["recall"], "f1": res["f1"], "iou": res["iou"]})
            fold_rows.append({
                "variant": v.id, "fold": fold,
                "recall": res["recall"], "f1": res["f1"], "iou": res["iou"],
                "f1_opt_threshold": res["f1_opt_threshold"],
            })
            # image-level bootstrap CI at the fold's F1-opt threshold (fold 0 reported)
            if fold.endswith("_0"):
                ci = res["_pr"].bootstrap_ci(n_iter=1000, ci=95.0)
                thr = res["_pr"].rows("TJII", key_field="dataset")
                thr = list(thr)[res["_best_i"]]["threshold"]
                if thr in ci:
                    for metric_name, (lo, mean, hi) in ci[thr].items():
                        image_ci_rows.append({
                            "variant": v.id, "threshold": thr, "metric": metric_name,
                            "ci_lo": lo, "mean": mean, "ci_hi": hi,
                        })
            # Free this fold's GPU model + cached blocks before loading the next;
            # otherwise reserved-but-unallocated segments accumulate across the 20
            # fold models and fragment the card into an OOM mid-run.
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        agg = aggregate_across_folds(per_fold, ["recall", "f1", "iou"], ci=95.0)
        row = {"variant": v.id, "n_folds": len(per_fold), "n_images": len(samples)}
        for m in ("recall", "f1", "iou"):
            row[f"{m}_mean"] = agg[m]["mean"]
            row[f"{m}_std"] = agg[m]["std"]
            row[f"{m}_ci_lo"] = agg[m]["ci_lo"]
            row[f"{m}_ci_hi"] = agg[m]["ci_hi"]
        summary_rows.append(row)
        print(f"  {v.id}: recall={row['recall_mean']:.3f}+/-{row['recall_std']:.3f} "
              f"f1={row['f1_mean']:.3f}+/-{row['f1_std']:.3f} "
              f"iou={row['iou_mean']:.3f}+/-{row['iou_std']:.3f} (n_folds={row['n_folds']})")

    # write outputs
    with (RESULTS / "TJII2021_ablation_folds.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "fold", "recall", "f1", "iou", "f1_opt_threshold"])
        w.writeheader()
        w.writerows(fold_rows)
    if summary_rows:
        with (RESULTS / "TJII2021_ablation.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    with (RESULTS / "TJII2021_ablation_imageci.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "threshold", "metric", "ci_lo", "mean", "ci_hi"])
        w.writeheader()
        w.writerows(image_ci_rows)
    print(f"\nWrote ablation results to {RESULTS}/TJII2021_ablation*.csv")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
