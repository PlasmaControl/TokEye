from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from tokeye.extra.eval.silbidopy.eval import Metrics
from tokeye.extra.eval.sweep import PRSweep
from tokeye.models.big_tf_unet.config_big_tf_unet import BigTFUNetConfig
from tokeye.models.big_tf_unet.model_big_tf_unet import BigTFUNetModel

device = "cuda" if torch.cuda.is_available() else "cpu"

PRIMARY_THRESHOLD = 0.5
PR_SWEEP_THRESHOLDS = [
    0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
]

MEAN = 17.84620821169868
STD = 25.016818830630463

root_path = Path("/scratch/gpfs/nc1514/tokeye")
data_path = root_path / "data" / "eval" / "TJII2021"
candidate_weights = [
    Path("/scratch/gpfs/nc1514/aemodes/model/big_mode_v1-5_weights.pt"),
    Path("/scratch/gpfs/nc1514/TokEye/model/big_mode_v1-5_weights.pt"),
    root_path / "model" / "big_mode_v1-5_weights.pt",
]
weights_path = next(
    (p for p in candidate_weights if p.exists()), candidate_weights[0]
)
results_dir = root_path / "data" / "eval" / "results"
output_main = results_dir / "TJII2021.csv"
output_pr_sweep = results_dir / "TJII2021_pr_sweep.csv"


def remap_legacy_state_dict(sd: dict) -> dict:
    idx_map = {"0": "0", "1": "1", "4": "3", "5": "4"}
    out = {}
    for k, v in sd.items():
        nk = k.replace(".double_conv.", ".conv.").replace(".maxpool_conv.1.", ".down.1.")
        parts = nk.split(".")
        for i, p in enumerate(parts):
            if p == "conv" and i + 1 < len(parts) and parts[i + 1] in idx_map:
                parts[i + 1] = idx_map[parts[i + 1]]
                break
        out[".".join(parts)] = v
    return out


def build_model() -> BigTFUNetModel:
    cfg = BigTFUNetConfig(
        in_channels=1, out_channels=2, num_layers=5,
        first_layer_size=32, dropout_rate=0.0,
    )
    model = BigTFUNetModel(cfg)
    sd = remap_legacy_state_dict(
        torch.load(weights_path, weights_only=True, map_location="cpu")
    )
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def main():
    print(f"device:  {device}")
    print(f"weights: {weights_path}")
    model = build_model()
    print("model loaded")

    metrics = Metrics(device="cpu")
    pr_sweep = PRSweep(PR_SWEEP_THRESHOLDS, track_per_image=True)
    shotns = [name.stem.split("_")[1] for name in data_path.glob("input/*.png")]
    print(f"{len(shotns)} shots")

    for shotn in tqdm(shotns):
        try:
            input_path = data_path / "input" / f"spectrogram_{shotn}.png"
            gt_path = data_path / "gt" / f"spectrogram_{shotn}.png"
            spec = np.array(Image.open(input_path).convert("L"))
            ann = np.array(Image.open(gt_path).convert("L"))
            spec = np.flip(spec, axis=0).copy()
            ann = np.flip(ann, axis=0).copy()
            spec = (spec - MEAN) / STD
            ann = (ann // 255).astype(np.float32)

            spec_t = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(ann).unsqueeze(0).unsqueeze(0).float()

            with torch.no_grad():
                out = model(spec_t)[0]
            sig = torch.sigmoid(out[:, 0:1]).cpu()

            metrics.update(sig > PRIMARY_THRESHOLD, mask_t)
            pr_sweep.update(sig, mask_t)
        except Exception as e:
            print(f"Error processing shot {shotn}: {e}")
            continue

    scores = {k: float(v) for k, v in metrics.compute().items()}
    results_dir.mkdir(parents=True, exist_ok=True)
    with output_main.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scores.keys()))
        w.writeheader()
        w.writerow(scores)

    with output_pr_sweep.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=[
                "dataset", "threshold", "precision", "recall",
                "f1", "tp", "fp", "fn",
            ],
        )
        w.writeheader()
        for row in pr_sweep.rows("TJII2021", key_field="dataset"):
            w.writerow(row)

    print(f"\nResults: {scores}")
    print(f"Saved to {output_main} and {output_pr_sweep}")

    # Bootstrap CIs + F1-optimal threshold
    print("Computing bootstrap CIs...")
    ci = pr_sweep.bootstrap_ci(n_iter=1000, ci=95.0)
    ci_path = results_dir / "TJII2021_pixel_ci.csv"
    with ci_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=[
                "dataset", "threshold", "metric",
                "ci_lo", "mean", "ci_hi",
            ],
        )
        w.writeheader()
        for thr, mdict in ci.items():
            for metric_name, (lo, mean, hi) in mdict.items():
                w.writerow({
                    "dataset": "TJII2021", "threshold": thr,
                    "metric": metric_name,
                    "ci_lo": lo, "mean": mean, "ci_hi": hi,
                })
    print(f"Saved bootstrap CIs to {ci_path}")

    sweep_rows = list(pr_sweep.rows("TJII2021", key_field="dataset"))
    best_i = max(range(len(sweep_rows)), key=lambda j: sweep_rows[j]["f1"])
    best = sweep_rows[best_i]
    opt_metrics = pr_sweep.metrics_at_threshold(best_i)
    f1_opt_path = results_dir / "TJII2021_f1_optimal.csv"
    with f1_opt_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=[
                "dataset", "f1_optimal_threshold",
                "precision_at_opt", "recall_at_opt", "f1_at_opt",
                "iou_global_at_opt", "iou_per_image_mean_at_opt",
                "dice_per_image_mean_at_opt",
            ],
        )
        w.writeheader()
        w.writerow({
            "dataset": "TJII2021",
            "f1_optimal_threshold": best["threshold"],
            "precision_at_opt": best["precision"],
            "recall_at_opt": best["recall"],
            "f1_at_opt": best["f1"],
            "iou_global_at_opt": opt_metrics.get("iou_global", 0.0),
            "iou_per_image_mean_at_opt": opt_metrics.get("iou_per_image_mean", 0.0),
            "dice_per_image_mean_at_opt": opt_metrics.get("dice_per_image_mean", 0.0),
        })
    print(f"F1-opt: thr={best['threshold']} f1={best['f1']:.3f} "
          f"iou_glob={opt_metrics.get('iou_global', 0):.3f} "
          f"iou_pi={opt_metrics.get('iou_per_image_mean', 0):.3f}")


if __name__ == "__main__":
    main()
