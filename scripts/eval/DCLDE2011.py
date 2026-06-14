from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter
from tqdm.auto import tqdm

from tokeye.extra.eval.silbidopy.data import AudioTonalDataset
from tokeye.extra.eval.silbidopy.eval import Metrics
from tokeye.extra.eval.sweep import (
    DetectionAccumulator,
    PRSweep,
    gt_components,
    predicted_components,
)
from tokeye.models.big_tf_unet.config_big_tf_unet import BigTFUNetConfig
from tokeye.models.big_tf_unet.model_big_tf_unet import BigTFUNetModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

PRIMARY_THRESHOLD = 0.5
PR_SWEEP_THRESHOLDS = [
    0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
]
DETECTION_IOU_THRESHOLDS = [0.1, 0.25, 0.5]
# Dolphin whistles are thin contour lines; sigmoid output has many isolated
# noise pixels. A small 2-D median filter on the sigmoid before thresholding
# suppresses these without affecting the continuous whistle line. Disable by
# setting MEDIAN_FILTER_SIZE = 0.
MEDIAN_FILTER_SIZE = 3


class Processor:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, x):
        return (x - self.mean) / self.std


STATS = {
    "Delphinus capensis": {"mean": 0.7080333218912784, "std": 0.051389602618240104},
    "Delphinus delphis": {"mean": 0.6542849744749529, "std": 0.049230019876810985},
    "Peponocephala electra": {"mean": 0.7745788766427888, "std": 0.06212775435505397},
    "StenellaLongirostrisLongirostris": {"mean": 0.7827474513079127, "std": 0.05873317039496551},
    "Tursiops truncatus-SoCal": {"mean": 0.3825516525048606, "std": 0.06272170328810193},
}


root_path = Path("/scratch/gpfs/nc1514/tokeye")
data_base = root_path / "data" / "eval" / "DCLDE2011"
candidate_weights = [
    Path("/scratch/gpfs/nc1514/aemodes/model/big_mode_v1-5_weights.pt"),
    Path("/scratch/gpfs/nc1514/TokEye/model/big_mode_v1-5_weights.pt"),
    root_path / "model" / "big_mode_v1-5_weights.pt",
]
weights_path = next(
    (p for p in candidate_weights if p.exists()), candidate_weights[0]
)
results_dir = root_path / "data" / "eval" / "results"
output_main = results_dir / "DCLDE2011.csv"
output_pr_sweep = results_dir / "DCLDE2011_pr_sweep.csv"
output_detection = results_dir / "DCLDE2011_detection.csv"


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
    print(f"weights: {weights_path}")
    model = build_model()
    print("model loaded")

    pixel_rows, pr_rows, detection_rows = [], [], []
    ci_rows: list[dict] = []
    f1_opt_rows: list[dict] = []

    for species_name, stats in STATS.items():
        print(f"\n=== {species_name} ===")
        try:
            metrics = Metrics(device="cpu")
            pr_sweep = PRSweep(PR_SWEEP_THRESHOLDS, track_per_image=True)
            detection = DetectionAccumulator(DETECTION_IOU_THRESHOLDS, n_classes=1)

            data_dir = data_base / species_name
            dataset = AudioTonalDataset(
                data_dir, data_dir,
                annotation_extension="ann",
                time_patch_frames=250, freq_patch_frames=250,
                post_processing_function=Processor(mean=stats["mean"], std=stats["std"]),
            )

            for i in tqdm(range(len(dataset))):
                spec, ann = dataset[i]
                spec = np.flip(spec, axis=0).copy()
                ann = np.flip(ann, axis=0).copy()

                spec_t = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0).to(device)
                mask_t = torch.from_numpy(ann).unsqueeze(0).unsqueeze(0).float()

                with torch.no_grad():
                    out = model(spec_t)[0]
                sig = torch.sigmoid(out[:, 0:1]).cpu()

                if MEDIAN_FILTER_SIZE > 0:
                    sig_np = sig.numpy()
                    sig_np = median_filter(
                        sig_np,
                        size=(1, 1, MEDIAN_FILTER_SIZE, MEDIAN_FILTER_SIZE),
                    )
                    sig = torch.from_numpy(sig_np)

                metrics.update(sig > PRIMARY_THRESHOLD, mask_t)
                pr_sweep.update(sig, mask_t)

                gt_boxes = [(0, *b) for b in gt_components(ann)]
                pred_comps = predicted_components(
                    sig.numpy()[0, 0],
                    score_thr=PRIMARY_THRESHOLD,
                    closing_radius=1,
                    min_area=8,
                    score_mode="mean",
                )
                detection.add_image(pred_comps, gt_boxes)

            scores = {k: float(v) for k, v in metrics.compute().items()}
            scores["species"] = species_name
            pixel_rows.append(scores)

            for row in pr_sweep.rows(species_name, key_field="species"):
                pr_rows.append(row)

            for iou_thr, det in detection.compute().items():
                detection_rows.append({"species": species_name, "iou_threshold": iou_thr, **det})

            print(f"  pixel: {scores}")
            for iou_thr, det in detection.compute().items():
                print(f"  det@IoU{iou_thr}: AP={det['ap']:.3f} P={det['precision']:.3f} R={det['recall']:.3f}")

            # Bootstrap CIs
            ci = pr_sweep.bootstrap_ci(n_iter=500, ci=95.0)
            for thr, mdict in ci.items():
                for metric_name, (lo, mean, hi) in mdict.items():
                    ci_rows.append({
                        "species": species_name, "threshold": thr,
                        "metric": metric_name,
                        "ci_lo": lo, "mean": mean, "ci_hi": hi,
                    })

            # F1-optimal threshold for this species
            sweep_rows = list(pr_sweep.rows(species_name, key_field="species"))
            best_i = max(range(len(sweep_rows)), key=lambda j: sweep_rows[j]["f1"])
            best = sweep_rows[best_i]
            opt_metrics = pr_sweep.metrics_at_threshold(best_i)
            f1_opt_rows.append({
                "species": species_name,
                "f1_optimal_threshold": best["threshold"],
                "precision_at_opt": best["precision"],
                "recall_at_opt": best["recall"],
                "f1_at_opt": best["f1"],
                "iou_global_at_opt": opt_metrics.get("iou_global", 0.0),
                "iou_per_image_mean_at_opt": opt_metrics.get("iou_per_image_mean", 0.0),
                "dice_per_image_mean_at_opt": opt_metrics.get("dice_per_image_mean", 0.0),
            })

        except Exception as e:
            print(f"error processing {species_name}: {e}")

    results_dir.mkdir(parents=True, exist_ok=True)

    if pixel_rows:
        fieldnames = ["species"] + [k for k in pixel_rows[0] if k != "species"]
        with output_main.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(pixel_rows)

    with output_pr_sweep.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["species", "threshold", "precision", "recall", "f1", "tp", "fp", "fn"])
        w.writeheader()
        w.writerows(pr_rows)

    with output_detection.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["species", "iou_threshold", "ap", "precision", "recall", "n_pred", "n_tp", "n_fp", "n_gt"])
        w.writeheader()
        w.writerows(detection_rows)

    ci_path = results_dir / "DCLDE2011_pixel_ci.csv"
    with ci_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=[
                "species", "threshold", "metric",
                "ci_lo", "mean", "ci_hi",
            ],
        )
        w.writeheader()
        w.writerows(ci_rows)

    f1_opt_path = results_dir / "DCLDE2011_f1_optimal.csv"
    with f1_opt_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=[
                "species", "f1_optimal_threshold",
                "precision_at_opt", "recall_at_opt", "f1_at_opt",
                "iou_global_at_opt", "iou_per_image_mean_at_opt",
                "dice_per_image_mean_at_opt",
            ],
        )
        w.writeheader()
        w.writerows(f1_opt_rows)

    print(f"\nResults written to {results_dir}")


if __name__ == "__main__":
    main()
