"""Ablation: turn each preprocessing shim on/off and report metrics.

Runs RadDet pipeline against one variant under several preprocessing
configurations, writing results to RadDet_ablation.csv. Use a small variant
(128x128) and a sample size for fast iteration.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import RadDet  # noqa: E402

CONFIGS = [
    # (name, dict of overrides)
    ("raw", {
        "FLIP_FREQ_AXIS": False, "INVERT_INTENSITY": False,
        "UN_FFT_SHIFT": False, "USE_BOTH_CHANNELS": False,
        "TTA_TIME_FLIP": False,
    }),
    ("+flip", {
        "FLIP_FREQ_AXIS": True, "INVERT_INTENSITY": False,
        "UN_FFT_SHIFT": False, "USE_BOTH_CHANNELS": False,
        "TTA_TIME_FLIP": False,
    }),
    ("+flip+invert", {
        "FLIP_FREQ_AXIS": True, "INVERT_INTENSITY": True,
        "UN_FFT_SHIFT": False, "USE_BOTH_CHANNELS": False,
        "TTA_TIME_FLIP": False,
    }),
    ("+flip+invert+unshift", {
        "FLIP_FREQ_AXIS": True, "INVERT_INTENSITY": True,
        "UN_FFT_SHIFT": True, "USE_BOTH_CHANNELS": False,
        "TTA_TIME_FLIP": False,
    }),
    ("+flip+invert+unshift+both", {
        "FLIP_FREQ_AXIS": True, "INVERT_INTENSITY": True,
        "UN_FFT_SHIFT": True, "USE_BOTH_CHANNELS": True,
        "TTA_TIME_FLIP": False,
    }),
    ("+flip+invert+unshift+both+TTA (full)", {
        "FLIP_FREQ_AXIS": True, "INVERT_INTENSITY": True,
        "UN_FFT_SHIFT": True, "USE_BOTH_CHANNELS": True,
        "TTA_TIME_FLIP": True,
    }),
]


ABLATION_VARIANT = "RadDet40k128HW009Tv2"  # signal-rich, fast resolution
MAX_PER_VARIANT_FOR_ABLATION = None  # full variant for tightest CIs

OUTPUT = RadDet.results_dir / "RadDet_ablation.csv"


def main():
    print(f"device:  {RadDet.device}")
    model = RadDet.build_model(RadDet.device)
    print("model loaded")

    RadDet.MAX_PER_VARIANT = MAX_PER_VARIANT_FOR_ABLATION
    RadDet.N_BOOTSTRAP = 200  # cheaper for ablation; main eval uses 1000

    rows = []
    for name, overrides in CONFIGS:
        print(f"\n=== {name} ===")
        for k, v in overrides.items():
            setattr(RadDet, k, v)
        for k in ("FLIP_FREQ_AXIS", "INVERT_INTENSITY", "UN_FFT_SHIFT",
                 "USE_BOTH_CHANNELS", "TTA_TIME_FLIP"):
            print(f"  {k}={getattr(RadDet, k)}")

        t0 = time.time()
        res = RadDet.score_variant(ABLATION_VARIANT, model)
        elapsed = time.time() - t0

        det50 = res["detection"].get(0.5, {})
        det10 = res["detection"].get(0.1, {})
        coco = res["coco_map"]

        det_ci = res.get("detection_ci", {})
        ap50_ci = det_ci.get(0.5, (0.0, 0.0, 0.0))
        ap10_ci = det_ci.get(0.1, (0.0, 0.0, 0.0))

        row = {
            "config": name,
            "variant": ABLATION_VARIANT,
            "elapsed_s": round(elapsed, 1),
            "pixel_p": res["pixel"]["precision"],
            "pixel_r": res["pixel"]["recall"],
            "pixel_iou": res["pixel"]["iou"],
            "pixel_dice": res["pixel"]["generalized_dice_score"],
            "ap_iou0.1": det10.get("ap", 0.0),
            "ap_iou0.1_lo": ap10_ci[0],
            "ap_iou0.1_hi": ap10_ci[2],
            "ap_iou0.5": det50.get("ap", 0.0),
            "ap_iou0.5_lo": ap50_ci[0],
            "ap_iou0.5_hi": ap50_ci[2],
            "coco_map": coco,
        }
        rows.append(row)
        print(f"  P={row['pixel_p']:.3f} R={row['pixel_r']:.3f} "
              f"AP@0.5={row['ap_iou0.5']:.3f} mAP={coco:.3f}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nablation results: {OUTPUT}")


if __name__ == "__main__":
    main()
