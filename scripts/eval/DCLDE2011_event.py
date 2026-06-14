"""Event-level detection scoring + false-positive morphology triage for DCLDE.

Pixel IoU (~0.14) and box-IoU AP (~0.005) both understate whistle detection:
box-IoU is degenerate for thin diagonal contours, and the model predicts a
filled envelope where the annotation is a sparse contour. This script scores
detection at the *event* (connected-component) level with contour-aware match
criteria, at the operating point we report (512 inference, the AP-optimal
extraction threshold). It also characterises the false positives by acoustic
morphology — distinguishing tonal/whistle-like FPs (candidate unannotated
calls / convention spillover) from broadband-impulsive FPs (clicks, snaps:
interference the annotators correctly excluded) — using objective time-
frequency shape, no expert labelling.

Inference runs at 512x512 (bilinear upsample of the native 250x250 patch);
the sigmoid is resampled back to the 250 annotation grid so GT and the
evaluation grid are fixed.

Outputs:
  data/eval/results/DCLDE2011_event.csv          (event R/P/F1/AP per criterion)
  data/eval/results/DCLDE2011_fp_morphology.csv  (per-component shape + class)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_closing, median_filter
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from DCLDE2011 import (  # noqa: E402
    MEDIAN_FILTER_SIZE,
    STATS,
    Processor,
    build_model,
    data_base,
)

from tokeye.extra.eval.events import EventAccumulator, label8  # noqa: E402
from tokeye.extra.eval.silbidopy.data import AudioTonalDataset  # noqa: E402
from tokeye.extra.eval.sweep import _disk_kernel  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"

UPSAMPLE = 512          # bilinear inference resolution (native patch is 250)
OPERATING_THRESHOLD = 0.5  # AP-optimal extraction threshold at 512
CLOSING_RADIUS = 1
PRED_MIN_AREA = 8
GT_MIN_AREA = 3

EVENT_CRITERIA = [
    ("any", {"min_overlap": 1}),
    ("center", {"center_tol": 1}),
    ("coverage", {"cov_frac": 0.2}),
]

# Morphology triage thresholds (250-grid bins / frames). A whistle is narrow
# in frequency at each instant (small per-column bandwidth) even as its centre
# sweeps over time; a click/snap is broadband at each instant; a fragment is
# too short to be a call. Distributions are written out so these cut points
# are descriptive, not load-bearing.
TONAL_INST_BW = 5    # max median per-column freq extent (bins) to be "tonal"
MIN_DUR = 5          # min time frames to be a call rather than a fragment

results_dir = Path("/scratch/gpfs/nc1514/tokeye/data/eval/results")
output_event = results_dir / "DCLDE2011_event.csv"
output_morph = results_dir / "DCLDE2011_fp_morphology.csv"


def infer(model, spec: np.ndarray) -> np.ndarray:
    """Sigmoid on the native 250 grid, via 512x512 inference."""
    t = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0).to(device)
    H, W = t.shape[-2], t.shape[-1]
    t = F.interpolate(t, size=(UPSAMPLE, UPSAMPLE), mode="bilinear",
                      align_corners=False)
    with torch.no_grad():
        o = model(t)[0]
    sig = torch.sigmoid(o[:, 0:1])
    sig = F.interpolate(sig, size=(H, W), mode="bilinear", align_corners=False)
    sig = sig.cpu().numpy()
    if MEDIAN_FILTER_SIZE > 0:
        sig = median_filter(sig, size=(1, 1, MEDIAN_FILTER_SIZE, MEDIAN_FILTER_SIZE))
    return sig[0, 0]


def component_features(mask: np.ndarray) -> dict:
    """Shape descriptors for one connected component (boolean mask)."""
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    bw = y1 - y0 + 1
    dur = x1 - x0 + 1
    area = int(mask.sum())
    inst_bws, ridge = [], []
    for x in range(x0, x1 + 1):
        col = np.where(mask[:, x])[0]
        if col.size:
            inst_bws.append(int(col.max() - col.min() + 1))
            ridge.append(float(col.mean()))
    inst_bw = float(np.median(inst_bws)) if inst_bws else float(bw)
    fm_range = (max(ridge) - min(ridge)) if ridge else 0.0
    return {
        "bw": bw, "dur": dur, "area": area,
        "aspect": dur / bw, "fill": area / (bw * dur),
        "inst_bw": inst_bw, "fm_range": fm_range,
    }


def classify(f: dict) -> str:
    if f["dur"] < MIN_DUR:
        return "short"          # fragment: too brief to be a call
    if f["inst_bw"] <= TONAL_INST_BW:
        return "tonal"          # narrowband-per-instant: whistle-like
    return "broadband"          # wide-per-instant: click / snap / impulsive


def score_species(species_name: str, stats: dict, model):
    event_accs = {name: EventAccumulator(criterion=name, **kw)
                  for name, kw in EVENT_CRITERIA}
    morph_rows: list[dict] = []

    ds = AudioTonalDataset(
        data_base / species_name, data_base / species_name,
        annotation_extension="ann", time_patch_frames=250, freq_patch_frames=250,
        post_processing_function=Processor(mean=stats["mean"], std=stats["std"]),
    )
    for i in tqdm(range(len(ds)), desc=species_name, mininterval=2.0):
        spec, ann = ds[i]
        spec = np.flip(spec, axis=0).copy()
        ann = np.flip(ann, axis=0).copy()
        ann_bool = ann > 0.5
        sig = infer(model, spec)
        pred_bool = sig > OPERATING_THRESHOLD
        if pred_bool.any():
            pred_bool = binary_closing(pred_bool, structure=_disk_kernel(CLOSING_RADIUS))

        for acc in event_accs.values():
            acc.add_image(ann_bool, pred_bool, sig,
                          gt_min_area=GT_MIN_AREA, pred_min_area=PRED_MIN_AREA)

        # FP morphology: classify each predicted component as TP (overlaps any
        # GT pixel) or FP, and record its shape.
        lab, n = label8(pred_bool)
        for cid in range(1, n + 1):
            cmask = lab == cid
            if cmask.sum() < PRED_MIN_AREA:
                continue
            kind = "TP" if (cmask & ann_bool).any() else "FP"
            f = component_features(cmask)
            morph_rows.append({"species": species_name, "kind": kind,
                               "klass": classify(f), **f})

    return {name: acc.compute() for name, acc in event_accs.items()}, morph_rows


def main():
    print(f"device: {device}  inference={UPSAMPLE}x{UPSAMPLE}  "
          f"thr={OPERATING_THRESHOLD}")
    model = build_model()
    print("model loaded")
    results_dir.mkdir(parents=True, exist_ok=True)

    event_rows: list[dict] = []
    all_morph: list[dict] = []

    for species_name, stats in STATS.items():
        print(f"\n=== {species_name} ===")
        try:
            ev, morph = score_species(species_name, stats, model)
        except Exception as e:
            print(f"error processing {species_name}: {e}")
            continue
        for crit, m in ev.items():
            event_rows.append({"species": species_name, "criterion": crit, **m})
            print(f"  event[{crit:>8}]: R={m['recall']:.3f} P={m['precision']:.3f}"
                  f" F1={m['f1']:.3f} AP={m['ap']:.3f} (n_gt={m['n_gt']})")
        all_morph.extend(morph)
        fp = [r for r in morph if r["kind"] == "FP"]
        if fp:
            n = len(fp)
            frac = {k: sum(r["klass"] == k for r in fp) / n
                    for k in ("tonal", "broadband", "short")}
            print(f"  FP morphology (n={n}): tonal={frac['tonal']:.2f} "
                  f"broadband={frac['broadband']:.2f} short={frac['short']:.2f}")

    with output_event.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["species", "criterion", "ap",
                                          "precision", "recall", "f1",
                                          "n_pred", "n_tp", "n_gt"])
        w.writeheader()
        for row in event_rows:
            w.writerow({k: row.get(k, "") for k in w.fieldnames})

    with output_morph.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["species", "kind", "klass", "bw",
                                          "dur", "area", "aspect", "fill",
                                          "inst_bw", "fm_range"])
        w.writeheader()
        w.writerows(all_morph)

    # Overall triage + TP-vs-FP median shape, pooled across species.
    def summarise(kind):
        rows = [r for r in all_morph if r["kind"] == kind]
        if not rows:
            return
        n = len(rows)
        kl = {k: sum(r["klass"] == k for r in rows) / n
              for k in ("tonal", "broadband", "short")}
        med = {f: float(np.median([r[f] for r in rows]))
               for f in ("inst_bw", "dur", "aspect", "fill")}
        print(f"{kind} (n={n}): tonal={kl['tonal']:.2f} "
              f"broadband={kl['broadband']:.2f} short={kl['short']:.2f} | "
              f"median inst_bw={med['inst_bw']:.1f} dur={med['dur']:.1f} "
              f"aspect={med['aspect']:.2f} fill={med['fill']:.2f}")

    print("\n=== pooled morphology (TP = whistle-like reference) ===")
    summarise("TP")
    summarise("FP")
    print(f"\nwrote {output_event}\nwrote {output_morph}")


if __name__ == "__main__":
    main()
