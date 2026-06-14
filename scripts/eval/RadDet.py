from __future__ import annotations

import contextlib
import csv
import io
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from tokeye.extra.eval.sweep import (
    DetectionAccumulator,
    PRSweep,
    merge_boxes_time,
    predicted_components,
)
from tokeye.models.big_tf_unet.config_big_tf_unet import BigTFUNetConfig
from tokeye.models.big_tf_unet.model_big_tf_unet import BigTFUNetModel

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

PRIMARY_THRESHOLD = 0.4
PR_SWEEP_THRESHOLDS = [
    0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
]
DETECTION_IOU_THRESHOLDS = [
    0.1, 0.25, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
]
COCO_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 95.0

# RadDet 128/256 variants are inferred on bilinearly upsampled inputs
# (x4 / x2); 512 runs native, like TJ-II / DCLDE / DIII-D (state the
# per-size factors in Methods). Rationale: the model's sigmoid halo is
# roughly constant in model pixels while RadDet GT boxes hug the signal
# core (median 11x6 px at 128^2), so native-resolution component bounding
# boxes are ~2-4x inflated and box IoU saturates near gt_area/pred_area
# ~ 0.27. All reported metrics come from one forward pass per variant:
# pixel metrics from the sigmoid resampled back to the native annotation
# grid, boxes from the inference grid via (1) component extraction,
# (2) shrink to the energetic core (>= alpha * component max sigmoid,
# FWHM-style), (3) merge of boxes overlapping in frequency across small
# time gaps (joins pulse-train fragments into the train-level detection
# RadDet annotates). Upsampling trades pixel agreement against RadDet's
# box-rasterised masks (tighter masks fill less box interior) for box
# localization; the factors above balance the two metric families.
# Alphas selected on the val split (.tmp/raddet_smoke.py).
DETECTION_THRESHOLD = 0.5
DETECTION_MIN_AREA_FRACTION = 0.0005  # of native H * W
DETECTION_SCORE_MODE = "mean"
# Every variant is resampled to an effective 512x512 before inference.
DETECTION_UPSAMPLE = {128: 4, 256: 2, 512: 1}  # native H -> upsample factor
DETECTION_CORE_ALPHA = {4: 0.75, 2: 0.9, 1: 0.85}  # factor -> alpha (val-tuned)
DETECTION_MERGE_GAP_FRACTION = 0.10  # of native W
DETECTION_MERGE_FREQ_OVERLAP = 0.5

FLIP_FREQ_AXIS = True
INVERT_INTENSITY = True
UN_FFT_SHIFT = True
USE_BOTH_CHANNELS = True
TTA_TIME_FLIP = True

MAX_PER_VARIANT: int | None = None
MAX_PIXELS_PER_BATCH = 256 * 256 * 256  # cap applies to the upsampled pass
SPLIT = "test"  # "val" for hyperparameter checks

VARIANTS = [
    "RadDet40k128HW001Tv2",
    "RadDet40k128HW009Tv2",
    "RadDet40k256HW001Tv2",
    "RadDet40k256HW009Tv2",
    "RadDet40k512HW001Tv2",
    "RadDet40k512HW009Tv2",
]

CLASS_NAMES = [
    "Rect", "Barker", "Frank", "P1", "P2", "P3", "P4",
    "Px", "ZadoffChu", "LFM", "FMCW",
]

root_path = Path("/scratch/gpfs/nc1514/tokeye")
data_path = root_path / "data" / "eval" / "RadDet"
candidate_weights = [
    Path("/scratch/gpfs/nc1514/aemodes/model/big_mode_v1-5_weights.pt"),
    Path("/scratch/gpfs/nc1514/TokEye/model/big_mode_v1-5_weights.pt"),
    root_path / "model" / "big_mode_v1-5_weights.pt",
]
weights_path = next(
    (p for p in candidate_weights if p.exists()), candidate_weights[0]
)
results_dir = root_path / "data" / "eval" / "results"
output_main = results_dir / "RadDet.csv"
output_pr_sweep = results_dir / "RadDet_pr_sweep.csv"
output_per_class = results_dir / "RadDet_per_class.csv"
output_per_class_ap = results_dir / "RadDet_per_class_ap.csv"
output_detection = results_dir / "RadDet_detection.csv"
output_pixel_ci = results_dir / "RadDet_pixel_ci.csv"
output_detection_ci = results_dir / "RadDet_detection_ci.csv"
output_f1_opt = results_dir / "RadDet_f1_optimal.csv"


# ---------------------------------------------------------------------------
# Model loading (legacy weight remap)
# ---------------------------------------------------------------------------


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


def build_model(device: str) -> BigTFUNetModel:
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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def parse_yolo_label(label_text: str, H: int, W: int):
    """Return list of (class_id, x0, y0, x1, y1) in pixel coords."""
    boxes = []
    for line in label_text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = (float(p) for p in parts[1:])
        x0 = max(0, int((cx - bw / 2) * W))
        x1 = min(W - 1, int((cx + bw / 2) * W))
        y0 = max(0, int((cy - bh / 2) * H))
        y1 = min(H - 1, int((cy + bh / 2) * H))
        boxes.append((cls, x0, y0, x1, y1))
    return boxes


def rasterize_box_list(boxes, H, W) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    for _cls, x0, y0, x1, y1 in boxes:
        mask[y0 : y1 + 1, x0 : x1 + 1] = 1
    return mask


def transform_boxes(boxes, H, W, do_flip: bool, do_roll: bool):
    """Apply same flip + roll the spec/mask got, in pixel coords."""
    if not (do_flip or do_roll):
        return boxes
    out = []
    dc = H // 2 - 1
    for cls, x0, y0, x1, y1 in boxes:
        if do_flip:
            y0_n = (H - 1) - y1
            y1_n = (H - 1) - y0
            y0, y1 = y0_n, y1_n
        if do_roll and 0 < dc < H - 1:
            # spec is np.roll(arr, -dc, axis=0): rows shift up by dc, wrap.
            # mask uses same roll but BEFORE the flip in our pipeline. The
            # combined effect on a box's (y0, y1) range needs to handle wrap.
            # Simpler: roll the rasterised box mask directly later, and
            # build per-image box list after rolling. For the wrap corner,
            # we accept that boxes which span the DC row split into two.
            pass
        out.append((cls, x0, y0, x1, y1))
    return out


def iter_test_pairs(variant: str):
    parts = sorted(data_path.glob(f"{variant}.tar.part-*"))
    if not parts:
        raise FileNotFoundError(f"no tar parts for {variant}")
    proc = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    pi: dict[str, bytes] = {}
    pl: dict[str, bytes] = {}
    try:
        tf = tarfile.open(fileobj=proc.stdout, mode="r|")
        for m in tf:
            if not m.isfile():
                continue
            name = m.name
            if f"/images/{SPLIT}/" in name and name.endswith(".png"):
                stem = Path(name).stem
                data = tf.extractfile(m).read()
                if stem in pl:
                    yield stem, data, pl.pop(stem)
                else:
                    pi[stem] = data
            elif f"/labels/{SPLIT}/" in name and name.endswith(".txt"):
                stem = Path(name).stem
                data = tf.extractfile(m).read()
                if stem in pi:
                    yield stem, pi.pop(stem), data
                else:
                    pl[stem] = data
        for stem, png in pi.items():
            yield stem, png, b""
        if pl:
            print(f"[{variant}] dropped {len(pl)} labels with no image")
    finally:
        with contextlib.suppress(Exception):
            proc.stdout.close()
        proc.wait()


def preprocess(png_bytes: bytes, lbl_bytes: bytes):
    """Returns (spec, mask, boxes_in_eval_frame).

    boxes are in the same coord frame as spec/mask after flip+roll, with
    class id retained for per-class breakdown.
    """
    spec = np.array(Image.open(io.BytesIO(png_bytes)).convert("L"))
    H, W = spec.shape
    boxes_orig = parse_yolo_label(lbl_bytes.decode().strip(), H, W)
    mask = rasterize_box_list(boxes_orig, H, W)

    if FLIP_FREQ_AXIS:
        spec = np.flip(spec, axis=0).copy()
        mask = np.flip(mask, axis=0).copy()

    spec = spec.astype(np.float32)
    if INVERT_INTENSITY:
        spec = 255.0 - spec
    dc = H // 2 - 1
    if 0 < dc < H - 1:
        spec[dc] = 0.5 * (spec[dc - 1] + spec[dc + 1])
    if UN_FFT_SHIFT and 0 < dc < H - 1:
        spec = np.roll(spec, -dc, axis=0)
        mask = np.roll(mask, -dc, axis=0)
    spec = spec - np.median(spec)
    std = spec.std()
    spec = spec / (std if std > 0 else 1.0)
    mask = mask.astype(np.float32)

    # Boxes in eval frame (after flip + roll). Per-class detection metric
    # works on box list, so transform them analytically.
    boxes_eval = []
    for cls, x0, y0, x1, y1 in boxes_orig:
        if FLIP_FREQ_AXIS:
            y0n = (H - 1) - y1
            y1n = (H - 1) - y0
            y0, y1 = y0n, y1n
        if UN_FFT_SHIFT and 0 < dc < H - 1:
            # roll(-dc) on axis 0 (rows) shifts row r → row (r - dc) mod H.
            # A box [y0, y1] either translates as a single block or wraps.
            shift = -dc
            y0n = (y0 + shift) % H
            y1n = (y1 + shift) % H
            if y0n <= y1n:
                boxes_eval.append((cls, x0, y0n, x1, y1n))
            else:
                # Wraps: split into two pieces.
                boxes_eval.append((cls, x0, y0n, x1, H - 1))
                boxes_eval.append((cls, x0, 0, x1, y1n))
            continue
        boxes_eval.append((cls, x0, y0, x1, y1))
    return spec, mask, boxes_eval


# ---------------------------------------------------------------------------
# Pixel-level metric accumulators
# ---------------------------------------------------------------------------


class FastMetrics:
    """Pixel metrics at the primary threshold (confusion-matrix based).

    Computes global P/R/F1/IoU/Dice from a running confusion matrix to avoid
    the dtype-dependent bug in torchmetrics MeanIoU + GeneralizedDiceScore.
    """

    def __init__(self, track_per_image: bool = True):
        self.track_per_image = track_per_image
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.per_image_iou: list[float] = []
        self.per_image_dice: list[float] = []

    def update(self, output, ann):
        pred = output if output.dtype == torch.bool else (output > 0.5)
        target = ann if ann.dtype == torch.bool else (ann > 0.5)
        self.tp += int((pred & target).sum().item())
        self.fp += int((pred & ~target).sum().item())
        self.fn += int((~pred & target).sum().item())
        self.tn += int((~pred & ~target).sum().item())
        if self.track_per_image:
            B = pred.shape[0]
            pred_b = pred.reshape(B, -1)
            target_b = target.reshape(B, -1)
            for i in range(B):
                tp_i = int((pred_b[i] & target_b[i]).sum().item())
                fp_i = int((pred_b[i] & ~target_b[i]).sum().item())
                fn_i = int((~pred_b[i] & target_b[i]).sum().item())
                if tp_i + fn_i == 0 and tp_i + fp_i == 0:
                    continue
                denom_iou = tp_i + fp_i + fn_i
                self.per_image_iou.append(
                    tp_i / denom_iou if denom_iou > 0 else 0.0
                )
                denom_dice = 2 * tp_i + fp_i + fn_i
                self.per_image_dice.append(
                    2 * tp_i / denom_dice if denom_dice > 0 else 0.0
                )

    def compute(self):
        tp, fp, fn = self.tp, self.fp, self.fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        iou_global = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice_global = (
            2 * tp / (2 * tp + fp + fn)
            if (2 * tp + fp + fn) > 0
            else 0.0
        )
        out = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou_global),
            "generalized_dice_score": float(dice_global),
        }
        if self.per_image_iou:
            out["iou_per_image_mean"] = float(
                np.mean(self.per_image_iou)
            )
            out["dice_per_image_mean"] = float(
                np.mean(self.per_image_dice)
            )
        return out


# ---------------------------------------------------------------------------
# Inference + accumulation
# ---------------------------------------------------------------------------


def _forward_sig(model, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        o = model(x)[0]
    s0 = torch.sigmoid(o[:, 0:1])
    if USE_BOTH_CHANNELS:
        s1 = torch.sigmoid(o[:, 1:2])
        return torch.maximum(s0, s1)
    return s0


def flush_batch(specs, masks, boxes_per_img, model,
                pixel_metrics: FastMetrics,
                pr_sweep: PRSweep,
                detection: DetectionAccumulator):
    if not specs:
        return
    spec_t = torch.from_numpy(np.stack(specs)).unsqueeze(1).float().to(device)
    mask_t = torch.from_numpy(np.stack(masks)).unsqueeze(1).float()

    # Single inference pass at upsampled resolution (see settings block).
    # All metrics derive from this one forward pass: detection boxes from
    # the upsampled grid (scaled back to native coords), pixel metrics from
    # the sigmoid bilinearly resampled to the native annotation grid.
    H, W = spec_t.shape[-2], spec_t.shape[-1]
    f = DETECTION_UPSAMPLE.get(H, 1)
    alpha = DETECTION_CORE_ALPHA[f]
    if f > 1:
        spec_up = torch.nn.functional.interpolate(
            spec_t, size=(f * H, f * W), mode="bilinear", align_corners=False,
        )
    else:
        spec_up = spec_t
    sig_up = _forward_sig(model, spec_up)
    if TTA_TIME_FLIP:
        sig_up = 0.5 * (sig_up + _forward_sig(model, spec_up.flip(-1)).flip(-1))

    if f > 1:
        sig_native = torch.nn.functional.interpolate(
            sig_up, size=(H, W), mode="bilinear", align_corners=False,
        ).cpu()
    else:
        sig_native = sig_up.cpu()
    pred_primary = sig_native > PRIMARY_THRESHOLD
    pixel_metrics.update(pred_primary, mask_t)
    pr_sweep.update(sig_native, mask_t)

    sig_up_np = sig_up.cpu().numpy()[:, 0]  # (B, f*H, f*W)

    min_area = max(4, int(DETECTION_MIN_AREA_FRACTION * H * W))
    merge_gap = int(DETECTION_MERGE_GAP_FRACTION * W)
    for i in range(sig_up_np.shape[0]):
        comps = predicted_components(
            sig_up_np[i],
            score_thr=DETECTION_THRESHOLD,
            min_area=f * f * min_area,
            score_mode=DETECTION_SCORE_MODE,
            core_alpha=alpha,
        )
        comps = [
            (s, x0 / f, y0 / f, x1 / f, y1 / f)
            for s, x0, y0, x1, y1 in comps
        ]
        comps = merge_boxes_time(
            comps, merge_gap, freq_overlap=DETECTION_MERGE_FREQ_OVERLAP,
        )
        detection.add_image(comps, boxes_per_img[i])


# ---------------------------------------------------------------------------
# Variant scoring
# ---------------------------------------------------------------------------


def score_variant(variant: str, model):
    pixel_metrics = FastMetrics()
    pr_sweep = PRSweep(PR_SWEEP_THRESHOLDS, track_per_image=True)
    pr_key = variant
    detection = DetectionAccumulator(
        DETECTION_IOU_THRESHOLDS, n_classes=len(CLASS_NAMES),
        track_per_image=True,
        bootstrap_iou_thresholds=[0.1, 0.5],
    )

    n = 0
    specs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    boxes_per_img: list[list] = []
    pbar = tqdm(desc=variant, unit="img", mininterval=2.0)
    batch_size: int | None = None
    for stem, png_bytes, lbl_bytes in iter_test_pairs(variant):
        if MAX_PER_VARIANT is not None and n >= MAX_PER_VARIANT:
            break
        try:
            spec, mask, boxes = preprocess(png_bytes, lbl_bytes)
        except Exception as e:
            print(f"Error preprocessing {variant}/{stem}: {e}")
            continue
        if batch_size is None:
            H, W = spec.shape
            f = DETECTION_UPSAMPLE.get(H, 2)
            batch_size = max(1, MAX_PIXELS_PER_BATCH // (H * W * f * f))
            print(f"[{variant}] H={H} W={W} up={f}x batch_size={batch_size}")
        specs.append(spec)
        masks.append(mask)
        boxes_per_img.append(boxes)
        n += 1
        pbar.update(1)
        if len(specs) >= batch_size:
            try:
                flush_batch(
                    specs, masks, boxes_per_img, model,
                    pixel_metrics, pr_sweep, detection,
                )
            except Exception as e:
                print(f"Error in batch in {variant}: {e}")
            specs.clear()
            masks.clear()
            boxes_per_img.clear()
    if specs:
        try:
            flush_batch(
                specs, masks, boxes_per_img, model,
                pixel_metrics, pr_sweep, detection,
            )
        except Exception as e:
            print(f"Error in final batch for {variant}: {e}")
    pbar.close()
    print(f"[{variant}] processed {n} images, {detection.gt_total} GT boxes")
    print(f"[{variant}] computing bootstrap CIs (n={N_BOOTSTRAP})...")
    pixel_ci = pr_sweep.bootstrap_ci(n_iter=N_BOOTSTRAP, ci=BOOTSTRAP_CI)
    detection_ci = detection.bootstrap_ap(
        iou_thresholds=[0.1, 0.5], n_iter=N_BOOTSTRAP, ci=BOOTSTRAP_CI,
    )
    coco_map = detection.coco_map(COCO_IOU_THRESHOLDS)
    pixel = pixel_metrics.compute()  # already includes f1, iou (global), dice

    sweep_rows = list(pr_sweep.rows(pr_key))
    best_i = max(range(len(sweep_rows)), key=lambda j: sweep_rows[j]["f1"])
    f1_opt_metrics = pr_sweep.metrics_at_threshold(best_i)
    f1_opt_metrics["threshold"] = sweep_rows[best_i]["threshold"]

    return {
        "pixel": pixel,
        "pr_sweep": sweep_rows,
        "pixel_ci": pixel_ci,
        "detection": detection.compute(),
        "detection_ci": detection_ci,
        "coco_map": coco_map,
        "per_class": detection.per_class_recall(iou_thr=0.5),
        "per_class_ap": detection.per_class_ap(iou_thr=0.5),
        "f1_opt": f1_opt_metrics,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_results(all_results: dict[str, dict]):
    results_dir.mkdir(parents=True, exist_ok=True)

    with output_main.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=[
                "variant", "precision", "recall", "f1",
                "iou", "generalized_dice_score",
                "iou_per_image_mean", "dice_per_image_mean",
                "coco_map",
            ],
        )
        w.writeheader()
        for variant, res in all_results.items():
            row = {"variant": variant, **res["pixel"], "coco_map": res["coco_map"]}
            w.writerow(row)

    with output_pr_sweep.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=[
                "variant", "threshold", "precision", "recall",
                "f1", "tp", "fp", "fn",
            ],
        )
        w.writeheader()
        for variant, res in all_results.items():
            for row in res["pr_sweep"]:
                w.writerow(row)

    with output_detection.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=[
                "variant", "iou_threshold", "ap",
                "precision", "recall", "n_pred",
                "n_tp", "n_fp", "n_gt",
            ],
        )
        w.writeheader()
        for variant, res in all_results.items():
            for iou_thr, det in res["detection"].items():
                w.writerow({
                    "variant": variant, "iou_threshold": iou_thr, **det,
                })

    with output_per_class.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=[
                "variant", "class_id", "class_name",
                "n_gt", "n_tp", "recall",
            ],
        )
        w.writeheader()
        for variant, res in all_results.items():
            for cls_id, stats in res["per_class"].items():
                w.writerow({
                    "variant": variant,
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES[cls_id],
                    **stats,
                })

    with output_pixel_ci.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=[
                "variant", "threshold", "metric",
                "ci_lo", "mean", "ci_hi",
            ],
        )
        w.writeheader()
        for variant, res in all_results.items():
            for thr, mdict in res.get("pixel_ci", {}).items():
                for metric_name, (lo, mean, hi) in mdict.items():
                    w.writerow({
                        "variant": variant, "threshold": thr,
                        "metric": metric_name,
                        "ci_lo": lo, "mean": mean, "ci_hi": hi,
                    })

    with output_detection_ci.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=[
                "variant", "iou_threshold", "ci_lo", "mean", "ci_hi",
            ],
        )
        w.writeheader()
        for variant, res in all_results.items():
            for iou_thr, (lo, mean, hi) in res.get("detection_ci", {}).items():
                w.writerow({
                    "variant": variant, "iou_threshold": iou_thr,
                    "ci_lo": lo, "mean": mean, "ci_hi": hi,
                })

    with output_per_class_ap.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=[
                "variant", "class_id", "class_name",
                "n_gt", "n_tp", "ap", "precision", "recall",
            ],
        )
        w.writeheader()
        for variant, res in all_results.items():
            for cls_id, stats in res.get("per_class_ap", {}).items():
                w.writerow({
                    "variant": variant,
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES[cls_id],
                    **stats,
                })

    with output_f1_opt.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=[
                "variant", "f1_optimal_threshold",
                "precision_at_opt", "recall_at_opt", "f1_at_opt",
                "iou_global_at_opt", "iou_per_image_mean_at_opt",
                "dice_per_image_mean_at_opt",
            ],
        )
        w.writeheader()
        for variant, res in all_results.items():
            opt = res["f1_opt"]
            w.writerow({
                "variant": variant,
                "f1_optimal_threshold": opt["threshold"],
                "precision_at_opt": opt["precision"],
                "recall_at_opt": opt["recall"],
                "f1_at_opt": opt["f1"],
                "iou_global_at_opt": opt.get("iou_global", 0.0),
                "iou_per_image_mean_at_opt": opt.get("iou_per_image_mean", 0.0),
                "dice_per_image_mean_at_opt": opt.get("dice_per_image_mean", 0.0),
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"device:  {device}")
    print(f"weights: {weights_path}")
    model = build_model(device)
    print("model loaded")

    all_results: dict[str, dict] = {}
    for variant in VARIANTS:
        print(f"\n=== {variant} ===")
        all_results[variant] = score_variant(variant, model)
        write_results(all_results)
        res = all_results[variant]
        print(f"[{variant}] pixel: {res['pixel']}")
        print(f"[{variant}] COCO mAP@[0.5:0.95]: {res['coco_map']:.4f}")
        for iou_thr, det in res["detection"].items():
            print(f"  det@IoU{iou_thr}: AP={det['ap']:.3f} "
                  f"P={det['precision']:.3f} R={det['recall']:.3f}")
        for iou_thr, (lo, m, hi) in res.get("detection_ci", {}).items():
            print(f"  det@IoU{iou_thr} AP {BOOTSTRAP_CI:.0f}% CI: "
                  f"[{lo:.3f}, {hi:.3f}] (mean {m:.3f})")

    print(f"\nresults dir: {results_dir}")


if __name__ == "__main__":
    main()
