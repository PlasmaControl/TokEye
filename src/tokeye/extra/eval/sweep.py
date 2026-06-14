"""Shared evaluation utilities used across the per-domain eval scripts.

`PRSweep` accumulates pixel-level confusion-matrix counts at multiple
sigmoid thresholds in a single pass. `predicted_components` and `box_iou`
are used to convert binary prediction masks into detection-style records.
`DetectionAccumulator` tracks score-ordered TP/FP records and computes
PASCAL-style AP at one or more box-IoU thresholds.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from scipy.ndimage import (
    binary_closing,
    generate_binary_structure,
    iterate_structure,
    label,
)


class PRSweep:
    """Pixel-level confusion-matrix counts at many thresholds.

    When `track_per_image=True`, per-image (TP, FP, FN) counts are stored
    so callers can bootstrap confidence intervals over the image distribution.
    """

    def __init__(self, thresholds, track_per_image: bool = False):
        self.thresholds = list(thresholds)
        n = len(self.thresholds)
        self.tp = np.zeros(n, dtype=np.int64)
        self.fp = np.zeros(n, dtype=np.int64)
        self.fn = np.zeros(n, dtype=np.int64)
        self.track_per_image = track_per_image
        self.per_image: list[np.ndarray] = []  # each entry shape (n_thr, 3)

    @torch.no_grad()
    def update(self, sigmoid: torch.Tensor, mask: torch.Tensor):
        # Vectorized across thresholds via sorted searchsorted (exact same
        # semantics as ``pred = sigmoid > thr``): ~100x faster than the per
        # (image, threshold) Python loop, which dominated multi-fold eval.
        sig = np.asarray(sigmoid.detach().cpu().numpy())
        pos = np.asarray((mask > 0.5).detach().cpu().numpy())
        thr = np.asarray(self.thresholds, dtype=np.float64)
        n_imgs = sig.shape[0]
        for img_i in range(n_imgs):
            s = sig[img_i].ravel().astype(np.float64)
            p = pos[img_i].ravel()
            pos_vals = np.sort(s[p])
            neg_vals = np.sort(s[~p])
            n_pos = pos_vals.size
            n_neg = neg_vals.size
            # count of values strictly greater than each threshold
            tp = n_pos - np.searchsorted(pos_vals, thr, side="right")
            fp = n_neg - np.searchsorted(neg_vals, thr, side="right")
            fn = n_pos - tp
            self.tp += tp.astype(np.int64)
            self.fp += fp.astype(np.int64)
            self.fn += fn.astype(np.int64)
            if self.track_per_image:
                self.per_image.append(
                    np.stack([tp, fp, fn], axis=1).astype(np.int64)
                )

    def rows(self, key: str, key_field: str = "variant"):
        for i, thr in enumerate(self.thresholds):
            tp, fp, fn = int(self.tp[i]), int(self.fp[i]), int(self.fn[i])
            p = tp / (tp + fp) if tp + fp > 0 else 0.0
            r = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
            yield {
                key_field: key, "threshold": thr,
                "precision": p, "recall": r, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn,
            }

    def metrics_at_threshold(self, thr_index: int) -> dict:
        """Global + per-image IoU/Dice/P/R/F1 at one threshold index.

        Returns dict with iou, iou_per_image_mean, dice (=F1), and the
        per-image dice mean, computed from the stored confusion matrices.
        Images with no GT positive AND no prediction are skipped for the
        per-image average (undefined for foreground IoU).
        """
        i = thr_index
        tp, fp, fn = int(self.tp[i]), int(self.fp[i]), int(self.fn[i])
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        iou_g = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        result = {
            "precision": p, "recall": r, "f1": f1,
            "iou_global": iou_g, "dice_global": f1,
        }
        if self.track_per_image and self.per_image:
            ious = []
            dices = []
            for cm in self.per_image:
                tpi, fpi, fni = (
                    int(cm[i, 0]), int(cm[i, 1]), int(cm[i, 2])
                )
                if tpi + fni == 0 and tpi + fpi == 0:
                    continue
                d_iou = tpi + fpi + fni
                ious.append(tpi / d_iou if d_iou > 0 else 0.0)
                d_dice = 2 * tpi + fpi + fni
                dices.append(2 * tpi / d_dice if d_dice > 0 else 0.0)
            if ious:
                result["iou_per_image_mean"] = float(np.mean(ious))
                result["dice_per_image_mean"] = float(np.mean(dices))
        return result

    def bootstrap_ci(self, n_iter: int = 1000, ci: float = 95.0,
                     rng_seed: int = 0):
        """Per-threshold (P, R, F1) CIs via image-level bootstrap.

        Returns dict: threshold -> {metric: (lo, mean, hi)}.
        Requires track_per_image=True at construction.
        """
        if not self.track_per_image or not self.per_image:
            return {}
        cm_stack = np.stack(self.per_image, axis=0)  # (N_img, n_thr, 3)
        n_img = cm_stack.shape[0]
        rng = np.random.default_rng(rng_seed)
        lo_q = (100 - ci) / 2
        hi_q = 100 - lo_q
        out = {}
        for i, thr in enumerate(self.thresholds):
            ps, rs, f1s = [], [], []
            for _ in range(n_iter):
                idx = rng.integers(0, n_img, size=n_img)
                sub = cm_stack[idx, i, :]
                tp = int(sub[:, 0].sum())
                fp = int(sub[:, 1].sum())
                fn = int(sub[:, 2].sum())
                p = tp / (tp + fp) if tp + fp > 0 else 0.0
                r = tp / (tp + fn) if tp + fn > 0 else 0.0
                f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
                ps.append(p); rs.append(r); f1s.append(f1)
            ps_a = np.array(ps); rs_a = np.array(rs); f1s_a = np.array(f1s)
            out[thr] = {
                "precision": (
                    float(np.percentile(ps_a, lo_q)),
                    float(ps_a.mean()),
                    float(np.percentile(ps_a, hi_q)),
                ),
                "recall": (
                    float(np.percentile(rs_a, lo_q)),
                    float(rs_a.mean()),
                    float(np.percentile(rs_a, hi_q)),
                ),
                "f1": (
                    float(np.percentile(f1s_a, lo_q)),
                    float(f1s_a.mean()),
                    float(np.percentile(f1s_a, hi_q)),
                ),
            }
        return out


def box_iou(b1, b2) -> float:
    x0a, y0a, x1a, y1a = b1
    x0b, y0b, x1b, y1b = b2
    ix0 = max(x0a, x0b)
    iy0 = max(y0a, y0b)
    ix1 = min(x1a, x1b)
    iy1 = min(y1a, y1b)
    iw = max(0, ix1 - ix0 + 1)
    ih = max(0, iy1 - iy0 + 1)
    inter = iw * ih
    a = (x1a - x0a + 1) * (y1a - y0a + 1)
    b = (x1b - x0b + 1) * (y1b - y0b + 1)
    union = a + b - inter
    return inter / union if union > 0 else 0.0


def _disk_kernel(radius: int) -> np.ndarray:
    """Return a square structuring element approximating a disk of given
    radius (used as the morphological structuring element)."""
    base = generate_binary_structure(2, 1)
    return iterate_structure(base, radius)


def predicted_components(sigmoid_2d: np.ndarray, score_thr: float,
                         closing_radius: int = 0,
                         min_area: int = 0,
                         score_mode: str = "max",
                         core_alpha: float | None = None):
    """Return list of (score, x0, y0, x1, y1) for connected components.

    Optional post-processing: morphological closing fills small gaps in
    fragmented predictions; min_area drops noise components; score_mode
    chooses how to aggregate the sigmoid into a single confidence per
    component ('max' or 'mean'). With core_alpha set, each component's box
    is shrunk to the pixels >= core_alpha * (component max sigmoid) — the
    FWHM-style energetic core — while min_area and score still use the
    full component.
    """
    binary = sigmoid_2d > score_thr
    if not binary.any():
        return []
    if closing_radius > 0:
        binary = binary_closing(binary, structure=_disk_kernel(closing_radius))
    lab, n = label(binary)
    out = []
    for comp_id in range(1, n + 1):
        mask = lab == comp_id
        if mask.sum() < min_area:
            continue
        if core_alpha is not None:
            tight = mask & (sigmoid_2d >= core_alpha * sigmoid_2d[mask].max())
            ys, xs = np.where(tight)
        else:
            ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        if score_mode == "mean":
            score = float(sigmoid_2d[mask].mean())
        else:
            score = float(sigmoid_2d[mask].max())
        out.append((score, x0, y0, x1, y1))
    return out


def merge_boxes_time(comps, max_gap: float, freq_overlap: float = 0.5):
    """Merge predicted boxes that overlap in frequency and are close in time.

    Repeatedly unions boxes whose y-ranges overlap by >= freq_overlap of the
    smaller box and whose time gap is <= max_gap, until a fixpoint. Joins
    pulse-train fragments (one GT box per train in e.g. RadDet) into a
    single detection; merged score is the max of the members.
    """
    comps = [list(c) for c in comps]
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(comps)
        for i in range(len(comps)):
            if used[i]:
                continue
            used[i] = True
            s, x0, y0, x1, y1 = comps[i]
            merged_any = True
            while merged_any:
                merged_any = False
                for j in range(len(comps)):
                    if used[j]:
                        continue
                    s2, a0, b0, a1, b1 = comps[j]
                    ov = min(y1, b1) - max(y0, b0) + 1
                    m = min(y1 - y0, b1 - b0) + 1
                    if m <= 0 or ov / m < freq_overlap:
                        continue
                    gap = max(a0 - x1 - 1, x0 - a1 - 1)
                    if gap > max_gap:
                        continue
                    x0, x1 = min(x0, a0), max(x1, a1)
                    y0, y1 = min(y0, b0), max(y1, b1)
                    s = max(s, s2)
                    used[j] = True
                    merged_any = True
                    changed = True
            out.append([s, x0, y0, x1, y1])
        comps = out
    return [tuple(c) for c in comps]


def gt_components(mask_2d: np.ndarray):
    """Return list of (x0, y0, x1, y1) for GT connected components."""
    binary = mask_2d > 0.5
    if not binary.any():
        return []
    lab, n = label(binary)
    out = []
    for comp_id in range(1, n + 1):
        ys, xs = np.where(lab == comp_id)
        if ys.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        out.append((x0, y0, x1, y1))
    return out


class DetectionAccumulator:
    """Greedy score-ordered match per IoU threshold; AP from PR curve.

    Boxes are passed as (cls, x0, y0, x1, y1) for GT (cls=-1 if class-agnostic)
    and (score, x0, y0, x1, y1) for predictions.
    """

    def __init__(self, iou_thresholds, n_classes: int = 1,
                 track_per_image: bool = False,
                 bootstrap_iou_thresholds=None):
        self.iou_thresholds = list(iou_thresholds)
        self.n_classes = n_classes
        self.records = {t: [] for t in self.iou_thresholds}
        self.gt_count_per_class = defaultdict(int)
        self.gt_total = 0
        self.track_per_image = track_per_image
        # Bootstrap only a subset of IoU thresholds to keep memory bounded.
        if bootstrap_iou_thresholds is None:
            self.bootstrap_iou_thresholds = self.iou_thresholds
        else:
            self.bootstrap_iou_thresholds = [
                t for t in bootstrap_iou_thresholds
                if t in self.iou_thresholds
            ]
        self.per_image_records: list[dict] = []
        self.per_image_gt_per_class: list[dict] = []

    def add_image(self, pred_boxes, gt_boxes):
        gt_per_class = defaultdict(int)
        for cls, *_ in gt_boxes:
            self.gt_count_per_class[cls] += 1
            self.gt_total += 1
            gt_per_class[cls] += 1
        sorted_preds = sorted(pred_boxes, key=lambda x: -x[0])
        per_img: dict = {} if self.track_per_image else None
        for iou_thr in self.iou_thresholds:
            matched = [False] * len(gt_boxes)
            entries = []
            for score, x0, y0, x1, y1 in sorted_preds:
                best_iou = 0.0
                best_j = -1
                for j, (cls, gx0, gy0, gx1, gy1) in enumerate(gt_boxes):
                    if matched[j]:
                        continue
                    iou = box_iou((x0, y0, x1, y1), (gx0, gy0, gx1, gy1))
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_j >= 0 and best_iou >= iou_thr:
                    matched[best_j] = True
                    cls = gt_boxes[best_j][0]
                    self.records[iou_thr].append((score, True, cls))
                    entries.append((score, True, cls))
                else:
                    self.records[iou_thr].append((score, False, -1))
                    entries.append((score, False, -1))
            if per_img is not None and iou_thr in self.bootstrap_iou_thresholds:
                per_img[iou_thr] = entries
        if self.track_per_image:
            self.per_image_records.append(per_img)
            self.per_image_gt_per_class.append(dict(gt_per_class))

    def compute(self):
        out = {}
        for iou_thr in self.iou_thresholds:
            recs = sorted(self.records[iou_thr], key=lambda r: -r[0])
            tp_arr = np.array([1 if r[1] else 0 for r in recs])
            fp_arr = 1 - tp_arr
            tp_cum = np.cumsum(tp_arr)
            fp_cum = np.cumsum(fp_arr)
            recall = tp_cum / max(self.gt_total, 1)
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
            ap = self._voc_ap(recall, precision)
            out[iou_thr] = {
                "ap": ap,
                "n_pred": len(recs),
                "n_tp": int(tp_arr.sum()),
                "n_fp": int(fp_arr.sum()),
                "n_gt": self.gt_total,
                "recall": float(recall[-1]) if recall.size else 0.0,
                "precision": float(precision[-1]) if precision.size else 0.0,
            }
        return out

    def per_class_recall(self, iou_thr):
        per_cls = defaultdict(int)
        for score, is_tp, cls in self.records[iou_thr]:
            if is_tp:
                per_cls[cls] += 1
        out = {}
        for cls in range(self.n_classes):
            n_gt = self.gt_count_per_class.get(cls, 0)
            r = per_cls.get(cls, 0) / n_gt if n_gt > 0 else float("nan")
            out[cls] = {
                "n_gt": n_gt, "n_tp": per_cls.get(cls, 0), "recall": r,
            }
        return out

    def per_class_ap(self, iou_thr):
        """Per-class AP for a class-agnostic detector.

        For each class C, a prediction is a TP if it matched a GT of class C.
        All other predictions (unmatched, or matched to GT of class != C) are
        FPs for class C. AP is the 11-point PASCAL VOC area under the PR
        curve. This is the standard PASCAL convention for a class-agnostic
        detector evaluated against a multi-class GT.
        """
        recs = sorted(self.records[iou_thr], key=lambda r: -r[0])
        out = {}
        for c in range(self.n_classes):
            n_gt_c = self.gt_count_per_class.get(c, 0)
            if n_gt_c == 0:
                out[c] = {
                    "ap": float("nan"), "n_gt": 0, "n_tp": 0,
                    "precision": 0.0, "recall": 0.0,
                }
                continue
            tp_c = np.array(
                [1 if (r[1] and r[2] == c) else 0 for r in recs]
            )
            fp_c = 1 - tp_c
            tp_cum = np.cumsum(tp_c)
            fp_cum = np.cumsum(fp_c)
            recall = tp_cum / n_gt_c
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
            ap = self._voc_ap(recall, precision)
            n_tp = int(tp_c.sum())
            out[c] = {
                "ap": ap, "n_gt": n_gt_c, "n_tp": n_tp,
                "precision": (
                    float(precision[-1]) if precision.size else 0.0
                ),
                "recall": float(recall[-1]) if recall.size else 0.0,
            }
        return out

    @staticmethod
    def _voc_ap(recall, precision):
        if recall.size == 0:
            return 0.0
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recall >= t
            p = float(precision[mask].max()) if mask.any() else 0.0
            ap += p / 11
        return float(ap)

    def bootstrap_ap(self, iou_thresholds=None, n_iter: int = 1000,
                     ci: float = 95.0, rng_seed: int = 0):
        """Image-level bootstrap of detection AP at one or more IoU thresholds.

        Returns dict: iou_thr -> (lo, mean, hi). Requires track_per_image=True.
        """
        if not self.track_per_image or not self.per_image_records:
            return {}
        if iou_thresholds is None:
            iou_thresholds = self.iou_thresholds
        n_img = len(self.per_image_records)
        rng = np.random.default_rng(rng_seed)
        lo_q = (100 - ci) / 2
        hi_q = 100 - lo_q
        out = {}
        for iou_thr in iou_thresholds:
            if iou_thr not in self.iou_thresholds:
                continue
            aps = []
            for _ in range(n_iter):
                idx = rng.integers(0, n_img, size=n_img)
                gt_total = 0
                recs = []
                for k in idx:
                    gt_total += sum(self.per_image_gt_per_class[k].values())
                    recs.extend(self.per_image_records[k][iou_thr])
                if gt_total == 0:
                    aps.append(0.0); continue
                recs.sort(key=lambda r: -r[0])
                tp_arr = np.array([1 if r[1] else 0 for r in recs])
                fp_arr = 1 - tp_arr
                tp_cum = np.cumsum(tp_arr)
                fp_cum = np.cumsum(fp_arr)
                recall = tp_cum / max(gt_total, 1)
                precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
                aps.append(self._voc_ap(recall, precision))
            arr = np.array(aps)
            out[iou_thr] = (
                float(np.percentile(arr, lo_q)),
                float(arr.mean()),
                float(np.percentile(arr, hi_q)),
            )
        return out

    def coco_map(self, coco_iou_thresholds):
        """COCO-style mAP averaged across an IoU range."""
        results = self.compute()
        aps = [results[t]["ap"] for t in coco_iou_thresholds if t in results]
        return float(np.mean(aps)) if aps else 0.0
