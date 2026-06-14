"""Contour-aware event-level detection scoring.

Bounding-box IoU is a poor match criterion for thin, diagonal tonal
contours (whistles): a contour's box is mostly empty space, and two
crossing contours share almost the same box, so box IoU collapses even
when the prediction traces the contour well. This module matches
predicted to ground-truth *connected components* by pixel-set criteria
that do not depend on box geometry:

  - ``any``      : prediction shares >= ``min_overlap`` pixels with the GT
                   component (the reviewer's "any-overlap" criterion).
  - ``coverage`` : prediction covers >= ``cov_frac`` of the GT component's
                   pixels.
  - ``center``   : the prediction passes through the GT component's
                   centroid, within ``center_tol`` pixels (the reviewer's
                   "center-hit" criterion).

Matching is greedy in descending prediction score (one prediction per GT
event), and AP is the PASCAL-VOC 11-point area under the resulting PR
curve, mirroring ``DetectionAccumulator`` so event AP and box AP are
directly comparable.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import generate_binary_structure, label

# 8-connectivity: a 1-pixel-wide diagonal contour (the whistle case) is a
# single connected component, not a string of isolated pixels. 4-conn would
# shatter diagonal contours into many fragments and wreck event counts.
_CONN8 = generate_binary_structure(2, 2)


def label8(binary: np.ndarray):
    """``scipy.ndimage.label`` with 8-connectivity."""
    return label(binary, structure=_CONN8)


def _drop_small(lab: np.ndarray, n: int, min_area: int):
    """Remove components smaller than ``min_area``; relabel to 1..k."""
    if min_area <= 0 or n == 0:
        return lab, n
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    remap = np.zeros(n + 1, dtype=np.int64)
    new_id = 0
    for old in range(1, n + 1):
        if sizes[old] >= min_area:
            new_id += 1
            remap[old] = new_id
    return remap[lab], new_id


def _overlap_matrix(gt_lab: np.ndarray, n_gt: int,
                    pred_lab: np.ndarray, n_pred: int) -> np.ndarray:
    """Return (n_gt+1, n_pred+1) pixel-overlap counts incl. background row/col.

    Entry [i, j] = number of pixels labelled GT-i and pred-j (label 0 is
    background). Computed in one pass via a flat 2-D histogram.
    """
    flat = gt_lab.ravel().astype(np.int64) * (n_pred + 1) + pred_lab.ravel()
    counts = np.bincount(flat, minlength=(n_gt + 1) * (n_pred + 1))
    return counts.reshape(n_gt + 1, n_pred + 1)


def match_events(gt_lab: np.ndarray, n_gt: int,
                 pred_lab: np.ndarray, n_pred: int,
                 pred_scores: np.ndarray,
                 criterion: str = "any",
                 min_overlap: int = 1,
                 cov_frac: float = 0.2,
                 center_tol: int = 0):
    """Greedy score-ordered event match. Returns (records, n_gt).

    ``records`` is a list of (score, is_tp) for every prediction, in
    arbitrary order; callers accumulate and sort globally for AP. GT
    components left unmatched are false negatives (n_gt - sum of TP).

    Labels are 1..n_gt / 1..n_pred (0 = background), as produced by
    ``scipy.ndimage.label``. ``pred_scores`` is indexed [0..n_pred] with
    index 0 unused.
    """
    if n_pred == 0:
        return [], n_gt
    ov = _overlap_matrix(gt_lab, n_gt, pred_lab, n_pred)
    gt_sizes = ov[1:, :].sum(axis=1)  # pixels per GT component (incl. bg col)
    ov_fg = ov[1:, 1:]  # [gt, pred] foreground overlap, 0-indexed

    # Candidate (gt, pred) pairs satisfying the criterion.
    if criterion == "any":
        ok = ov_fg >= min_overlap
    elif criterion == "coverage":
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = ov_fg / gt_sizes[:, None]
        ok = frac >= cov_frac
    elif criterion == "center":
        ok = np.zeros_like(ov_fg, dtype=bool)
        H, W = gt_lab.shape
        for i in range(n_gt):
            ys, xs = np.where(gt_lab == i + 1)
            cy = int(round(ys.mean()))
            cx = int(round(xs.mean()))
            y0, y1 = max(0, cy - center_tol), min(H, cy + center_tol + 1)
            x0, x1 = max(0, cx - center_tol), min(W, cx + center_tol + 1)
            labs = np.unique(pred_lab[y0:y1, x0:x1])
            for j in labs:
                if j > 0:
                    ok[i, j - 1] = True
    else:
        raise ValueError(f"unknown criterion {criterion!r}")

    order = np.argsort(-pred_scores[1:])  # pred indices (0-based) by score
    matched_gt = np.zeros(n_gt, dtype=bool)
    records = []
    for j in order:
        score = float(pred_scores[j + 1])
        cand = np.where(ok[:, j] & ~matched_gt)[0]
        if cand.size:
            best = cand[np.argmax(ov_fg[cand, j])]
            matched_gt[best] = True
            records.append((score, True))
        else:
            records.append((score, False))
    return records, n_gt


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        p = float(precision[mask].max()) if mask.any() else 0.0
        ap += p / 11
    return float(ap)


class EventAccumulator:
    """Accumulate event matches across images for one criterion."""

    def __init__(self, criterion: str = "any", min_overlap: int = 1,
                 cov_frac: float = 0.2, center_tol: int = 0):
        self.criterion = criterion
        self.kw = {"min_overlap": min_overlap, "cov_frac": cov_frac,
                   "center_tol": center_tol}
        self.records: list[tuple[float, bool]] = []
        self.n_gt = 0

    def add_image(self, gt_bool: np.ndarray, pred_bool: np.ndarray,
                  score_map: np.ndarray, gt_min_area: int = 0,
                  pred_min_area: int = 0):
        gt_lab, n_gt = _drop_small(*label8(gt_bool), gt_min_area)
        pred_lab, n_pred = _drop_small(*label8(pred_bool), pred_min_area)
        if n_pred:
            sums = np.bincount(pred_lab.ravel(), weights=score_map.ravel(),
                               minlength=n_pred + 1)
            cnts = np.bincount(pred_lab.ravel(), minlength=n_pred + 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                scores = np.where(cnts > 0, sums / cnts, 0.0)
        else:
            scores = np.zeros(1)
        recs, ng = match_events(
            gt_lab, n_gt, pred_lab, n_pred, scores,
            criterion=self.criterion, **self.kw,
        )
        self.records.extend(recs)
        self.n_gt += ng

    def compute(self) -> dict:
        recs = sorted(self.records, key=lambda r: -r[0])
        tp = np.array([1 if r[1] else 0 for r in recs], dtype=np.int64)
        fp = 1 - tp
        n_pred = len(recs)
        n_tp = int(tp.sum())
        if n_pred == 0 or self.n_gt == 0:
            return {
                "ap": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_pred": n_pred, "n_tp": n_tp, "n_gt": self.n_gt,
            }
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / self.n_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        ap = _voc_ap(recall, precision)
        p = n_tp / n_pred
        r = n_tp / self.n_gt
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {
            "ap": ap, "precision": p, "recall": r, "f1": f1,
            "n_pred": n_pred, "n_tp": n_tp, "n_gt": self.n_gt,
        }
