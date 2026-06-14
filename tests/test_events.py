from __future__ import annotations

import numpy as np

from tokeye.extra.eval.events import EventAccumulator, label8, match_events


def _labels(bool_arr):
    return label8(bool_arr)


def test_any_overlap_matches_thin_contour():
    """A prediction sharing pixels with a diagonal GT contour is a TP under
    any-overlap, even though their bounding-box IoU is tiny."""
    gt = np.zeros((20, 20), bool)
    pred = np.zeros((20, 20), bool)
    for k in range(20):
        gt[k, k] = True
        pred[k, min(k + 1, 19)] = True  # parallel, offset by 1; boxes ~overlap
    pred[5, 5] = True  # one shared pixel
    gl, ng = _labels(gt)
    pl, npd = _labels(pred)
    scores = np.array([0.0] + [0.9] * npd)
    recs, n_gt = match_events(gl, ng, pl, npd, scores, criterion="any")
    assert n_gt == 1
    assert sum(1 for _, tp in recs if tp) == 1  # exactly one TP


def test_no_pixel_overlap_is_fp_even_if_boxes_overlap():
    """Two diagonal contours crossing the same region but sharing no pixels
    must NOT match (the box-IoU failure mode this module fixes)."""
    gt = np.zeros((20, 20), bool)
    pred = np.zeros((20, 20), bool)
    for k in range(20):
        gt[k, k] = True            # main diagonal
        pred[k, 19 - k] = True     # anti-diagonal: same bbox, crosses once
    # remove the single crossing pixel so there is zero pixel overlap
    if gt[10, 10] and pred[10, 10]:
        pred[10, 10] = False
        pred[11, 8] = True
    gl, ng = _labels(gt)
    pl, npd = _labels(pred)
    scores = np.array([0.0] + [0.9] * npd)
    recs, n_gt = match_events(gl, ng, pl, npd, scores, criterion="any")
    assert sum(1 for _, tp in recs if tp) == 0


def test_coverage_threshold():
    gt = np.zeros((10, 10), bool)
    gt[0, :10] = True  # 10-pixel horizontal contour
    pred = np.zeros((10, 10), bool)
    pred[0, :3] = True  # covers 3/10 = 0.3
    gl, ng = _labels(gt)
    pl, npd = _labels(pred)
    scores = np.array([0.0, 0.9])
    tp = sum(1 for _, t in match_events(
        gl, ng, pl, npd, scores, criterion="coverage", cov_frac=0.2)[0] if t)
    assert tp == 1
    tp = sum(1 for _, t in match_events(
        gl, ng, pl, npd, scores, criterion="coverage", cov_frac=0.5)[0] if t)
    assert tp == 0


def test_center_hit():
    gt = np.zeros((10, 10), bool)
    gt[5, 2:9] = True  # centroid at (5, 5)
    through = np.zeros((10, 10), bool)
    through[3:8, 5] = True  # vertical line through (5,5)
    miss = np.zeros((10, 10), bool)
    miss[0, 0:3] = True
    gl, ng = _labels(gt)
    pl, npd = _labels(through)
    scores = np.array([0.0, 0.9])
    assert sum(1 for _, t in match_events(
        gl, ng, pl, npd, scores, criterion="center")[0] if t) == 1
    pl, npd = _labels(miss)
    assert sum(1 for _, t in match_events(
        gl, ng, pl, npd, scores, criterion="center")[0] if t) == 0


def test_one_prediction_matches_at_most_one_gt():
    """A single big prediction overlapping two GT contours scores one TP,
    one FN (greedy one-to-one), not two TPs."""
    gt = np.zeros((10, 10), bool)
    gt[2, :] = True
    gt[6, :] = True  # two separate horizontal contours
    pred = np.ones((10, 10), bool)  # one giant blob covering both
    acc = EventAccumulator(criterion="any")
    acc.add_image(gt, pred, np.full((10, 10), 0.8))
    out = acc.compute()
    assert out["n_gt"] == 2
    assert out["n_tp"] == 1
    assert out["recall"] == 0.5


def test_accumulator_perfect_and_empty():
    gt = np.zeros((10, 10), bool)
    gt[3, 1:8] = True
    acc = EventAccumulator(criterion="any")
    acc.add_image(gt, gt.copy(), gt.astype(float))      # perfect hit
    acc.add_image(np.zeros((10, 10), bool),
                  np.zeros((10, 10), bool), np.zeros((10, 10)))  # empty/empty
    out = acc.compute()
    assert out["n_gt"] == 1 and out["n_tp"] == 1
    assert out["recall"] == 1.0 and out["precision"] == 1.0
