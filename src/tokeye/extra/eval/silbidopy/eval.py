"""Binary-segmentation metrics computed from a global confusion matrix.

Replaces the previous torchmetrics-based implementation, which had subtle
dtype-dependent bugs (MeanIoU and GeneralizedDiceScore produced wrong values
when targets were passed as float instead of int). All metrics here are
computed exactly, from a running confusion matrix accumulated across all
update() calls.

Reports both `global` metrics (computed from the dataset-level confusion
matrix — the standard convention in semantic-segmentation papers) and
`per_image` metrics (mean of per-image IoU / Dice — the convention used in
Bustos et al. 2021 for TJII). Reporting both lets the paper compare directly
against either convention.
"""
from __future__ import annotations

import numpy as np
import torch


class Metrics:
    def __init__(self, device: str = "cpu", track_per_image: bool = True):
        self.device = device
        self.track_per_image = track_per_image
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.per_image_iou: list[float] = []
        self.per_image_dice: list[float] = []

    def update(self, output, ann):
        """Accept (B, 1, H, W) or any broadcastable shape; bool / int / float
        coerced to binary via > 0.5 thresholding for both pred and target.
        """
        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output)
        if isinstance(ann, np.ndarray):
            ann = torch.from_numpy(ann)
        pred = (output > 0.5) if output.dtype != torch.bool else output
        target = (ann > 0.5) if ann.dtype != torch.bool else ann
        tp_batch = (pred & target).sum().item()
        fp_batch = (pred & ~target).sum().item()
        fn_batch = (~pred & target).sum().item()
        tn_batch = (~pred & ~target).sum().item()
        self.tp += tp_batch
        self.fp += fp_batch
        self.fn += fn_batch
        self.tn += tn_batch
        if self.track_per_image:
            # Compute per-image IoU & Dice (foreground only). Image with no
            # GT positive AND no prediction → skip (undefined); image with GT
            # but no TP → IoU = Dice = 0.
            # pred and target shape: (B, ...). Flatten all but batch dim.
            B = pred.shape[0]
            pred_b = pred.reshape(B, -1)
            target_b = target.reshape(B, -1)
            for i in range(B):
                tp_i = int((pred_b[i] & target_b[i]).sum().item())
                fp_i = int((pred_b[i] & ~target_b[i]).sum().item())
                fn_i = int((~pred_b[i] & target_b[i]).sum().item())
                if tp_i + fn_i == 0 and tp_i + fp_i == 0:
                    # No GT and no prediction: skip (undefined for foreground)
                    continue
                denom_iou = tp_i + fp_i + fn_i
                self.per_image_iou.append(
                    tp_i / denom_iou if denom_iou > 0 else 0.0
                )
                denom_dice = 2 * tp_i + fp_i + fn_i
                self.per_image_dice.append(
                    2 * tp_i / denom_dice if denom_dice > 0 else 0.0
                )

    def compute(self) -> dict:
        tp, fp, fn = self.tp, self.fp, self.fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        iou_global = (
            tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        )
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
            out["iou_per_image_mean"] = float(np.mean(self.per_image_iou))
            out["dice_per_image_mean"] = float(np.mean(self.per_image_dice))
        return out
