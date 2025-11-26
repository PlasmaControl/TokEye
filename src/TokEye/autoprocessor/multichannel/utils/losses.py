"""
Noise-robust loss functions for semantic segmentation with noisy labels.
Includes label smoothing, symmetric losses, and Dice-based losses.
Also includes pixel-wise supervised contrastive learning for segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingBCELoss(nn.Module):
    """
    Binary Cross-Entropy with Label Smoothing.
    Smooths hard 0/1 labels to reduce overconfidence on noisy labels.
    """
    
    def __init__(self, smoothing: float = 0.1):
        """
        Args:
            smoothing: Label smoothing factor (0 = no smoothing, 1 = maximum smoothing)
                      Labels are transformed: 0 -> smoothing, 1 -> 1-smoothing
        """
        super().__init__()
        self.smoothing = smoothing
        assert 0 <= smoothing < 0.5, "Smoothing should be in [0, 0.5)"
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (before sigmoid), shape (B, C, H, W) or (B, H, W)
            targets: Ground truth labels (0/1), same shape as logits
            
        Returns:
            Scalar loss value
        """
        # Apply label smoothing
        targets_smooth = targets * (1 - self.smoothing) + self.smoothing
        
        # Compute BCE with logits
        loss = F.binary_cross_entropy_with_logits(logits, targets_smooth)
        return loss


class SymmetricCrossEntropyLoss(nn.Module):
    """
    Symmetric Cross-Entropy Loss for robust learning with noisy labels.
    Combines standard CE with reverse CE to make loss symmetric and robust.
    
    Reference: "Symmetric Cross Entropy for Robust Learning with Noisy Labels"
    Wang et al., ICCV 2019
    """
    
    def __init__(self, alpha: float = 0.1, beta: float = 1.0):
        """
        Args:
            alpha: Weight for reverse cross-entropy term
            beta: Weight for standard cross-entropy term
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (before sigmoid)
            targets: Ground truth labels (0/1)
            
        Returns:
            Scalar loss value
        """
        # Get predictions
        probs = torch.sigmoid(logits)
        
        # Standard cross-entropy: -y*log(p) - (1-y)*log(1-p)
        ce = F.binary_cross_entropy_with_logits(logits, targets)
        
        # Reverse cross-entropy: -p*log(y) - (1-p)*log(1-y)
        # Add small epsilon to targets to avoid log(0)
        targets_eps = torch.clamp(targets, min=1e-7, max=1-1e-7)
        rce = -probs * torch.log(targets_eps) - (1 - probs) * torch.log(1 - targets_eps)
        rce = rce.mean()
        
        # Combine
        loss = self.alpha * rce + self.beta * ce
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    More robust to class imbalance and boundary errors than BCE.
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (before sigmoid)
            targets: Ground truth labels (0/1)
            
        Returns:
            Scalar loss value (1 - Dice coefficient)
        """
        # Get predictions
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        # Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return loss (1 - Dice)
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions with configurable weights.
    """
    
    def __init__(self, losses: list, weights: list):
        """
        Args:
            losses: List of loss modules
            weights: List of weights for each loss (should sum to 1.0)
        """
        super().__init__()
        assert len(losses) == len(weights), "Number of losses and weights must match"
        self.losses = nn.ModuleList(losses)
        self.weights = weights
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted combination of all losses.
        """
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(logits, targets)
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses on hard examples by down-weighting easy ones.
    
    Reference: "Focal Loss for Dense Object Detection" Lin et al., ICCV 2017
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for class balance
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (before sigmoid)
            targets: Ground truth labels (0/1)
            
        Returns:
            Scalar loss value
        """
        # Get predictions
        probs = torch.sigmoid(logits)
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal weight
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Combine
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss.
    Directly optimizes IoU metric.
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (before sigmoid)
            targets: Ground truth labels (0/1)
            
        Returns:
            Scalar loss value (1 - IoU)
        """
        # Get predictions
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        # IoU
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Return loss (1 - IoU)
        return 1 - iou


class PixelContrastiveLoss(nn.Module):
    """
    Pixel-wise Supervised Contrastive Loss for semantic segmentation.
    
    Learns discriminative pixel embeddings by pulling together pixels of the same class
    and pushing apart pixels of different classes. Supports multi-class segmentation
    with background class.
    
    Reference: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
    Adapted for dense pixel-wise prediction with efficient sampling.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        num_samples: int = 512,
        dim: int = 128,
    ):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
            num_samples_per_class: Number of pixels to sample per class per batch
            embedding_dim: Expected embedding dimension (for validation)
        """
        super().__init__()
        self.temperature = temperature
        self.num_samples = num_samples
        self.dim = dim
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pixel-wise contrastive loss.
        
        Args:
            embeddings: Pixel embeddings of shape (B, D, H_emb, W_emb) where D is embedding dimension
            masks: Ground truth masks of shape (B, C, H_mask, W_mask) where C=2 (normal, baseline channels)
                   Background is defined as pixels where both channels are 0
                   Will be downsampled to match embedding spatial resolution
            
        Returns:
            Scalar contrastive loss
        """
        B, D, H, W = embeddings.shape
        _, C, H_mask, W_mask = masks.shape
        
        # Downsample masks to match embedding spatial resolution
        if H != H_mask or W != W_mask:
            masks = F.interpolate(masks, size=(H, W), mode='nearest')
        
        # Flatten spatial dimensions: (B, D, H*W) -> (B*H*W, D)
        embeddings_flat = embeddings.permute(0, 2, 3, 1).reshape(-1, D)
        
        # Create class labels from masks: (B, C, H, W) -> (B*H*W,)
        # Class 0: background (both channels are 0)
        # Class 1: normal (channel 0 is 1)
        # Class 2: baseline (channel 1 is 1)
        masks_flat = masks.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        
        # Assign class labels based on mask channels
        # Priority: baseline > normal > background (in case of overlap)
        class_labels = torch.zeros(B * H * W, dtype=torch.long, device=embeddings.device)
        # First assign normal (class 1)
        class_labels = torch.where(masks_flat[:, 0] > 0.5, 
                                   torch.ones_like(class_labels), 
                                   class_labels)
        # Then assign baseline (class 2) - overrides normal if overlap
        class_labels = torch.where(masks_flat[:, 1] > 0.5, 
                                   torch.full_like(class_labels, 2), 
                                   class_labels)
        
        # Sample pixels from each class to make computation tractable
        sampled_indices = []
        sampled_labels = []
        
        for class_id in range(3):  # 0: background, 1: normal, 2: baseline
            class_mask = (class_labels == class_id)
            class_indices = torch.where(class_mask)[0]
            
            if len(class_indices) == 0:
                continue
            
            # Sample pixels from this class
            num_samples = min(self.num_samples, len(class_indices))
            if num_samples > 0:
                sampled_idx = class_indices[
                    torch.randperm(len(class_indices), device=embeddings.device)[:num_samples]
                ]
                sampled_indices.append(sampled_idx)
                sampled_labels.append(
                    torch.full((num_samples,), class_id, dtype=torch.long, device=embeddings.device)
                )
        
        if len(sampled_indices) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        sampled_indices = torch.cat(sampled_indices)
        sampled_labels = torch.cat(sampled_labels)
        sampled_embeddings = embeddings_flat[sampled_indices]
        
        sampled_embeddings = F.normalize(sampled_embeddings, p=2, dim=1)
        
        similarity_matrix = torch.matmul(sampled_embeddings, sampled_embeddings.t()) / self.temperature
        
        label_mask = (sampled_labels.unsqueeze(0) == sampled_labels.unsqueeze(1)).float()
        
        logits_mask = torch.ones_like(label_mask) - torch.eye(len(sampled_labels), device=embeddings.device)
        label_mask = label_mask * logits_mask
        
        similarity_matrix = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True)[0].detach()
        
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-7)
        
        num_positives_per_row = label_mask.sum(dim=1)
        valid_rows = num_positives_per_row > 0
        
        if valid_rows.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        mean_log_prob_pos = (label_mask * log_prob).sum(dim=1) / (num_positives_per_row + 1e-7)
        mean_log_prob_pos = mean_log_prob_pos[valid_rows]
        
        loss = -mean_log_prob_pos.mean()
        
        return loss


def get_loss_function(settings: dict) -> nn.Module:
    """
    Create loss function from settings.
    
    Args:
        settings: Dictionary with loss configuration
        
    Returns:
        Loss module
    """
    loss_type = settings.get('loss_type', 'bce')
    
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    
    elif loss_type == 'label_smooth_bce':
        smoothing = settings.get('label_smoothing', 0.1)
        return LabelSmoothingBCELoss(smoothing=smoothing)
    
    elif loss_type == 'symmetric_bce':
        alpha = settings.get('symmetric_alpha', 0.1)
        beta = settings.get('symmetric_beta', 1.0)
        return SymmetricCrossEntropyLoss(alpha=alpha, beta=beta)
    
    elif loss_type == 'dice':
        return DiceLoss()
    
    elif loss_type == 'dice_bce':
        dice_weight = settings.get('dice_weight', 0.5)
        bce_weight = settings.get('bce_weight', 0.5)
        return CombinedLoss(
            losses=[DiceLoss(), nn.BCEWithLogitsLoss()],
            weights=[dice_weight, bce_weight]
        )
    
    elif loss_type == 'symmetric_bce_dice':
        # Default: symmetric BCE + Dice for maximum robustness
        symmetric_weight = settings.get('symmetric_weight', 0.5)
        dice_weight = settings.get('dice_weight', 0.5)
        alpha = settings.get('symmetric_alpha', 0.1)
        beta = settings.get('symmetric_beta', 1.0)
        return CombinedLoss(
            losses=[
                SymmetricCrossEntropyLoss(alpha=alpha, beta=beta),
                DiceLoss()
            ],
            weights=[symmetric_weight, dice_weight]
        )
    
    elif loss_type == 'focal':
        alpha = settings.get('focal_alpha', 0.25)
        gamma = settings.get('focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'focal_dice':
        focal_weight = settings.get('focal_weight', 0.5)
        dice_weight = settings.get('dice_weight', 0.5)
        alpha = settings.get('focal_alpha', 0.25)
        gamma = settings.get('focal_gamma', 2.0)
        return CombinedLoss(
            losses=[FocalLoss(alpha=alpha, gamma=gamma), DiceLoss()],
            weights=[focal_weight, dice_weight]
        )
    
    elif loss_type == 'iou':
        return IoULoss()
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Compute Dice coefficient metric.
    
    Args:
        logits: Model predictions (before sigmoid)
        targets: Ground truth labels (0/1)
        threshold: Threshold for binarizing predictions
        
    Returns:
        Dice coefficient (scalar)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)
    
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum()
    
    if union == 0:
        return torch.tensor(1.0, device=logits.device)
    
    dice = (2.0 * intersection) / union
    return dice


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Compute IoU (Jaccard) score.
    
    Args:
        logits: Model predictions (before sigmoid)
        targets: Ground truth labels (0/1)
        threshold: Threshold for binarizing predictions
        
    Returns:
        IoU score (scalar)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)
    
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection
    
    if union == 0:
        return torch.tensor(1.0, device=logits.device)
    
    iou = intersection / union
    return iou

