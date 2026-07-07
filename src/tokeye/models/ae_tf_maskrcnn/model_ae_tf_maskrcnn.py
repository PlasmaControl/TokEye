import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from .config_ae_tf_maskrcnn import AETFMaskConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AETFMaskModel(nn.Module):
    def __init__(self, config: AETFMaskConfig):
        super().__init__()
        self.config = config

        # config.weights=None skips the COCO download entirely; pass
        # AETFMaskConfig(weights=None) when a checkpoint will overwrite
        # every parameter anyway (e.g. the hub registry builder).
        model = maskrcnn_resnet50_fpn_v2(weights=config.weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore[union-attr]

        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, config.num_classes
        )
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels  # type: ignore[union-attr]

        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            config.hidden_layer,
            config.num_classes,
        )
        model.transform = GeneralizedRCNNTransform(
            min_size=config.min_size,  # 800
            max_size=config.max_size,  # 1333
            image_mean=config.image_mean,  # No normalization
            image_std=config.image_std,  # No normalization
        )
        # Copy all modules from model to self
        for name, module in model.named_children():
            setattr(self, name, module)

    def forward(self, images, targets=None):
        # During training, targets should be provided
        # During inference, targets should be None
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses
        return detections
