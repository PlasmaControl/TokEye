"""
Data augmentation transforms for robust segmentation training with noisy labels.
Implements geometric and intensity transforms that apply consistently to image-mask pairs.
Also includes SpecAugment for spectrogram-specific augmentations.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, map_coordinates

# Module-level random generator for reproducibility
_rng = np.random.default_rng()


class SegmentationAugmentation:
    """
    Compose multiple augmentation transforms for segmentation.
    Applies transforms consistently to both image and mask.
    """

    def __init__(
        self,
        rotation_degrees: float = 180,
        prob_flip_h: float = 0.5,
        prob_flip_v: float = 0.5,
        elastic: bool = True,
        elastic_alpha: float = 50.0,
        elastic_sigma: float = 5.0,
        scale_range: tuple[float, float] = (0.8, 1.2),
        intensity_transforms: bool = True,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        contrast_range: tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.05,
        blur_prob: float = 0.3,
        blur_sigma_range: tuple[float, float] = (0.5, 1.5),
        gamma_range: tuple[float, float] = (0.8, 1.2),
        apply_prob: float = 0.8,
        specaugment: SpecAugment | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Args:
            rotation_degrees: Maximum rotation angle in degrees (±)
            prob_flip_h: Probability of horizontal flip
            prob_flip_v: Probability of vertical flip
            elastic: Enable elastic deformation
            elastic_alpha: Elastic deformation alpha parameter
            elastic_sigma: Elastic deformation sigma parameter
            scale_range: Random scaling range (min, max)
            intensity_transforms: Enable intensity augmentations
            brightness_range: Brightness multiplier range (min, max)
            contrast_range: Contrast multiplier range (min, max)
            noise_std: Standard deviation of Gaussian noise
            blur_prob: Probability of applying Gaussian blur
            blur_sigma_range: Gaussian blur sigma range (min, max)
            gamma_range: Gamma correction range (min, max)
            apply_prob: Probability of applying augmentation at all
            specaugment: Optional SpecAugment instance for spectrogram-specific augmentations
            rng: Optional random number generator for reproducibility
        """
        self.rotation_degrees = rotation_degrees
        self.prob_flip_h = prob_flip_h
        self.prob_flip_v = prob_flip_v
        self.elastic = elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.scale_range = scale_range
        self.intensity_transforms = intensity_transforms
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.blur_prob = blur_prob
        self.blur_sigma_range = blur_sigma_range
        self.gamma_range = gamma_range
        self.apply_prob = apply_prob
        self.specaugment = specaugment
        self._rng = rng if rng is not None else _rng

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation to image and mask consistently.

        Args:
            image: Tensor of shape (C, H, W) or (H, W)
            mask: Tensor of shape (C, H, W) or (H, W)

        Returns:
            Augmented (image, mask) pair
        """
        if self._rng.random() > self.apply_prob:
            return image, mask

        # Ensure proper shape
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # Geometric transforms (apply to both image and mask)
        image, mask = self._apply_geometric_transforms(image, mask)

        # Intensity transforms (apply only to image)
        if self.intensity_transforms:
            image = self._apply_intensity_transforms(image)

        return image, mask

    def _apply_geometric_transforms(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply geometric transforms consistently to image and mask."""

        # Random rotation
        if self.rotation_degrees > 0:
            angle = self._rng.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = self._rotate(image, angle, mode="bilinear")
            mask = self._rotate(mask, angle, mode="nearest")

        # Random horizontal flip
        if self._rng.random() < self.prob_flip_h:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])

        # Random vertical flip
        if self._rng.random() < self.prob_flip_v:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])

        # Random scaling
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            scale = self._rng.uniform(self.scale_range[0], self.scale_range[1])
            image = self._scale(image, scale, mode="bilinear")
            mask = self._scale(mask, scale, mode="nearest")

        # Elastic deformation
        if self.elastic and self._rng.random() < 0.5:
            image, mask = self._elastic_transform(image, mask)

        return image, mask

    def _apply_intensity_transforms(self, image: torch.Tensor) -> torch.Tensor:
        """Apply intensity transforms to image only."""

        # Apply SpecAugment if provided (spectrogram-specific augmentation)
        if self.specaugment is not None:
            image = self.specaugment(image)

        # Random brightness
        if self.brightness_range[0] != 1.0 or self.brightness_range[1] != 1.0:
            brightness = self._rng.uniform(
                self.brightness_range[0], self.brightness_range[1]
            )
            image = image * brightness

        # Random contrast
        if self.contrast_range[0] != 1.0 or self.contrast_range[1] != 1.0:
            contrast = self._rng.uniform(self.contrast_range[0], self.contrast_range[1])
            mean = image.mean()
            image = (image - mean) * contrast + mean

        # Gaussian noise
        if self.noise_std > 0 and self._rng.random() < 0.5:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise

        # Gaussian blur
        if self._rng.random() < self.blur_prob:
            sigma = self._rng.uniform(
                self.blur_sigma_range[0], self.blur_sigma_range[1]
            )
            image = self._gaussian_blur(image, sigma)

        # Gamma correction
        if (
            self.gamma_range[0] != 1.0 or self.gamma_range[1] != 1.0
        ) and self._rng.random() < 0.5:
            gamma = self._rng.uniform(self.gamma_range[0], self.gamma_range[1])
            # Normalize to [0, 1], apply gamma, then denormalize
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image_norm = (image - img_min) / (img_max - img_min)
                image_norm = torch.pow(image_norm, gamma)
                image = image_norm * (img_max - img_min) + img_min

        return image

    @staticmethod
    def _rotate(
        tensor: torch.Tensor, angle: float, mode: str = "bilinear"
    ) -> torch.Tensor:
        """Rotate tensor by angle in degrees."""
        angle_rad = angle * np.pi / 180.0
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Rotation matrix
        theta = torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]],
            dtype=tensor.dtype,
            device=tensor.device,
        ).unsqueeze(0)

        # Add batch dimension if needed
        needs_squeeze = False
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            needs_squeeze = True

        grid = F.affine_grid(theta, tensor.size(), align_corners=False)
        rotated = F.grid_sample(tensor, grid, mode=mode, align_corners=False)

        if needs_squeeze:
            rotated = rotated.squeeze(0)

        return rotated

    @staticmethod
    def _scale(
        tensor: torch.Tensor, scale: float, mode: str = "bilinear"
    ) -> torch.Tensor:
        """Scale tensor by scale factor."""
        theta = torch.tensor(
            [[scale, 0, 0], [0, scale, 0]], dtype=tensor.dtype, device=tensor.device
        ).unsqueeze(0)

        # Add batch dimension if needed
        needs_squeeze = False
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            needs_squeeze = True

        grid = F.affine_grid(theta, tensor.size(), align_corners=False)
        scaled = F.grid_sample(tensor, grid, mode=mode, align_corners=False)

        if needs_squeeze:
            scaled = scaled.squeeze(0)

        return scaled

    def _elastic_transform(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply elastic deformation to image and mask consistently.
        Based on Simard et al. "Best Practices for Convolutional Neural Networks"
        """
        # Convert to numpy for scipy operations
        shape = image.shape[-2:]

        # Generate random displacement fields
        dx = (
            gaussian_filter(
                (self._rng.random(shape) * 2 - 1),
                self.elastic_sigma,
                mode="constant",
                cval=0,
            )
            * self.elastic_alpha
        )
        dy = (
            gaussian_filter(
                (self._rng.random(shape) * 2 - 1),
                self.elastic_sigma,
                mode="constant",
                cval=0,
            )
            * self.elastic_alpha
        )

        # Create meshgrid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # Apply to image
        image_np = image.cpu().numpy()
        image_warped = np.empty_like(image_np)
        for c in range(image_np.shape[0]):
            image_warped[c] = map_coordinates(
                image_np[c], indices, order=1, mode="reflect"
            ).reshape(shape)

        # Apply to mask
        mask_np = mask.cpu().numpy()
        mask_warped = np.empty_like(mask_np)
        for c in range(mask_np.shape[0]):
            mask_warped[c] = map_coordinates(
                mask_np[c], indices, order=0, mode="reflect"
            ).reshape(shape)

        return torch.from_numpy(image_warped), torch.from_numpy(mask_warped)

    @staticmethod
    def _gaussian_blur(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur to tensor."""
        tensor_np = tensor.cpu().numpy()
        blurred = np.empty_like(tensor_np)
        for c in range(tensor_np.shape[0]):
            blurred[c] = gaussian_filter(tensor_np[c], sigma=sigma)
        return torch.from_numpy(blurred).to(tensor.device)


class SpecAugment:
    """
    SpecAugment for spectrograms: Time Warping, Frequency Masking, Time Masking.

    Reference: "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
    Park et al., Interspeech 2019

    Note: This augmentation is applied only to images, not to masks.
    """

    def __init__(
        self,
        time_warp_W: int = 40,
        freq_mask_F: int = 15,
        time_mask_T: int = 20,
        num_freq_masks: int = 0,
        num_time_masks: int = 0,
        apply_prob: float = 0.8,
        rng: np.random.Generator | None = None,
    ):
        """
        Args:
            time_warp_W: Time warp parameter W (max displacement along time axis)
            freq_mask_F: Maximum width of frequency mask
            time_mask_T: Maximum width of time mask
            num_freq_masks: Number of frequency masks to apply (0 to disable)
            num_time_masks: Number of time masks to apply (0 to disable)
            apply_prob: Probability of applying SpecAugment
            rng: Optional random number generator for reproducibility
        """
        self.time_warp_W = time_warp_W
        self.freq_mask_F = freq_mask_F
        self.time_mask_T = time_mask_T
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.apply_prob = apply_prob
        self._rng = rng if rng is not None else _rng

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to image only (not mask).

        Args:
            image: Tensor of shape (C, H, W) - H=frequency, W=time

        Returns:
            Augmented image
        """
        if self._rng.random() > self.apply_prob:
            return image

        # Ensure proper shape
        if image.ndim == 2:
            image = image.unsqueeze(0)

        # Apply time warping
        if self.time_warp_W > 0:
            image = self._time_warp(image)

        # Apply frequency masking
        for _ in range(self.num_freq_masks):
            image = self._freq_mask(image)

        # Apply time masking
        for _ in range(self.num_time_masks):
            image = self._time_mask(image)

        return image

    def _time_warp(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping along the time axis (horizontal).
        Warps the spectrogram by displacing a random point along the time axis.
        """
        C, H, W = image.shape

        if 2 * self.time_warp_W >= W:
            # Image too small for warping
            return image

        # Choose a random center point in the middle of time axis
        center = self._rng.integers(self.time_warp_W, W - self.time_warp_W)

        # Random warp displacement
        warp = self._rng.integers(-self.time_warp_W, self.time_warp_W + 1)

        # Create warping grid
        # Generate base grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=image.device),
            torch.arange(W, dtype=torch.float32, device=image.device),
            indexing="ij",
        )

        # Apply warping to x coordinates around the center
        if warp != 0:
            # Linear warping from center
            left_dist = torch.clamp(center - x_grid, min=0)
            right_dist = torch.clamp(x_grid - center, min=0)

            # Warp left side
            warp_left = (left_dist / center) * warp if center > 0 else 0
            # Warp right side
            warp_right = (right_dist / (W - center)) * warp if center < W else 0

            x_grid = x_grid + torch.where(x_grid < center, warp_left, -warp_right)

        # Normalize to [-1, 1] for grid_sample
        x_grid = 2.0 * x_grid / (W - 1) - 1.0
        y_grid = 2.0 * y_grid / (H - 1) - 1.0

        # Stack to (H, W, 2) and add batch dimension
        grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0)

        # Apply warping
        image_batch = image.unsqueeze(0)
        warped = F.grid_sample(
            image_batch,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return warped.squeeze(0)

    def _freq_mask(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency masking (vertical stripes).
        Masks consecutive frequency bins.
        """
        C, H, W = image.shape

        # Random mask width
        f = self._rng.integers(0, self.freq_mask_F + 1)

        if f == 0 or f >= H:
            return image

        # Random starting frequency
        f0 = self._rng.integers(0, H - f + 1)

        # Create masked image
        masked = image.clone()
        masked[:, f0 : f0 + f, :] = 0

        return masked

    def _time_mask(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply time masking (horizontal stripes).
        Masks consecutive time frames.
        """
        C, H, W = image.shape

        # Random mask width
        t = self._rng.integers(0, self.time_mask_T + 1)

        if t == 0 or t >= W:
            return image

        # Random starting time
        t0 = self._rng.integers(0, W - t + 1)

        # Create masked image
        masked = image.clone()
        masked[:, :, t0 : t0 + t] = 0

        return masked


class NoAugmentation:
    """Identity transform - no augmentation applied."""

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return image, mask


def get_augmentation(settings: dict) -> SegmentationAugmentation | NoAugmentation:
    """
    Create augmentation pipeline from settings.

    Args:
        settings: Dictionary with augmentation parameters

    Returns:
        Augmentation callable
    """
    if not settings.get("augmentation", False):
        return NoAugmentation()

    # Create SpecAugment if enabled
    specaugment = None
    if settings.get("specaugment", False):
        specaugment = SpecAugment(
            time_warp_W=settings.get("specaug_time_warp_W", 40),
            freq_mask_F=settings.get("specaug_freq_mask_F", 15),
            time_mask_T=settings.get("specaug_time_mask_T", 20),
            num_freq_masks=settings.get("specaug_freq_mask_num", 0),
            num_time_masks=settings.get("specaug_time_mask_num", 0),
            apply_prob=settings.get("aug_apply_prob", 0.8),
        )

    return SegmentationAugmentation(
        rotation_degrees=settings.get("aug_rotation_degrees", 180),
        prob_flip_h=settings.get("aug_prob_flip", 0.5),
        prob_flip_v=settings.get("aug_prob_flip", 0.5),
        elastic=settings.get("aug_elastic", True),
        elastic_alpha=settings.get("aug_elastic_alpha", 50.0),
        elastic_sigma=settings.get("aug_elastic_sigma", 5.0),
        scale_range=tuple(settings.get("aug_scale_range", [0.8, 1.2])),
        intensity_transforms=settings.get("aug_intensity", True),
        brightness_range=tuple(settings.get("aug_brightness_range", [0.8, 1.2])),
        contrast_range=tuple(settings.get("aug_contrast_range", [0.8, 1.2])),
        noise_std=settings.get("aug_noise_std", 0.05),
        blur_prob=settings.get("aug_blur_prob", 0.3),
        blur_sigma_range=tuple(settings.get("aug_blur_sigma_range", [0.5, 1.5])),
        gamma_range=tuple(settings.get("aug_gamma_range", [0.8, 1.2])),
        apply_prob=settings.get("aug_apply_prob", 0.8),
        specaugment=specaugment,
    )
