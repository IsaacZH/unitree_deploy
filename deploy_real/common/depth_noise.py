"""Stereo depth noise simulator for deployment.

Realistic depth noise injection matching training behavior.
Converts depth → disparity, applies filtered random noise, converts back → depth.
Parameters (focal_length, baseline) must match training camera to ensure consistency.

Typical usage
-------------
    noise_simulator = DepthNoise(
        focal_length=391.9765,      # D435i focal length (pixels)
        baseline=0.049974,          # D435i stereo baseline (meters)
        min_depth=0.25,
        max_depth=10.0,
    )
    
    # In preprocessing pipeline:
    noisy_depth = noise_simulator(depth_tensor)  # depth_tensor shape: [B, 1, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthNoise(nn.Module):
    """Stereo depth noise simulator.
    
    Simulates realistic stereo matching errors by:
    1. Converting depth → disparity using focal length & baseline
    2. Applying spatially-filtered random noise and occlusion
    3. Converting noisy disparity back → depth
    4. Clipping to valid range [min_depth, max_depth]
    
    Parameters match training depth_noise_encoder.DepthNoise exactly.
    """
    
    def __init__(
        self,
        focal_length: float,
        baseline: float,
        min_depth: float,
        max_depth: float,
        filter_size: int = 3,
        inlier_thred_range: tuple = (0.01, 0.05),
        prob_range: tuple = (0.4, 0.6),
        invalid_disp: float = 1e7,
    ):
        """Initialize stereo depth noise simulator.
        
        Args:
            focal_length: Camera focal length in pixels (e.g., 391.9765 for D435i).
            baseline: Stereo baseline distance in meters (e.g., 0.049974 for D435i).
            min_depth: Minimum valid depth in meters.
            max_depth: Maximum valid depth in meters.
            filter_size: Kernel size for local disparity filtering (default 3).
            inlier_thred_range: Threshold for disparity match inliers (default (0.01, 0.05)).
            prob_range: Probability of pixel matching (default (0.4, 0.6)).
            invalid_disp: Sentinel value for invalid disparities (default 1e7).
        """
        super().__init__()
        self.focal_length = focal_length
        self.baseline = baseline
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.invalid_disp = invalid_disp
        self.inlier_thred_range = inlier_thred_range
        self.prob_range = prob_range
        self.filter_size = filter_size

        # Precompute filter weights once (reuse across batches)
        weights, substitutes = self._compute_weights(filter_size)
        self.register_buffer('weights', weights.view(1, 1, filter_size, filter_size))
        self.register_buffer('substitutes', substitutes.view(1, 1, filter_size, filter_size))

    def _compute_weights(self, filter_size: int) -> tuple:
        """Compute radial weights for local disparity filtering.
        
        Args:
            filter_size: Kernel size (e.g., 3, 5).
            
        Returns:
            (weights, substitutes) normalized kernel tensors.
        """
        center = filter_size // 2
        idx = torch.arange(filter_size) - center
        x_filter, y_filter = torch.meshgrid(idx, idx, indexing='ij')
        sqr_radius = x_filter ** 2 + y_filter ** 2
        sqrt_radius = torch.sqrt(sqr_radius)
        
        # Radial inverse distance weighting
        weights = 1 / torch.where(sqr_radius == 0, torch.ones_like(sqrt_radius), sqrt_radius)
        weights = weights / weights.sum()
        
        # Spatial substitution mask for filling
        fill_weights = 1 / (1 + sqrt_radius)
        fill_weights = torch.where(sqr_radius > filter_size, -1.0, fill_weights)
        substitutes = (fill_weights > 0).float()

        return weights, substitutes

    def filter_disparity(self, disparity: torch.Tensor) -> torch.Tensor:
        """Apply realistic stereo matching noise to disparity map.
        
        Simulates:
        - Random pixel matching failures
        - Noisy disparity at matched pixels
        - Spatial filtering to fill invalid regions
        
        Args:
            disparity: Tensor of shape (B, 1, H, W).
            
        Returns:
            Noisy disparity map with invalid pixels set to self.invalid_disp.
        """
        B, _, H, W = disparity.shape
        device = disparity.device
        center = self.filter_size // 2

        output_disparity = torch.full_like(disparity, self.invalid_disp)

        # Randomly select which pixels will be matched (prob_range % of pixels)
        prob = (
            torch.rand(B, 1, 1, 1, device=device)
            * (self.prob_range[1] - self.prob_range[0])
            + self.prob_range[0]
        )
        random_mask = (torch.rand(B, 1, H, W, device=device) < prob)

        # Compute local mean disparity (moving average)
        weighted_disparity = F.conv2d(disparity, self.weights, padding=center)

        # Compute differences from local mean (normalized)
        differences = torch.abs(disparity - weighted_disparity)

        # Normalize differences using per-batch statistics
        differences_flat = differences.view(B, -1)
        mean_diff = torch.mean(differences_flat, dim=1, keepdim=True)
        std_diff = torch.std(differences_flat, dim=1, keepdim=True) + 1e-6

        normalized_differences_flat = (differences_flat - mean_diff) / std_diff
        normalized_differences = normalized_differences_flat.view_as(differences)

        # Inlier threshold: random per-batch inlier_thred_range
        threshold = (
            torch.rand(B, 1, 1, 1, device=device)
            * (self.inlier_thred_range[1] - self.inlier_thred_range[0])
            + self.inlier_thred_range[0]
        )
        update_mask = (normalized_differences < threshold) & random_mask

        # Quantize matched disparities to 1/32 precision (typical stereo resolution)
        disparity_quantized = torch.round(disparity * 32.0) / 32.0

        # Place quantized disparity where matched
        output_disparity = torch.where(update_mask, disparity_quantized, output_disparity)

        # Fill unmatched pixels using neighboring matched disparities
        filled_values = F.conv2d(update_mask.float() * disparity_quantized, self.substitutes, padding=center)
        counts = F.conv2d(update_mask.float(), self.substitutes, padding=center) + 1e-9
        average_filled_values = filled_values / counts
        output_disparity = torch.where(counts >= 1, average_filled_values, output_disparity)

        return output_disparity

    def forward(self, depth: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Apply depth noise and clipping.
        
        Args:
            depth: Depth tensor of shape (B, 1, H, W) or (B, H, W).
            add_noise: If False, only clipping is applied (default True).
            
        Returns:
            Noisy depth tensor, shape (B, 1, H, W).
        """
        # Ensure 4D shape
        if len(depth.shape) == 3:
            depth = depth.unsqueeze(1)

        assert depth.shape[1] == 1, f"Expected shape (B, 1, H, W), got {depth.shape}."
        assert len(depth.shape) == 4, f"Expected 4D tensor, got {depth.shape}."

        depth = depth.clone()  # Avoid in-place modification of input

        if add_noise:
            # Clamp to avoid division by zero in depth→disparity conversion
            depth = torch.clamp(depth, min=1.0 / self.invalid_disp)

            # Step 1: Convert depth to disparity
            disparity = self.focal_length * self.baseline / depth

            # Step 2: Apply stereo matching noise (spatially filtered)
            filtered_disparity = self.filter_disparity(disparity)

            # Step 3: Convert noisy disparity back to depth
            depth = self.focal_length * self.baseline / filtered_disparity

            # Step 4: Clamp below min_depth to 0 (unobservable)
            depth[depth < self.min_depth] = 0.0

        # Step 5: Clamp above max_depth to 0 (out of range)
        depth[depth > self.max_depth] = 0.0

        return depth
