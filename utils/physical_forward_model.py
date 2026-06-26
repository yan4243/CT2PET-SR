"""
Physical PET Forward Model for Data Consistency

This module provides the PhysicalPETForwardModel class that ensures the EXACT
same forward degradation is used in both:
1. Generating observed measurements (y_obs) during simulation
2. Computing DC gradients during reconstruction

Key Design Principle:
=====================
The forward model A includes ALL degradation operations in a single operator:
    y = A(x) = Rebin(Background(Project(PSF(x))))

The DC gradient is computed as:
    ∇_x DC = A^T(A(x) - y_obs)  for L2 loss
    ∇_x DC = A^T(1 - y_obs / A(x))  for Poisson loss

Both A and A^T use the same physical parameters stored in PhysicalDegradationMeta.

Author: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Union, Any, Callable
from dataclasses import asdict

from .physical_pet_simulation import (
    PhysicalPETProjector,
    PhysicalPETSimulator,
    PhysicalDegradationMeta,
    create_gaussian_kernel_2d,
    compute_sigma_px,
    compute_spacing_mm,
    DEFAULT_FOV_MM,
    DEFAULT_HR_SIZE,
    DEFAULT_FWHM_MM,
    DEFAULT_DOSE_ALPHA,
    DEFAULT_REBIN_FACTOR,
    DEFAULT_BACKGROUND_BETA,
    DEFAULT_NUM_ANGLES,
    DEFAULT_SCALE_COUNTS,
    FWHM_TO_SIGMA,
)


class PhysicalPETForwardModel(nn.Module):
    """
    Physical PET Forward Model for Data Consistency
    
    This class provides the forward operator A and adjoint A^T that are
    EXACTLY consistent with the degradation used in simulation.
    
    The forward model includes:
    - PSF blurring (mm-based FWHM)
    - Forward projection (Radon transform)
    - Angular rebinning
    - Radial rebinning  
    - Background addition
    
    All parameters are stored in PhysicalDegradationMeta and must match
    exactly between simulation and DC.
    
    Args:
        meta: PhysicalDegradationMeta from simulation (ensures consistency)
        device: Target device
        
    Alternative constructor (from explicit parameters):
        PhysicalPETForwardModel.from_params(...)
    
    Example:
        >>> # Option 1: Create from meta (RECOMMENDED for DC)
        >>> y_obs, x_lr, meta = simulator.degrade_pet(x_hr, ...)
        >>> forward_model = PhysicalPETForwardModel(meta, device)
        >>> grad = forward_model.dc_gradient(x, y_obs)
        
        >>> # Option 2: Create from explicit parameters
        >>> forward_model = PhysicalPETForwardModel.from_params(
        ...     fov_mm=256, image_size=256, fwhm_mm=4.0, ...
        ... )
    """
    
    def __init__(
        self,
        meta: PhysicalDegradationMeta,
        device: Union[str, torch.device] = 'cpu'
    ):
        super().__init__()
        
        self.meta = meta
        self.device = device if isinstance(device, str) else str(device)
        
        # Extract key parameters from meta
        self.fov_mm = meta.fov_mm
        self.image_size = meta.hr_size  # Operating at HR size
        self.fwhm_mm = meta.fwhm_mm
        self.sigma_px = meta.sigma_hr_px
        self.dose_alpha = meta.dose_alpha
        self.rebin_factor = meta.rebin_factor
        self.rebin_radial_factor = meta.rebin_radial_factor
        self.background_beta = meta.background_beta
        self.num_angles = meta.num_angles
        self.num_angles_rebinned = meta.num_angles_rebinned
        self.num_bins = meta.num_bins
        self.num_bins_rebinned = meta.num_bins_rebinned
        self.scale_counts = meta.scale_counts
        
        # Create projector matching the simulation geometry
        # The projector operates at HR size but produces rebinned output
        self._projector = PhysicalPETProjector(
            fov_mm=self.fov_mm,
            image_size=self.image_size,
            num_angles=self.num_angles,
            num_bins=self.num_bins,
            fwhm_mm=self.fwhm_mm,
            rebin_factor=self.rebin_factor,
            rebin_radial_factor=self.rebin_radial_factor,
            background_beta=self.background_beta,
            device=device
        )
        
        # Setup PSF kernel (for explicit control)
        if self.fwhm_mm > 0 and self.sigma_px > 0.1:
            kernel = create_gaussian_kernel_2d(self.sigma_px, device=device)
            self.register_buffer('psf_kernel', kernel)
            self.psf_padding = kernel.shape[-1] // 2
        else:
            self.psf_kernel = None
            self.psf_padding = 0
        
        # Pre-compute sensitivity image for normalization
        self._sensitivity = None
    
    @classmethod
    def from_params(
        cls,
        fov_mm: float = DEFAULT_FOV_MM,
        image_size: int = DEFAULT_HR_SIZE,
        fwhm_mm: float = DEFAULT_FWHM_MM,
        dose_alpha: float = DEFAULT_DOSE_ALPHA,
        rebin_factor: int = DEFAULT_REBIN_FACTOR,
        rebin_radial_factor: int = 1,
        background_beta: float = DEFAULT_BACKGROUND_BETA,
        num_angles: int = DEFAULT_NUM_ANGLES,
        num_bins: Optional[int] = None,
        scale_counts: float = DEFAULT_SCALE_COUNTS,
        device: Union[str, torch.device] = 'cpu'
    ) -> 'PhysicalPETForwardModel':
        """
        Create PhysicalPETForwardModel from explicit parameters.
        
        This is useful when you don't have a meta from simulation, but
        note that you must ensure the parameters EXACTLY match any
        simulation you want to be consistent with.
        
        Args:
            fov_mm: Field of View in mm
            image_size: Image grid size
            fwhm_mm: PSF FWHM in mm
            dose_alpha: Dose scaling factor
            rebin_factor: Angular rebinning factor
            rebin_radial_factor: Radial rebinning factor
            background_beta: Background fraction
            num_angles: Number of projection angles
            num_bins: Number of detector bins
            scale_counts: Counts scaling for Poisson
            device: Target device
        
        Returns:
            PhysicalPETForwardModel instance
        """
        if num_bins is None:
            num_bins = int(np.ceil(image_size * np.sqrt(2)))
        
        spacing_mm = compute_spacing_mm(fov_mm, image_size)
        sigma_px = compute_sigma_px(fwhm_mm, spacing_mm) if fwhm_mm > 0 else 0.0
        
        meta = PhysicalDegradationMeta(
            fov_mm=fov_mm,
            hr_size=image_size,
            lr_size=image_size,
            spacing_hr_mm=spacing_mm,
            spacing_lr_mm=spacing_mm,
            fwhm_mm=fwhm_mm,
            sigma_hr_px=sigma_px,
            sigma_lr_px=sigma_px,
            dose_alpha=dose_alpha,
            rebin_factor=rebin_factor,
            rebin_radial_factor=rebin_radial_factor,
            background_beta=background_beta,
            num_angles=num_angles,
            num_angles_rebinned=max(1, num_angles // rebin_factor),
            num_bins=num_bins,
            num_bins_rebinned=max(1, num_bins // rebin_radial_factor),
            scale_counts=scale_counts
        )
        
        return cls(meta, device)
    
    @classmethod
    def from_simulator(
        cls,
        simulator: PhysicalPETSimulator,
        dose_alpha: float = DEFAULT_DOSE_ALPHA,
        rebin_factor: int = DEFAULT_REBIN_FACTOR,
        rebin_radial_factor: int = 1,
        background_beta: float = DEFAULT_BACKGROUND_BETA,
        device: Optional[Union[str, torch.device]] = None
    ) -> 'PhysicalPETForwardModel':
        """
        Create PhysicalPETForwardModel from a PhysicalPETSimulator.
        
        This ensures geometry consistency with the simulator.
        
        Args:
            simulator: PhysicalPETSimulator instance
            dose_alpha: Dose scaling for this forward model
            rebin_factor: Angular rebinning factor
            rebin_radial_factor: Radial rebinning factor
            background_beta: Background fraction
            device: Target device (uses simulator's device if None)
        
        Returns:
            PhysicalPETForwardModel instance
        """
        if device is None:
            device = simulator.device
        
        return cls.from_params(
            fov_mm=simulator.fov_mm,
            image_size=simulator.hr_size,
            fwhm_mm=simulator.fwhm_mm,
            dose_alpha=dose_alpha,
            rebin_factor=rebin_factor,
            rebin_radial_factor=rebin_radial_factor,
            background_beta=background_beta,
            num_angles=simulator.num_angles,
            num_bins=simulator.num_bins,
            scale_counts=simulator.scale_counts,
            device=device
        )
    
    # =========================================================================
    # Forward and Adjoint Operators
    # =========================================================================
    
    def forward(
        self,
        x: torch.Tensor,
        apply_psf: bool = True,
        apply_rebin: bool = True,
        add_background: bool = True
    ) -> torch.Tensor:
        """
        Forward projection: A(x) = sinogram
        
        Applies the complete forward model:
        1. PSF blurring (if apply_psf and fwhm_mm > 0)
        2. Radon transform
        3. Angular rebinning (if apply_rebin and rebin_factor > 1)
        4. Radial rebinning (if apply_rebin and rebin_radial_factor > 1)
        5. Add background (if add_background and background_beta > 0)
        
        Args:
            x: Input image [B, C, H, W], [B, H, W], or [H, W]
            apply_psf: Whether to apply PSF (default: True)
            apply_rebin: Whether to apply rebinning (default: True)
            add_background: Whether to add background (default: True)
        
        Returns:
            sinogram: [B, num_angles, num_bins] or [num_angles, num_bins]
        """
        # Handle dimensions
        orig_shape = x.shape
        squeeze_batch = False
        
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze_batch = True
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        
        batch_size = x.shape[0]
        
        # Step 1: Apply PSF
        if apply_psf and self.psf_kernel is not None:
            x = F.conv2d(x, self.psf_kernel.to(x.dtype), padding=self.psf_padding)
        
        # Steps 2-4: Forward project with rebinning
        sinogram = self._projector.forward_project(
            x, apply_psf=False, apply_rebin=apply_rebin
        )
        
        # Step 5: Add background
        if add_background and self.background_beta > 0:
            mean_signal = sinogram.mean()
            background = self.background_beta * mean_signal
            sinogram = sinogram + background
        
        if squeeze_batch:
            sinogram = sinogram.squeeze(0)
        
        return sinogram
    
    def adjoint(
        self,
        y: torch.Tensor,
        apply_psf_adjoint: bool = True
    ) -> torch.Tensor:
        """
        Adjoint (back projection): A^T(y) = image
        
        Applies the adjoint of the forward model:
        1. Back projection
        2. PSF adjoint (PSF is self-adjoint for symmetric kernel)
        
        Note: Rebinning adjoint is approximated by using the rebinned angles
        in back projection, which is what the projector does.
        
        Args:
            y: Sinogram [B, num_angles, num_bins] or [num_angles, num_bins]
            apply_psf_adjoint: Whether to apply PSF adjoint (default: True)
        
        Returns:
            image: [B, H, W] or [H, W]
        """
        squeeze_batch = False
        if y.dim() == 2:
            y = y.unsqueeze(0)
            squeeze_batch = True
        
        # Back projection (uses rebinned angles if applicable)
        image = self._projector.back_project(
            y, filtered=False, target_size=self.image_size
        )
        
        # Apply PSF adjoint (Gaussian PSF is self-adjoint)
        if apply_psf_adjoint and self.psf_kernel is not None:
            # Add channel dimension for conv2d
            image_4d = image.unsqueeze(1) if image.dim() == 3 else image.unsqueeze(0).unsqueeze(0)
            image_4d = F.conv2d(image_4d, self.psf_kernel.to(image.dtype), padding=self.psf_padding)
            image = image_4d.squeeze(1) if image.dim() == 3 else image_4d.squeeze(0).squeeze(0)
        
        if squeeze_batch:
            image = image.squeeze(0)
        
        return image
    
    def fbp(self, y: torch.Tensor) -> torch.Tensor:
        """
        Filtered Back Projection for reconstruction.
        
        Note: This is for visualization/initialization only.
        DC uses the unfiltered adjoint.
        
        Args:
            y: Sinogram
        
        Returns:
            Reconstructed image
        """
        squeeze_batch = False
        if y.dim() == 2:
            y = y.unsqueeze(0)
            squeeze_batch = True
        
        image = self._projector.fbp(y, target_size=self.image_size)
        
        if squeeze_batch:
            image = image.squeeze(0)
        
        return image
    
    # =========================================================================
    # Data Consistency Gradient Computation
    # =========================================================================
    
    def dc_gradient(
        self,
        x: torch.Tensor,
        y_obs: torch.Tensor,
        loss_type: str = 'l2',
        eps: float = 1e-8,
        apply_psf: bool = True,
        add_background: bool = True
    ) -> torch.Tensor:
        """
        Compute Data Consistency gradient.
        
        The DC gradient pushes x towards being consistent with y_obs.
        
        Mathematical formulation:
        -------------------------
        
        L2 Loss (Gaussian noise model):
            Loss = 0.5 * ||A(x) - y_obs||^2
            Gradient = A^T(A(x) - y_obs)
        
        Poisson NLL Loss:
            Loss = sum(A(x) - y_obs * log(A(x)))
            Gradient = A^T(1 - y_obs / (A(x) + eps))
        
        Args:
            x: Current image estimate [B, C, H, W], [B, H, W], or [H, W]
            y_obs: Observed sinogram (must match simulation output shape)
            loss_type: 'l2' or 'poisson'
            eps: Numerical stability constant
            apply_psf: Whether to apply PSF in forward model
            add_background: Whether to add background in forward model
        
        Returns:
            gradient: DC gradient with same shape as x
        
        Note:
            The gradient is NOT normalized. The step size should be adjusted
            based on the image/gradient scales.
        """
        # Store original shape
        orig_shape = x.shape
        
        # Ensure 3D for processing
        x_3d = self._ensure_3d(x)
        y_obs_3d = self._ensure_3d_sinogram(y_obs)
        
        # Forward project: A(x)
        Ax = self.forward(x_3d, apply_psf=apply_psf, add_background=add_background)
        
        # Compute residual based on loss type
        if loss_type == 'l2':
            # L2 residual: A(x) - y_obs
            residual = Ax - y_obs_3d
            
        elif loss_type == 'poisson':
            # Poisson NLL gradient: 1 - y_obs / (A(x) + eps)
            Ax_safe = Ax + eps
            residual = 1.0 - y_obs_3d / Ax_safe
            
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'l2' or 'poisson'.")
        
        # Back project residual: A^T(residual)
        gradient = self.adjoint(residual, apply_psf_adjoint=apply_psf)
        
        # Restore original shape
        gradient = self._restore_shape(gradient, orig_shape)
        
        return gradient
    
    def dc_loss(
        self,
        x: torch.Tensor,
        y_obs: torch.Tensor,
        loss_type: str = 'l2',
        eps: float = 1e-8,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute Data Consistency loss value.
        
        Args:
            x: Current image estimate
            y_obs: Observed sinogram
            loss_type: 'l2' or 'poisson'
            eps: Numerical stability constant
            reduction: 'mean', 'sum', or 'none'
        
        Returns:
            loss: Scalar loss value
        """
        x_3d = self._ensure_3d(x)
        y_obs_3d = self._ensure_3d_sinogram(y_obs)
        
        Ax = self.forward(x_3d)
        
        if loss_type == 'l2':
            loss = 0.5 * (Ax - y_obs_3d) ** 2
        elif loss_type == 'poisson':
            Ax_safe = Ax + eps
            loss = Ax_safe - y_obs_3d * torch.log(Ax_safe)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def dc_step(
        self,
        x: torch.Tensor,
        y_obs: torch.Tensor,
        step_size: float = 0.1,
        loss_type: str = 'l2',
        clamp_min: Optional[float] = 0.0,
        clamp_max: Optional[float] = None,
        normalize_gradient: bool = True
    ) -> torch.Tensor:
        """
        Apply one Data Consistency gradient descent step.
        
        x_new = x - step_size * gradient
        
        Args:
            x: Current image estimate
            y_obs: Observed sinogram
            step_size: Gradient step size
            loss_type: 'l2' or 'poisson'
            clamp_min: Minimum value after update (default: 0.0 for PET)
            clamp_max: Maximum value after update (default: None)
            normalize_gradient: Whether to normalize gradient magnitude
        
        Returns:
            x_updated: Updated image estimate
        """
        gradient = self.dc_gradient(x, y_obs, loss_type)
        
        # Optionally normalize gradient
        if normalize_gradient:
            grad_max = torch.abs(gradient).max() + 1e-10
            x_max = torch.abs(x).max() + 1e-10
            scale = x_max / grad_max
            gradient = gradient * scale
        
        x_updated = x - step_size * gradient
        
        # Apply clamping
        if clamp_min is not None or clamp_max is not None:
            x_updated = torch.clamp(x_updated, min=clamp_min, max=clamp_max)
        
        return x_updated
    
    # =========================================================================
    # Route 1 API: HR-to-LR measurement DC primitives
    # These methods make explicit that the forward model handles ALL
    # resolution mismatch internally. The DC wrapper should call these
    # instead of applying any lowres hacks on the image side.
    # =========================================================================

    def forward_project_hr_to_lr_measurement(
        self,
        x_hr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full HR-to-LR forward projection: y_hat = A_LR(H(x_HR)) + r
        
        Internally applies:
          1. Resolution degradation H (PSF blur controlled by fwhm_mm)
          2. LR projection geometry A_LR (Radon + rebinning)
          3. Additive background r
        
        Input is always the HR image; resolution mismatch is handled here.
        
        Args:
            x_hr: HR image in physical space [B, C, H, W], [B, H, W], or [H, W]
        
        Returns:
            y_hat: Predicted LR measurement [B, num_angles_rebinned, num_bins_rebinned]
        """
        return self.forward(x_hr, apply_psf=True, apply_rebin=True, add_background=True)

    def dc_gradient_wrt_x_hr(
        self,
        x_hr: torch.Tensor,
        y_obs: torch.Tensor,
        loss_type: str = 'l2',
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        DC gradient w.r.t. HR image: grad = H^T A^T (residual)
        
        The full adjoint chain propagates through both the projection
        adjoint A^T and the PSF adjoint H^T (symmetric blur => H^T = H).
        
        For L2:      grad = H^T A^T ( A(Hx) + r - y_obs )
        For Poisson: grad = H^T A^T ( 1 - y_obs / (A(Hx) + r + eps) )
        
        The input must be the HR image in physical space. No lowres hacks.
        
        Args:
            x_hr: HR image in physical space [B, C, H, W] or compatible
            y_obs: Observed LR sinogram (must match forward projection shape)
            loss_type: 'l2' or 'poisson'
            eps: Numerical stability for Poisson
        
        Returns:
            gradient: DC gradient w.r.t. x_hr, same shape as x_hr
        """
        grad = self.dc_gradient(
            x_hr, y_obs, loss_type=loss_type, eps=eps,
            apply_psf=True, add_background=True
        )

        # --- sanity checks (cheap, always on) ---
        assert grad.shape == x_hr.shape, (
            f"DC gradient shape {grad.shape} != input shape {x_hr.shape}"
        )
        if not torch.isfinite(grad).all():
            import warnings
            warnings.warn(
                "[Route1] DC gradient contains NaN/Inf! "
                f"grad range [{grad.min():.4g}, {grad.max():.4g}]"
            )
        return grad

    def poisson_gradient(
        self,
        x_hr: torch.Tensor,
        y_obs: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Convenience: Poisson NLL gradient w.r.t. HR image.
        
        Equivalent to dc_gradient_wrt_x_hr(x_hr, y_obs, loss_type='poisson').
        """
        return self.dc_gradient_wrt_x_hr(x_hr, y_obs, loss_type='poisson', eps=eps)

    # =========================================================================
    # Sensitivity and Utility Methods
    # =========================================================================
    
    def compute_sensitivity(self) -> torch.Tensor:
        """
        Compute sensitivity image: A^T(1)
        
        Returns:
            sensitivity: Sensitivity image [H, W]
        """
        if self._sensitivity is not None:
            return self._sensitivity
        
        # Create ones sinogram with proper dimensions
        ones_sino = torch.ones(
            1, self.num_angles_rebinned, self.num_bins_rebinned,
            device=self.device, dtype=torch.float32
        )
        
        sensitivity = self.adjoint(ones_sino, apply_psf_adjoint=True)
        sensitivity = sensitivity.squeeze(0)
        
        # Cache for reuse
        self._sensitivity = sensitivity
        
        return sensitivity
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration as dict."""
        return {
            'meta': self.meta.to_dict(),
            'device': self.device
        }
    
    # =========================================================================
    # Shape Handling Helpers
    # =========================================================================
    
    def _ensure_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure image tensor is 3D [B, H, W]."""
        if x.dim() == 2:
            return x.unsqueeze(0)
        elif x.dim() == 4:
            return x.squeeze(1)  # Remove channel dim
        return x
    
    def _ensure_3d_sinogram(self, y: torch.Tensor) -> torch.Tensor:
        """Ensure sinogram tensor is 3D [B, num_angles, num_bins]."""
        if y.dim() == 2:
            return y.unsqueeze(0)
        return y
    
    def _restore_shape(self, x: torch.Tensor, orig_shape: Tuple) -> torch.Tensor:
        """Restore tensor to original shape."""
        if len(orig_shape) == 2:
            return x.squeeze(0)
        elif len(orig_shape) == 4:
            if x.dim() == 3:
                return x.unsqueeze(1)
        return x
    
    def to(self, device: Union[str, torch.device]) -> 'PhysicalPETForwardModel':
        """Move to device."""
        self.device = device if isinstance(device, str) else str(device)
        self._projector = self._projector.to(device)
        self._sensitivity = None  # Clear cached sensitivity
        return super().to(device)


# =============================================================================
# Factory Functions
# =============================================================================

def create_physical_forward_model(
    meta: Optional[PhysicalDegradationMeta] = None,
    fov_mm: float = DEFAULT_FOV_MM,
    image_size: int = DEFAULT_HR_SIZE,
    fwhm_mm: float = DEFAULT_FWHM_MM,
    dose_alpha: float = DEFAULT_DOSE_ALPHA,
    rebin_factor: int = DEFAULT_REBIN_FACTOR,
    background_beta: float = DEFAULT_BACKGROUND_BETA,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs
) -> PhysicalPETForwardModel:
    """
    Factory function to create PhysicalPETForwardModel.
    
    If meta is provided, uses those parameters (recommended for DC).
    Otherwise creates from explicit parameters.
    
    Args:
        meta: PhysicalDegradationMeta from simulation (takes precedence)
        fov_mm: Field of View in mm
        image_size: Image grid size
        fwhm_mm: PSF FWHM in mm
        dose_alpha: Dose scaling factor
        rebin_factor: Angular rebinning factor
        background_beta: Background fraction
        device: Target device
        **kwargs: Additional parameters passed to from_params
    
    Returns:
        PhysicalPETForwardModel instance
    """
    if meta is not None:
        return PhysicalPETForwardModel(meta, device)
    else:
        return PhysicalPETForwardModel.from_params(
            fov_mm=fov_mm,
            image_size=image_size,
            fwhm_mm=fwhm_mm,
            dose_alpha=dose_alpha,
            rebin_factor=rebin_factor,
            background_beta=background_beta,
            device=device,
            **kwargs
        )


def create_matched_simulation_and_forward_model(
    fov_mm: float = DEFAULT_FOV_MM,
    hr_size: int = DEFAULT_HR_SIZE,
    fwhm_mm: float = DEFAULT_FWHM_MM,
    num_angles: int = DEFAULT_NUM_ANGLES,
    scale_counts: float = DEFAULT_SCALE_COUNTS,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple['PhysicalPETSimulator', 'Callable']:
    """
    Create a simulator and a factory for matched forward models.
    
    This ensures that forward models created for DC will exactly match
    the simulator's geometry.
    
    Args:
        fov_mm: Field of View in mm
        hr_size: High-resolution grid size
        fwhm_mm: PSF FWHM in mm
        num_angles: Number of projection angles
        scale_counts: Counts scaling for Poisson
        device: Target device
    
    Returns:
        simulator: PhysicalPETSimulator instance
        forward_model_factory: Function that creates matched forward models
    
    Example:
        >>> simulator, fm_factory = create_matched_simulation_and_forward_model(...)
        >>> y_obs, x_lr, meta = simulator.degrade_pet(x_hr, dose_alpha=0.1, ...)
        >>> forward_model = fm_factory(meta)  # Exactly matches simulation
        >>> grad = forward_model.dc_gradient(x, y_obs)
    """
    from .physical_pet_simulation import PhysicalPETSimulator
    
    simulator = PhysicalPETSimulator(
        fov_mm=fov_mm,
        hr_size=hr_size,
        fwhm_mm=fwhm_mm,
        num_angles=num_angles,
        scale_counts=scale_counts,
        device=device
    )
    
    def forward_model_factory(meta: PhysicalDegradationMeta) -> PhysicalPETForwardModel:
        return PhysicalPETForwardModel(meta, device)
    
    return simulator, forward_model_factory


# =============================================================================
# Test Code
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Physical PET Forward Model")
    print("=" * 60)
    
    from .physical_pet_simulation import PhysicalPETSimulator
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Physical parameters
    fov_mm = 256.0
    hr_size = 128
    fwhm_mm = 4.0
    dose_alpha = 0.1
    rebin_factor = 2
    background_beta = 0.05
    
    print(f"\nPhysical parameters:")
    print(f"  FOV: {fov_mm} mm")
    print(f"  Image size: {hr_size}")
    print(f"  FWHM: {fwhm_mm} mm")
    print(f"  Dose alpha: {dose_alpha}")
    print(f"  Rebin factor: {rebin_factor}")
    print(f"  Background beta: {background_beta}")
    
    # Create test phantom
    phantom = torch.zeros(1, 1, hr_size, hr_size, device=device)
    center = hr_size // 2
    y, x = torch.meshgrid(
        torch.arange(hr_size, device=device, dtype=torch.float32),
        torch.arange(hr_size, device=device, dtype=torch.float32),
        indexing='ij'
    )
    mask = ((x - center)**2 + (y - center)**2) < (hr_size//4)**2
    phantom[0, 0, mask] = 1.0
    
    print(f"\nPhantom shape: {phantom.shape}")
    
    # Create simulator
    simulator = PhysicalPETSimulator(
        fov_mm=fov_mm,
        hr_size=hr_size,
        fwhm_mm=fwhm_mm,
        device=device
    )
    
    # Simulate degradation
    print("\n" + "-" * 40)
    print("Simulating degradation...")
    y_obs, x_lr, meta = simulator.degrade_pet(
        phantom,
        dose_alpha=dose_alpha,
        rebin_factor=rebin_factor,
        background_beta=background_beta
    )
    
    print(f"y_obs shape: {y_obs.shape}")
    print(f"x_lr shape: {x_lr.shape}")
    print(f"Meta: {meta}")
    
    # Create forward model from meta (CRITICAL: uses same parameters)
    print("\n" + "-" * 40)
    print("Creating matched forward model...")
    forward_model = PhysicalPETForwardModel(meta, device)
    
    print(f"Forward model image_size: {forward_model.image_size}")
    print(f"Forward model rebin_factor: {forward_model.rebin_factor}")
    print(f"Forward model background_beta: {forward_model.background_beta}")
    
    # Test forward projection
    print("\n" + "-" * 40)
    print("Testing forward projection...")
    y_forward = forward_model.forward(phantom)
    print(f"y_forward shape: {y_forward.shape}")
    print(f"y_forward range: [{y_forward.min():.4f}, {y_forward.max():.4f}]")
    
    # Test DC gradient
    print("\n" + "-" * 40)
    print("Testing DC gradient...")
    
    # Use the degraded image as current estimate
    x_current = x_lr.unsqueeze(0).unsqueeze(0) if x_lr.dim() == 2 else x_lr.unsqueeze(1) if x_lr.dim() == 3 else x_lr
    
    grad_l2 = forward_model.dc_gradient(x_current, y_obs, loss_type='l2')
    print(f"L2 gradient shape: {grad_l2.shape}")
    print(f"L2 gradient range: [{grad_l2.min():.6f}, {grad_l2.max():.6f}]")
    
    grad_poisson = forward_model.dc_gradient(x_current, y_obs, loss_type='poisson')
    print(f"Poisson gradient shape: {grad_poisson.shape}")
    print(f"Poisson gradient range: [{grad_poisson.min():.6f}, {grad_poisson.max():.6f}]")
    
    # Test DC loss
    print("\n" + "-" * 40)
    print("Testing DC loss...")
    loss_l2 = forward_model.dc_loss(x_current, y_obs, loss_type='l2')
    loss_poisson = forward_model.dc_loss(x_current, y_obs, loss_type='poisson')
    print(f"L2 loss: {loss_l2.item():.4f}")
    print(f"Poisson loss: {loss_poisson.item():.4f}")
    
    # Test DC step
    print("\n" + "-" * 40)
    print("Testing DC step...")
    x_updated = forward_model.dc_step(x_current, y_obs, step_size=0.1)
    print(f"x_updated shape: {x_updated.shape}")
    print(f"x_updated range: [{x_updated.min():.4f}, {x_updated.max():.4f}]")
    
    # Verify loss decreased
    loss_after = forward_model.dc_loss(x_updated, y_obs, loss_type='l2')
    print(f"L2 loss after DC step: {loss_after.item():.4f}")
    print(f"Loss decreased: {loss_after.item() < loss_l2.item()}")
    
    # Test sensitivity
    print("\n" + "-" * 40)
    print("Testing sensitivity image...")
    sensitivity = forward_model.compute_sensitivity()
    print(f"Sensitivity shape: {sensitivity.shape}")
    print(f"Sensitivity range: [{sensitivity.min():.4f}, {sensitivity.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
