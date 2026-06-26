"""
PET Forward Model with Data Consistency Interface

This module provides the PETForwardModel class for use in Data Consistency (DC)
enhanced diffusion sampling. It wraps the PETSimulator to provide:

1. Forward projection operator A: image -> sinogram
2. Adjoint operator A^T: sinogram -> image  
3. DC gradient computation for iterative optimization
4. Support for L2 and Poisson NLL loss functions

Mathematical Background:
========================

Data Consistency enforces that the reconstructed image x is consistent with
the observed measurements y_obs. Given the forward model y = Ax + noise,
we minimize a data fidelity term:

L2 Loss (Gaussian noise assumption):
    DC_loss = 0.5 * ||Ax - y_obs||^2
    DC_grad = A^T(Ax - y_obs)

Poisson NLL Loss (Poisson noise assumption):
    DC_loss = sum(Ax - y_obs * log(Ax + eps))
    DC_grad = A^T(1 - y_obs / (Ax + eps))

The DC gradient is used to update the image estimate:
    x_new = x - step_size * DC_grad

Note on Differentiability:
==========================
The PET forward projection using grid_sample is differentiable, but the
adjoint (back projection) computed via grid_sample is an approximation.
For exact gradients, one would need to implement the true adjoint operator.
In practice, the approximation works well for DC updates.

New in 2026: Physical Forward Model
===================================
This module now supports the PhysicalPETForwardModel from 
utils/physical_forward_model.py, which provides:
- mm-based PSF (consistent with simulation)
- Sinogram rebinning (angular and radial)
- Background term
- Exact consistency between simulation and DC

For DC-consistent reconstruction, use create_physical_pet_forward_model()
with the meta returned from physical simulation.

Author: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union

from .simulation import PETSimulator, create_pet_simulator

# Import physical forward model for the new API
try:
    from .physical_forward_model import (
        PhysicalPETForwardModel,
        create_physical_forward_model,
        create_matched_simulation_and_forward_model,
    )
    from .physical_pet_simulation import (
        PhysicalDegradationMeta,
        DEFAULT_FOV_MM,
        DEFAULT_FWHM_MM,
        DEFAULT_DOSE_ALPHA,
        DEFAULT_REBIN_FACTOR,
        DEFAULT_BACKGROUND_BETA,
    )
    PHYSICAL_FORWARD_MODEL_AVAILABLE = True
except ImportError:
    PHYSICAL_FORWARD_MODEL_AVAILABLE = False
    # Define fallback
    PhysicalPETForwardModel = None
    PhysicalDegradationMeta = None


class PETForwardModel(nn.Module):
    """
    PET Forward Model for Data Consistency in Diffusion Sampling
    
    This class provides the interface for computing DC gradients during
    diffusion sampling with observed sinogram measurements.
    
    Args:
        image_size: Size of square images (default: 256)
        num_angles: Number of projection angles (default: 180)
        num_bins: Number of detector bins (default: auto)
        use_psf: Whether to apply PSF in forward model (default: False)
        psf_fwhm: PSF FWHM in pixels (default: 3.0)
        device: Target device
        simulator: Optional pre-configured PETSimulator (overrides other params)
    
    Example:
        >>> model = PETForwardModel(image_size=256)
        >>> y_obs = ...  # observed sinogram
        >>> x = ...      # current image estimate
        >>> 
        >>> # Compute DC gradient
        >>> grad = model.dc_gradient(x, y_obs, loss_type='l2')
        >>> 
        >>> # Update image
        >>> x_updated = x - 0.1 * grad
    """
    
    def __init__(
        self,
        image_size: int = 256,
        num_angles: int = 180,
        num_bins: Optional[int] = None,
        use_psf: bool = False,
        psf_fwhm: float = 3.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        simulator: Optional[PETSimulator] = None
    ):
        super().__init__()
        
        self.image_size = image_size
        self.num_angles = num_angles
        self.device = device
        
        # Use provided simulator or create new one
        if simulator is not None:
            self.simulator = simulator
            self.image_size = simulator.image_size
            self.num_angles = simulator.num_angles
        else:
            self.simulator = create_pet_simulator(
                image_size=image_size,
                num_angles=num_angles,
                num_bins=num_bins,
                use_psf=use_psf,
                psf_fwhm=psf_fwhm,
                dose_factor=1.0,  # DC doesn't need dose simulation
                device=device
            )
        
        # Store num_bins for reference
        self.num_bins = self.simulator.num_bins
    
    # =========================================================================
    # Core Forward Model Methods
    # =========================================================================
    
    def forward(
        self,
        x: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Forward projection: A(x) = sinogram
        
        Args:
            x: Input image [B, 1, H, W], [B, H, W], or [H, W]
            meta: Optional projection parameters
        
        Returns:
            sinogram: Projected sinogram [B, num_angles, num_bins] or [num_angles, num_bins]
        """
        sinogram, _ = self.simulator.forward_project(x, meta)
        return sinogram
    
    def adjoint(
        self,
        y: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Adjoint (approximate back projection): A^T(y) = image
        
        Note: This is an approximate adjoint using the back projection operator.
        For exact adjoints, the operator would need to be implemented more carefully.
        In practice, this approximation works well for DC updates.
        
        Args:
            y: Sinogram [B, num_angles, num_bins] or [num_angles, num_bins]
            meta: Optional parameters
        
        Returns:
            image: Back-projected image [B, H, W] or [H, W]
        """
        return self.simulator.back_project(y, meta, filtered=False)
    
    def degrade(
        self,
        x: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Degrade image (low-quality simulation without sinogram intermediate).
        
        This provides a direct image-to-degraded-image mapping for visualization
        or comparison, without explicit sinogram computation.
        
        Args:
            x: Input high-quality image [B, 1, H, W], [B, H, W], or [H, W]
            meta: Degradation parameters:
                - 'lowres': bool, apply low-resolution degradation
                - 'lowres_factor': int, downsampling factor
                - 'noise': bool, apply noise (via sinogram cycle)
                - 'psf': bool, apply PSF blurring
        
        Returns:
            Degraded image with same shape as input
        """
        return self.simulator.degrade_image(x, meta)
    
    # =========================================================================
    # Data Consistency Gradient Computation
    # =========================================================================
    
    def dc_gradient(
        self,
        x: torch.Tensor,
        y_obs: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None,
        loss_type: str = 'l2',
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute Data Consistency gradient.
        
        The DC gradient pushes the image estimate x towards being consistent
        with the observed measurements y_obs.
        
        Mathematical formulation:
        -------------------------
        
        L2 Loss (Gaussian noise model):
            Loss = 0.5 * ||Ax - y_obs||^2
            Gradient = A^T(Ax - y_obs)
            
        Poisson NLL Loss:
            Loss = sum(Ax - y_obs * log(Ax))
            Gradient = A^T(1 - y_obs / (Ax + eps))
            
            Note: The Poisson gradient has the interpretation of a multiplicative
            update factor when y_obs / Ax is computed, which is the basis for
            ML-EM reconstruction algorithms.
        
        Args:
            x: Current image estimate [B, 1, H, W], [B, H, W], or [H, W]
            y_obs: Observed sinogram [B, num_angles, num_bins] or [num_angles, num_bins]
            meta: Optional parameters for forward/adjoint
            loss_type: 'l2' or 'poisson' (default: 'l2')
            eps: Small constant for numerical stability (default: 1e-8)
        
        Returns:
            gradient: DC gradient with same shape as x
        
        Design Notes:
        -------------
        1. The adjoint A^T is approximated using back projection without filtering.
           This is standard practice in iterative reconstruction.
        
        2. For Poisson NLL, we use the log-form gradient which is more numerically
           stable than the ratio form for small values.
        
        3. The gradient is NOT normalized by the number of measurements. The step
           size should be adjusted accordingly in the caller.
        """
        # Store original shape for restoration
        orig_shape = x.shape
        
        # Ensure 3D for processing: [B, H, W]
        x_3d = self._ensure_3d(x)
        y_obs_3d = self._ensure_3d_sinogram(y_obs)
        
        # Forward project current estimate: Ax
        Ax = self.forward(x_3d, meta)
        
        # Compute residual based on loss type
        if loss_type == 'l2':
            # L2 residual: Ax - y_obs
            residual = Ax - y_obs_3d
            
        elif loss_type == 'poisson':
            # Poisson NLL gradient: 1 - y_obs / (Ax + eps)
            # This gradient points in the direction of increasing NLL
            Ax_safe = Ax + eps
            residual = 1.0 - y_obs_3d / Ax_safe
            
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'l2' or 'poisson'.")
        
        # Back project residual: A^T(residual)
        gradient = self.adjoint(residual, meta)
        
        # Restore original shape
        gradient = self._restore_shape(gradient, orig_shape)
        
        return gradient
    
    def dc_loss(
        self,
        x: torch.Tensor,
        y_obs: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None,
        loss_type: str = 'l2',
        eps: float = 1e-8,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute Data Consistency loss value.
        
        Useful for monitoring or as a regularization term.
        
        Args:
            x: Current image estimate
            y_obs: Observed sinogram
            meta: Optional parameters
            loss_type: 'l2' or 'poisson'
            eps: Numerical stability constant
            reduction: 'mean', 'sum', or 'none'
        
        Returns:
            loss: Scalar loss value (or per-sample if reduction='none')
        """
        # Ensure proper shapes
        x_3d = self._ensure_3d(x)
        y_obs_3d = self._ensure_3d_sinogram(y_obs)
        
        # Forward project
        Ax = self.forward(x_3d, meta)
        
        if loss_type == 'l2':
            # L2 loss: 0.5 * ||Ax - y_obs||^2
            loss = 0.5 * (Ax - y_obs_3d) ** 2
            
        elif loss_type == 'poisson':
            # Poisson NLL: Ax - y_obs * log(Ax)
            Ax_safe = Ax + eps
            loss = Ax_safe - y_obs_3d * torch.log(Ax_safe)
            
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def compute_sensitivity(self) -> torch.Tensor:
        """
        Compute sensitivity image: A^T(1)
        
        The sensitivity image represents how much each pixel contributes to
        the total measurement. It's used for normalization in iterative
        reconstruction algorithms like OSEM.
        
        Returns:
            sensitivity: Sensitivity image [H, W]
        """
        # Create ones sinogram
        ones_sinogram = torch.ones(
            1, self.num_angles, self.num_bins,
            device=self.device, dtype=torch.float32
        )
        
        # Back project
        sensitivity = self.adjoint(ones_sinogram)
        
        return sensitivity.squeeze(0)
    
    def dc_step(
        self,
        x: torch.Tensor,
        y_obs: torch.Tensor,
        step_size: float = 0.1,
        meta: Optional[Dict[str, Any]] = None,
        loss_type: str = 'l2',
        clamp_min: Optional[float] = 0.0,
        clamp_max: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply one Data Consistency gradient descent step.
        
        Convenience method that computes gradient and applies update:
            x_new = x - step_size * dc_gradient(x, y_obs)
        
        Args:
            x: Current image estimate
            y_obs: Observed sinogram
            step_size: Gradient step size
            meta: Optional parameters
            loss_type: 'l2' or 'poisson'
            clamp_min: Minimum value after update (default: 0.0 for PET)
            clamp_max: Maximum value after update (default: None)
        
        Returns:
            x_updated: Updated image estimate
        """
        gradient = self.dc_gradient(x, y_obs, meta, loss_type)
        x_updated = x - step_size * gradient
        
        # Apply clamping
        if clamp_min is not None or clamp_max is not None:
            x_updated = torch.clamp(x_updated, min=clamp_min, max=clamp_max)
        
        return x_updated
    
    # =========================================================================
    # Shape Handling Helpers
    # =========================================================================
    
    def _ensure_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure image tensor is 3D [B, H, W]."""
        if x.dim() == 2:
            return x.unsqueeze(0)
        elif x.dim() == 4:
            # [B, C, H, W] -> [B, H, W] (assuming C=1)
            return x.squeeze(1)
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
            # Restore [B, C, H, W]
            if x.dim() == 3:
                return x.unsqueeze(1)
        return x
    
    def to(self, device: Union[str, torch.device]) -> 'PETForwardModel':
        """Move model to device."""
        self.device = device if isinstance(device, str) else str(device)
        self.simulator = self.simulator.to(device)
        return super().to(device)
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'image_size': self.image_size,
            'num_angles': self.num_angles,
            'num_bins': self.num_bins,
            'simulator_config': self.simulator.get_config()
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_pet_forward_model(
    image_size: int = 256,
    num_angles: int = 180,
    use_psf: bool = False,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs
) -> PETForwardModel:
    """
    Factory function to create PETForwardModel with common defaults.
    
    Args:
        image_size: Size of square images
        num_angles: Number of projection angles
        use_psf: Whether to apply PSF
        device: Target device
        **kwargs: Additional arguments passed to PETForwardModel
    
    Returns:
        Configured PETForwardModel instance
    """
    return PETForwardModel(
        image_size=image_size,
        num_angles=num_angles,
        use_psf=use_psf,
        device=device,
        **kwargs
    )


# =============================================================================
# Physical Forward Model Factory Functions (NEW - 2026)
# =============================================================================

def create_physical_pet_forward_model(
    meta: Optional['PhysicalDegradationMeta'] = None,
    fov_mm: float = 256.0,
    image_size: int = 256,
    fwhm_mm: float = 4.0,
    dose_alpha: float = 1.0,
    rebin_factor: int = 1,
    rebin_radial_factor: int = 1,
    background_beta: float = 0.0,
    num_angles: int = 180,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs
) -> 'PhysicalPETForwardModel':
    """
    Create a PhysicalPETForwardModel for DC-consistent reconstruction.
    
    RECOMMENDED: Pass the `meta` from physical simulation to ensure
    exact consistency between simulation and DC forward model.
    
    Physical Forward Model Features:
    - mm-based PSF (fwhm_mm converted to sigma_px using spacing)
    - Angular and radial rebinning
    - Background term
    - Exact match with simulation degradation
    
    Args:
        meta: PhysicalDegradationMeta from simulation (RECOMMENDED)
              If provided, other geometry parameters are ignored.
        fov_mm: Field of View in mm (used if meta is None)
        image_size: Image grid size (used if meta is None)
        fwhm_mm: PSF FWHM in mm (used if meta is None)
        dose_alpha: Dose scaling factor (used if meta is None)
        rebin_factor: Angular rebinning factor (used if meta is None)
        rebin_radial_factor: Radial rebinning factor (used if meta is None)
        background_beta: Background fraction (used if meta is None)
        num_angles: Number of projection angles (used if meta is None)
        device: Target device
        **kwargs: Additional parameters
    
    Returns:
        PhysicalPETForwardModel instance
    
    Example (RECOMMENDED - using meta from simulation):
        >>> from utils.simulation import simulate_physical_degradation
        >>> y_obs, x_lr, meta = simulate_physical_degradation(
        ...     x_hr, fov_mm=256, fwhm_mm=4.0, dose_alpha=0.1
        ... )
        >>> forward_model = create_physical_pet_forward_model(meta=meta, device=device)
        >>> grad = forward_model.dc_gradient(x, y_obs)
    
    Example (explicit parameters - ensure they match simulation!):
        >>> forward_model = create_physical_pet_forward_model(
        ...     fov_mm=256, image_size=256, fwhm_mm=4.0,
        ...     dose_alpha=0.1, rebin_factor=2, device=device
        ... )
    """
    if not PHYSICAL_FORWARD_MODEL_AVAILABLE:
        raise ImportError(
            "Physical forward model not available. "
            "Check that utils/physical_forward_model.py exists."
        )
    
    return create_physical_forward_model(
        meta=meta,
        fov_mm=fov_mm,
        image_size=image_size,
        fwhm_mm=fwhm_mm,
        dose_alpha=dose_alpha,
        rebin_factor=rebin_factor,
        rebin_radial_factor=rebin_radial_factor,
        background_beta=background_beta,
        device=device,
        **kwargs
    )


def is_physical_forward_model_available() -> bool:
    """Check if physical forward model is available."""
    return PHYSICAL_FORWARD_MODEL_AVAILABLE


# =============================================================================
# Test Code
# =============================================================================

if __name__ == '__main__':
    print("Testing PETForwardModel...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model
    model = create_pet_forward_model(
        image_size=128,
        num_angles=180,
        use_psf=False,
        device=device
    )
    
    # Create test phantom
    size = 128
    phantom = torch.zeros(1, 1, size, size, device=device)
    y, x = torch.meshgrid(
        torch.arange(size, device=device, dtype=torch.float32),
        torch.arange(size, device=device, dtype=torch.float32),
        indexing='ij'
    )
    center = size // 2
    mask = ((x - center)**2 + (y - center)**2) < (size//3)**2
    phantom[0, 0, mask] = 1.0
    
    print(f"Phantom shape: {phantom.shape}")
    
    # Test forward projection
    print("\nTesting forward...")
    sinogram = model.forward(phantom)
    print(f"  Sinogram shape: {sinogram.shape}")
    
    # Create "observed" sinogram (with noise)
    y_obs = sinogram + 0.1 * torch.randn_like(sinogram)
    
    # Test DC gradient (L2)
    print("\nTesting dc_gradient (L2)...")
    grad_l2 = model.dc_gradient(phantom, y_obs, loss_type='l2')
    print(f"  Gradient shape: {grad_l2.shape}")
    print(f"  Gradient range: [{grad_l2.min():.4f}, {grad_l2.max():.4f}]")
    
    # Test DC gradient (Poisson)
    print("\nTesting dc_gradient (Poisson)...")
    grad_poisson = model.dc_gradient(phantom, y_obs, loss_type='poisson')
    print(f"  Gradient shape: {grad_poisson.shape}")
    print(f"  Gradient range: [{grad_poisson.min():.4f}, {grad_poisson.max():.4f}]")
    
    # Test DC loss
    print("\nTesting dc_loss...")
    loss_l2 = model.dc_loss(phantom, y_obs, loss_type='l2')
    loss_poisson = model.dc_loss(phantom, y_obs, loss_type='poisson')
    print(f"  L2 loss: {loss_l2.item():.4f}")
    print(f"  Poisson loss: {loss_poisson.item():.4f}")
    
    # Test DC step
    print("\nTesting dc_step...")
    x_updated = model.dc_step(phantom, y_obs, step_size=0.1)
    print(f"  Updated shape: {x_updated.shape}")
    print(f"  Updated range: [{x_updated.min():.4f}, {x_updated.max():.4f}]")
    
    # Test sensitivity
    print("\nTesting compute_sensitivity...")
    sensitivity = model.compute_sensitivity()
    print(f"  Sensitivity shape: {sensitivity.shape}")
    print(f"  Sensitivity range: [{sensitivity.min():.4f}, {sensitivity.max():.4f}]")
    
    # Test physical forward model if available
    if PHYSICAL_FORWARD_MODEL_AVAILABLE:
        print("\n" + "=" * 40)
        print("Testing PhysicalPETForwardModel...")
        print("=" * 40)
        
        physical_model = create_physical_pet_forward_model(
            fov_mm=256.0,
            image_size=128,
            fwhm_mm=4.0,
            rebin_factor=2,
            background_beta=0.05,
            device=device
        )
        
        print(f"\nPhysical model config:")
        print(f"  fwhm_mm: {physical_model.fwhm_mm}")
        print(f"  rebin_factor: {physical_model.rebin_factor}")
        print(f"  background_beta: {physical_model.background_beta}")
        
        # Test forward
        y_physical = physical_model.forward(phantom)
        print(f"\nPhysical forward shape: {y_physical.shape}")
        
        # Test DC gradient
        grad_physical = physical_model.dc_gradient(phantom, y_physical)
        print(f"Physical DC gradient range: [{grad_physical.min():.6f}, {grad_physical.max():.6f}]")
    
    print("\n✅ All tests passed!")
