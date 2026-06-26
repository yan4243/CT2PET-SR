"""
PET Simulation Module

This module provides a unified interface for PET image simulation, including:
- Image degradation (low-dose, low-resolution)
- Forward projection (sinogram generation)
- Measurement simulation

The code is reorganized from utils/sim/ for cleaner API and integration
with the Data Consistency (DC) sampling framework.

Note: The physical modeling logic is preserved from the original implementation.
Only the interface is restructured.

New in 2026: Physical Degradation Model
=======================================
This module now supports physically meaningful degradation via the 
degrade_pet() function and PhysicalPETSimulator class from 
utils/physical_pet_simulation.py. The physical model includes:
- mm-based PSF (fwhm_mm, spacing_mm)
- Projection-domain Poisson noise (dose_alpha)
- Sinogram rebinning (rebin_factor)
- Background term (background_beta)

For DC-consistent simulation, use the physical model which ensures
the same forward operator is used in simulation and DC gradient.

Author: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Union, Any
import warnings

# Import physical simulation module for the new API
try:
    from .physical_pet_simulation import (
        degrade_pet,
        PhysicalPETSimulator,
        PhysicalPETProjector,
        PhysicalDegradationMeta,
        create_physical_simulator,
        create_physical_projector,
        compute_spacing_mm,
        compute_sigma_px,
        DEFAULT_FOV_MM,
        DEFAULT_HR_SIZE,
        DEFAULT_FWHM_MM,
        DEFAULT_DOSE_ALPHA,
        DEFAULT_REBIN_FACTOR,
        DEFAULT_BACKGROUND_BETA,
        DEFAULT_NUM_ANGLES,
        DEFAULT_SCALE_COUNTS,
    )
    PHYSICAL_SIMULATION_AVAILABLE = True
except ImportError:
    PHYSICAL_SIMULATION_AVAILABLE = False
    # Define fallback constants if physical simulation not available
    DEFAULT_FOV_MM = 256.0
    DEFAULT_HR_SIZE = 256
    DEFAULT_FWHM_MM = 4.0
    DEFAULT_DOSE_ALPHA = 1.0
    DEFAULT_REBIN_FACTOR = 1
    DEFAULT_BACKGROUND_BETA = 0.0
    DEFAULT_NUM_ANGLES = 180
    DEFAULT_SCALE_COUNTS = 1e6


class PETSimulator(nn.Module):
    """
    Unified PET Simulation Interface
    
    This class provides a clean API for simulating PET measurements and
    degraded images. It wraps the projection, noise, and resolution models
    into a single coherent interface.
    
    Features:
    - Forward projection (Radon transform)
    - Back projection (adjoint / FBP)
    - Low-dose simulation (Poisson noise)
    - Low-resolution simulation (downsampling)
    - Optional PSF modeling
    
    Args:
        image_size: Size of square images (default: 256)
        num_angles: Number of projection angles (default: 180)
        num_bins: Number of detector bins (default: auto, sqrt(2)*image_size)
        dose_factor: Low-dose simulation factor (1.0 = full dose, 0.1 = 10% dose)
        scale_factor: Counts scaling factor for Poisson simulation
        resolution_factor: Downsampling factor for low-res simulation (1 = no downsampling)
        use_psf: Whether to apply PSF blurring
        psf_fwhm: FWHM of PSF in pixels
        device: 'cuda' or 'cpu'
    
    Example:
        >>> simulator = PETSimulator(image_size=256, dose_factor=0.1)
        >>> degraded = simulator.degrade_image(pet_image, meta={'noise': True, 'lowres': True})
        >>> sinogram, aux = simulator.forward_project(pet_image, meta={})
        >>> measurement = simulator.simulate_measurement(pet_image, meta={'add_noise': True})
    """
    
    def __init__(
        self,
        image_size: int = 256,
        num_angles: int = 180,
        num_bins: Optional[int] = None,
        dose_factor: float = 1.0,
        scale_factor: float = 1e6,
        resolution_factor: int = 1,
        use_psf: bool = False,
        psf_fwhm: float = 3.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.image_size = image_size
        self.num_angles = num_angles
        self.num_bins = num_bins if num_bins else int(np.ceil(image_size * np.sqrt(2)))
        self.dose_factor = dose_factor
        self.scale_factor = scale_factor
        self.resolution_factor = resolution_factor
        self.use_psf = use_psf
        self.psf_fwhm = psf_fwhm
        self.device = device
        
        # Initialize projection geometry
        self._setup_geometry()
        
        # Initialize PSF kernel if needed
        if use_psf:
            self._setup_psf_kernel()
    
    def _setup_geometry(self):
        """Pre-compute projection geometry matrices."""
        # Projection angles (0 to pi)
        angles = torch.linspace(0, np.pi, self.num_angles, device=self.device)
        self.register_buffer('angles', angles)
        
        # Detector positions
        det_pos = torch.linspace(
            -self.num_bins // 2,
            self.num_bins // 2,
            self.num_bins,
            device=self.device
        )
        self.register_buffer('det_pos', det_pos)
        
        # Image coordinates for back projection
        coords = torch.linspace(
            -self.image_size // 2,
            self.image_size // 2,
            self.image_size,
            device=self.device
        )
        y_coords, x_coords = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('x_coords', x_coords)
        self.register_buffer('y_coords', y_coords)
    
    def _setup_psf_kernel(self):
        """Setup PSF convolution kernel (Gaussian)."""
        sigma = self.psf_fwhm / 2.355  # FWHM to sigma conversion
        kernel_size = int(np.ceil(self.psf_fwhm * 3)) | 1  # Ensure odd size
        
        x = torch.arange(kernel_size, device=self.device) - kernel_size // 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        # Shape for conv2d: [out_ch, in_ch, H, W]
        self.register_buffer('psf_kernel', kernel_2d.unsqueeze(0).unsqueeze(0))
        self.psf_padding = kernel_size // 2
    
    def apply_psf(self, image: torch.Tensor) -> torch.Tensor:
        """Apply PSF blurring to image."""
        if not self.use_psf:
            return image
        
        # Handle different input dimensions
        squeeze_batch = False
        squeeze_channel = False
        
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
            squeeze_batch = True
            squeeze_channel = True
        elif image.dim() == 3:
            image = image.unsqueeze(1)
            squeeze_channel = True
        
        blurred = F.conv2d(image, self.psf_kernel, padding=self.psf_padding)
        
        if squeeze_channel:
            blurred = blurred.squeeze(1)
        if squeeze_batch:
            blurred = blurred.squeeze(0)
        
        return blurred
    
    # =========================================================================
    # Main API Methods
    # =========================================================================
    
    def degrade_image(
        self,
        x: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Degrade a high-quality PET image to simulate low-quality acquisition.
        
        This creates a degraded version of the input image by applying:
        1. Low-resolution simulation (if enabled)
        2. Noise via forward/back projection cycle (if enabled)
        3. PSF blurring (if enabled)
        
        Args:
            x: Input high-quality PET image [B, C, H, W] or [B, H, W] or [H, W]
            meta: Optional dict with degradation parameters:
                - 'lowres': bool, apply low-resolution degradation (default: False)
                - 'lowres_factor': int, downsampling factor (default: self.resolution_factor)
                - 'noise': bool, apply noise via sinogram (default: False)
                - 'dose_factor': float, override default dose factor
                - 'psf': bool, apply PSF blurring (default: self.use_psf)
        
        Returns:
            Degraded image with same shape as input
        """
        meta = meta or {}
        
        # Parse options
        apply_lowres = meta.get('lowres', False)
        lowres_factor = meta.get('lowres_factor', self.resolution_factor)
        apply_noise = meta.get('noise', False)
        dose_factor = meta.get('dose_factor', self.dose_factor)
        apply_psf = meta.get('psf', self.use_psf)
        
        # Ensure proper shape
        x_orig_shape = x.shape
        x = self._ensure_4d(x)
        
        # Apply low-resolution degradation
        if apply_lowres and lowres_factor > 1:
            x = self._apply_lowres(x, lowres_factor)
        
        # Apply noise via sinogram cycle
        if apply_noise:
            x = self._apply_noise_via_sinogram(x, dose_factor)
        
        # Apply PSF
        if apply_psf:
            x = self.apply_psf(x)
        
        # Restore original shape
        x = self._restore_shape(x, x_orig_shape)
        
        return x
    
    def forward_project(
        self,
        x: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward projection: image -> sinogram
        
        Computes the Radon transform of the input image.
        
        Args:
            x: Input PET image [B, C, H, W], [B, H, W], or [H, W]
            meta: Optional dict with projection parameters:
                - 'apply_psf': bool, apply PSF before projection (default: self.use_psf)
        
        Returns:
            sinogram: Projection data [B, num_angles, num_bins] or [num_angles, num_bins]
            aux: Dict with auxiliary information:
                - 'image_shape': original image shape
                - 'sinogram_shape': output sinogram shape
        """
        meta = meta or {}
        apply_psf = meta.get('apply_psf', self.use_psf)
        
        # Store original shape
        orig_shape = x.shape
        squeeze_batch = False
        
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze_batch = True
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        
        batch_size = x.shape[0]
        
        # Apply PSF if enabled
        if apply_psf:
            x = self.apply_psf(x)
        
        # Forward projection using rotation-based method (fast, differentiable)
        sinograms = []
        
        for theta in self.angles:
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            
            # Create rotation matrix
            rotation_matrix = torch.tensor([
                [cos_t, -sin_t, 0],
                [sin_t, cos_t, 0]
            ], device=self.device, dtype=x.dtype).unsqueeze(0).expand(batch_size, -1, -1)
            
            # Generate affine grid
            grid = F.affine_grid(rotation_matrix, x.shape, align_corners=False)
            
            # Rotate image
            rotated = F.grid_sample(x, grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=False)
            
            # Sum along columns (projection)
            projection = rotated.sum(dim=2)  # [B, 1, W]
            sinograms.append(projection)
        
        sinogram = torch.cat(sinograms, dim=1)  # [B, num_angles, num_bins]
        
        # Adjust size if needed
        if sinogram.shape[-1] != self.num_bins:
            sinogram = F.interpolate(
                sinogram, size=self.num_bins, mode='linear', align_corners=False
            )
        
        if squeeze_batch:
            sinogram = sinogram.squeeze(0)
        
        aux = {
            'image_shape': orig_shape,
            'sinogram_shape': sinogram.shape
        }
        
        return sinogram, aux
    
    def back_project(
        self,
        sinogram: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None,
        filtered: bool = False
    ) -> torch.Tensor:
        """
        Back projection: sinogram -> image
        
        Computes the adjoint of the Radon transform (or FBP if filtered=True).
        
        Args:
            sinogram: Projection data [B, num_angles, num_bins] or [num_angles, num_bins]
            meta: Optional dict (reserved for future use)
            filtered: If True, apply ramp filter for FBP reconstruction
        
        Returns:
            image: Reconstructed image [B, H, W] or [H, W]
        """
        squeeze_batch = False
        if sinogram.dim() == 2:
            sinogram = sinogram.unsqueeze(0)
            squeeze_batch = True
        
        batch_size = sinogram.shape[0]
        
        # Apply ramp filter if requested
        if filtered:
            sinogram = self._apply_ramp_filter(sinogram)
        
        # Initialize output
        image = torch.zeros(
            batch_size, 1, self.image_size, self.image_size,
            device=self.device, dtype=sinogram.dtype
        )
        
        # Back projection
        for i, theta in enumerate(self.angles):
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            
            # Create smearing grid - use same dtype as sinogram
            y_grid = torch.linspace(-1, 1, self.image_size, device=self.device, dtype=sinogram.dtype)
            x_grid = torch.linspace(-1, 1, self.image_size, device=self.device, dtype=sinogram.dtype)
            yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
            
            # Projection position (normalized)
            s = xx * cos_t.to(sinogram.dtype) + yy * sin_t.to(sinogram.dtype)
            
            # Sampling grid
            grid_x = s
            grid_y = torch.zeros_like(s)
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            
            # Sample from sinogram slice
            sino_slice = sinogram[:, i:i+1, :].unsqueeze(1)  # [B, 1, 1, num_bins]
            sino_2d = sino_slice.expand(-1, -1, self.image_size, -1)
            
            sampled = F.grid_sample(
                sino_2d, grid, mode='bilinear',
                padding_mode='zeros', align_corners=False
            )
            
            image += sampled
        
        # Normalize
        image = image * (np.pi / self.num_angles)
        
        if squeeze_batch:
            image = image.squeeze(0).squeeze(0)
        else:
            image = image.squeeze(1)
        
        return image
    
    def fbp(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Filtered Back Projection (convenience wrapper)."""
        return self.back_project(sinogram, filtered=True)
    
    def simulate_measurement(
        self,
        x: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Simulate a complete PET measurement from image to noisy sinogram.
        
        Pipeline: image -> (PSF) -> forward_project -> (Poisson noise) -> sinogram
        
        Args:
            x: Input PET activity image [B, C, H, W], [B, H, W], or [H, W]
            meta: Optional dict with simulation parameters:
                - 'add_noise': bool, add Poisson noise (default: True)
                - 'dose_factor': float, dose factor for noise (default: self.dose_factor)
                - 'apply_psf': bool, apply PSF (default: self.use_psf)
                - 'return_clean': bool, also return clean sinogram (default: False)
        
        Returns:
            sinogram: (Noisy) sinogram [B, num_angles, num_bins] or [num_angles, num_bins]
            aux: Dict with auxiliary information:
                - 'clean_sinogram': Clean sinogram (if return_clean=True)
                - 'image_shape': original image shape
                - 'sinogram_shape': output sinogram shape
                - 'noise_std': estimated noise standard deviation
        """
        meta = meta or {}
        add_noise = meta.get('add_noise', True)
        dose_factor = meta.get('dose_factor', self.dose_factor)
        return_clean = meta.get('return_clean', False)
        
        # Forward project
        sinogram_clean, aux = self.forward_project(x, meta)
        
        # Add Poisson noise if requested
        if add_noise:
            sinogram, noise_info = self._add_poisson_noise(sinogram_clean, dose_factor)
            aux['noise_std'] = noise_info['noise_std']
        else:
            sinogram = sinogram_clean
            aux['noise_std'] = 0.0
        
        if return_clean:
            aux['clean_sinogram'] = sinogram_clean
        
        return sinogram, aux
    
    def simulate_degradation(
        self,
        x: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Simulate complete degradation: GT → (degraded image, observed sinogram).
        
        This is the recommended method for DC sampling as it:
        1. Performs forward projection only ONCE
        2. Avoids cumulative errors from separate degrade + project operations
        3. Returns both the degraded image (via FBP) and the observed sinogram
        
        Pipeline:
            GT image → (optional PSF) → (optional lowres) → forward project 
            → add Poisson noise → sinogram (y_obs)
            → FBP → degraded image (lr_pet)
        
        Args:
            x: Input high-quality PET image [B, C, H, W] or [B, H, W] or [H, W]
            meta: Optional dict with degradation parameters:
                - 'lowres': bool, apply low-resolution degradation (default: False)
                - 'lowres_factor': int, downsampling factor (default: self.resolution_factor)
                - 'add_noise': bool, add Poisson noise (default: True)
                - 'dose_factor': float, dose factor for noise (default: self.dose_factor)
                - 'apply_psf': bool, apply PSF (default: self.use_psf)
                - 'return_clean_sinogram': bool, also return clean sinogram (default: False)
        
        Returns:
            degraded_image: Degraded PET image (via FBP from noisy sinogram)
            y_obs: Observed sinogram (noisy, used for DC)
            aux: Dict with auxiliary information:
                - 'clean_sinogram': Clean sinogram (if return_clean_sinogram=True)
                - 'image_shape': original image shape
                - 'sinogram_shape': sinogram shape
                - 'noise_std': estimated noise standard deviation
                - 'dose_factor': actual dose factor used
                - 'lowres_factor': actual lowres factor used
        """
        meta = meta or {}
        
        # Parse options
        apply_lowres = meta.get('lowres', False)
        lowres_factor = meta.get('lowres_factor', self.resolution_factor)
        add_noise = meta.get('add_noise', True)
        dose_factor = meta.get('dose_factor', self.dose_factor)
        apply_psf = meta.get('apply_psf', self.use_psf)
        return_clean = meta.get('return_clean_sinogram', False)
        
        # Store original shape
        x_orig_shape = x.shape
        x = self._ensure_4d(x)
        
        # Step 1: Apply low-resolution degradation (in image domain)
        if apply_lowres and lowres_factor > 1:
            x = self._apply_lowres(x, lowres_factor)
        
        # Step 2: Apply PSF blurring (before projection)
        if apply_psf:
            x = self.apply_psf(x)
        
        # Step 3: Forward project (ONLY ONCE)
        sinogram_clean, proj_aux = self.forward_project(x, {'apply_psf': False})
        
        # Step 4: Add Poisson noise to sinogram
        if add_noise:
            y_obs, noise_info = self._add_poisson_noise(sinogram_clean, dose_factor)
            noise_std = noise_info['noise_std']
        else:
            y_obs = sinogram_clean
            noise_std = 0.0
        
        # Step 5: FBP to get degraded image (consistent with y_obs)
        degraded_image = self.fbp(y_obs)
        
        # Ensure non-negative
        degraded_image = torch.clamp(degraded_image, min=0)
        
        # Restore channel dimension if needed
        if degraded_image.dim() == 3:
            degraded_image = degraded_image.unsqueeze(1)
        
        # Restore original shape for degraded image
        degraded_image = self._restore_shape(degraded_image, x_orig_shape)
        
        # Build aux dict
        aux = {
            'image_shape': x_orig_shape,
            'sinogram_shape': y_obs.shape,
            'noise_std': noise_std,
            'dose_factor': dose_factor,
            'lowres_factor': lowres_factor if apply_lowres else 1,
        }
        
        if return_clean:
            aux['clean_sinogram'] = sinogram_clean
        
        return degraded_image, y_obs, aux
    
    # =========================================================================
    # Internal Helper Methods
    # =========================================================================
    
    def _ensure_4d(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is 4D [B, C, H, W]."""
        if x.dim() == 2:
            return x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            return x.unsqueeze(1)
        return x
    
    def _restore_shape(self, x: torch.Tensor, orig_shape: Tuple) -> torch.Tensor:
        """Restore tensor to original shape."""
        if len(orig_shape) == 2:
            return x.squeeze(0).squeeze(0)
        elif len(orig_shape) == 3:
            return x.squeeze(1)
        return x
    
    def _apply_lowres(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """Apply low-resolution degradation (downsample then upsample)."""
        orig_size = x.shape[-1]
        lr_size = orig_size // factor
        
        # Downsample
        x_lr = F.interpolate(x, size=(lr_size, lr_size), mode='bilinear', align_corners=False)
        
        # Upsample back
        x_up = F.interpolate(x_lr, size=(orig_size, orig_size), mode='bilinear', align_corners=False)
        
        return x_up
    
    def _apply_noise_via_sinogram(self, x: torch.Tensor, dose_factor: float) -> torch.Tensor:
        """Apply noise by forward projecting, adding Poisson noise, and back projecting."""
        sinogram, _ = self.forward_project(x, {'apply_psf': False})
        noisy_sinogram, _ = self._add_poisson_noise(sinogram, dose_factor)
        recon = self.fbp(noisy_sinogram)
        
        # Restore channel dimension
        if recon.dim() == 3:
            recon = recon.unsqueeze(1)
        
        return torch.clamp(recon, min=0)
    
    def _add_poisson_noise(
        self,
        sinogram: torch.Tensor,
        dose_factor: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Add Poisson noise to sinogram."""
        # Scale to counts
        counts = sinogram * self.scale_factor * dose_factor
        counts = torch.clamp(counts, min=0)
        
        # Add Poisson noise
        if self.training or not torch.is_grad_enabled():
            # During training, use actual Poisson sampling
            noisy_counts = torch.poisson(counts)
        else:
            # During inference with gradients, use Gaussian approximation
            # This is differentiable but approximate
            std = torch.sqrt(counts + 1e-8)
            noisy_counts = counts + std * torch.randn_like(counts)
            noisy_counts = torch.clamp(noisy_counts, min=0)
        
        # Scale back
        noisy_sinogram = noisy_counts / self.scale_factor
        
        # Estimate noise statistics
        noise_std = float(torch.std(noisy_sinogram - sinogram).item()) if sinogram.numel() > 0 else 0.0
        
        return noisy_sinogram, {'noise_std': noise_std}
    
    def _apply_ramp_filter(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Apply ramp filter in frequency domain for FBP."""
        # FFT along detector dimension
        sino_fft = torch.fft.fft(sinogram, dim=-1)
        
        # Ramp filter (|frequency|)
        freqs = torch.fft.fftfreq(self.num_bins, device=self.device)
        ramp = torch.abs(freqs)
        
        # Apply filter
        filtered_fft = sino_fft * ramp.unsqueeze(0).unsqueeze(0)
        
        # Inverse FFT
        filtered = torch.fft.ifft(filtered_fft, dim=-1).real
        
        return filtered
    
    # =========================================================================
    # Convenience Properties
    # =========================================================================
    
    @property
    def output_sinogram_shape(self) -> Tuple[int, int]:
        """Return the shape of output sinograms (num_angles, num_bins)."""
        return (self.num_angles, self.num_bins)
    
    def get_config(self) -> Dict[str, Any]:
        """Return simulator configuration as dict."""
        return {
            'image_size': self.image_size,
            'num_angles': self.num_angles,
            'num_bins': self.num_bins,
            'dose_factor': self.dose_factor,
            'scale_factor': self.scale_factor,
            'resolution_factor': self.resolution_factor,
            'use_psf': self.use_psf,
            'psf_fwhm': self.psf_fwhm,
            'device': str(self.device)
        }
    
    def to(self, device: Union[str, torch.device]) -> 'PETSimulator':
        """Move simulator to device."""
        self.device = device if isinstance(device, str) else str(device)
        return super().to(device)


# =============================================================================
# Factory function for easy creation
# =============================================================================

def create_pet_simulator(
    image_size: int = 256,
    num_angles: int = 180,
    dose_factor: float = 1.0,
    use_psf: bool = False,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs
) -> PETSimulator:
    """
    Factory function to create a PETSimulator with common defaults.
    
    Args:
        image_size: Size of square images
        num_angles: Number of projection angles
        dose_factor: Low-dose simulation factor
        use_psf: Whether to apply PSF blurring
        device: Target device
        **kwargs: Additional arguments passed to PETSimulator
    
    Returns:
        Configured PETSimulator instance
    """
    return PETSimulator(
        image_size=image_size,
        num_angles=num_angles,
        dose_factor=dose_factor,
        use_psf=use_psf,
        device=device,
        **kwargs
    )


# =============================================================================
# Physical PET Simulation Wrapper (NEW - 2026)
# =============================================================================
# 
# The functions below provide a unified interface to the new physical
# degradation model. They wrap the PhysicalPETSimulator class for convenience.
#
# Key advantages over legacy PETSimulator:
# 1. mm-based PSF (FWHM in millimeters, not pixels)
# 2. Explicit FOV and voxel spacing
# 3. Projection-domain Poisson noise (not image domain)
# 4. Sinogram rebinning (angular and radial)
# 5. Background/scatter proxy
# 6. Consistent with DC forward model
#

def simulate_physical_degradation(
    x_hr: torch.Tensor,
    fov_mm: float = DEFAULT_FOV_MM,
    fwhm_mm: float = DEFAULT_FWHM_MM,
    dose_alpha: float = DEFAULT_DOSE_ALPHA,
    rebin_factor: int = DEFAULT_REBIN_FACTOR,
    rebin_radial_factor: int = 1,
    background_beta: float = DEFAULT_BACKGROUND_BETA,
    num_angles: int = DEFAULT_NUM_ANGLES,
    scale_counts: float = DEFAULT_SCALE_COUNTS,
    return_sinogram: bool = True,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """
    Simulate physically meaningful PET degradation.
    
    This is a convenience wrapper for degrade_pet() from the physical
    simulation module. It provides a unified interface for generating
    low-dose and low-resolution PET data with explicit physical parameters.
    
    Physical Degradation Model:
    ---------------------------
    1. PSF blur: Gaussian with FWHM in mm (sigma_px = fwhm_mm / (2.355 * spacing_mm))
    2. Forward projection: Radon transform
    3. Angular rebinning: Combine adjacent angles
    4. Radial rebinning: Pool detector bins
    5. Background: Add beta * mean(sinogram) before noise
    6. Poisson noise: y ~ Poisson(dose_alpha * y_ideal)
    7. FBP reconstruction: Get degraded image
    
    Args:
        x_hr: High-resolution PET image [B, C, H, W], [B, H, W], or [H, W]
        fov_mm: Field of View in mm (default: 256)
        fwhm_mm: PSF FWHM in mm (default: 4.0)
        dose_alpha: Dose scaling (1.0 = full, 0.1 = 10%)
        rebin_factor: Angular rebinning factor (default: 1 = no rebinning)
        rebin_radial_factor: Radial rebinning factor (default: 1)
        background_beta: Background fraction (default: 0 = no background)
        num_angles: Number of projection angles (default: 180)
        scale_counts: Counts scaling for Poisson (default: 1e6)
        return_sinogram: Whether to return sinogram (default: True)
        device: Target device (auto-detect if None)
    
    Returns:
        y_obs: Observed sinogram (if return_sinogram=True, else None)
        x_lr: Degraded image (same size as input)
        meta: PhysicalDegradationMeta with all parameters
    
    Example:
        >>> y_obs, x_lr, meta = simulate_physical_degradation(
        ...     x_hr, fov_mm=256, fwhm_mm=4.0, dose_alpha=0.1,
        ...     rebin_factor=2, background_beta=0.05
        ... )
        >>> # meta contains all parameters for DC forward model
    
    Note:
        The returned meta should be passed to PhysicalPETForwardModel
        to ensure DC gradient uses exactly the same forward operator.
    """
    if not PHYSICAL_SIMULATION_AVAILABLE:
        raise ImportError(
            "Physical simulation module not available. "
            "Check that utils/physical_pet_simulation.py exists."
        )
    
    return degrade_pet(
        x_hr=x_hr,
        fov_mm=fov_mm,
        fwhm_mm=fwhm_mm,
        dose_alpha=dose_alpha,
        rebin_factor=rebin_factor,
        rebin_radial_factor=rebin_radial_factor,
        background_beta=background_beta,
        num_angles=num_angles,
        scale_counts=scale_counts,
        return_sinogram=return_sinogram,
        device=device
    )


def create_physical_pet_simulator(
    fov_mm: float = DEFAULT_FOV_MM,
    hr_size: int = DEFAULT_HR_SIZE,
    fwhm_mm: float = DEFAULT_FWHM_MM,
    num_angles: int = DEFAULT_NUM_ANGLES,
    scale_counts: float = DEFAULT_SCALE_COUNTS,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> 'PhysicalPETSimulator':
    """
    Create a PhysicalPETSimulator for repeated use.
    
    For single degradations, use simulate_physical_degradation() which
    creates a simulator internally. For repeated degradations with the
    same geometry, create a simulator once with this function.
    
    Args:
        fov_mm: Field of View in mm
        hr_size: High-resolution grid size
        fwhm_mm: PSF FWHM in mm
        num_angles: Number of projection angles
        scale_counts: Counts scaling for Poisson
        device: Target device
    
    Returns:
        PhysicalPETSimulator instance
    
    Example:
        >>> simulator = create_physical_pet_simulator(fov_mm=256, fwhm_mm=4.0)
        >>> for batch in dataloader:
        ...     y_obs, x_lr, meta = simulator.degrade_pet(batch, dose_alpha=0.1)
    """
    if not PHYSICAL_SIMULATION_AVAILABLE:
        raise ImportError(
            "Physical simulation module not available. "
            "Check that utils/physical_pet_simulation.py exists."
        )
    
    return create_physical_simulator(
        fov_mm=fov_mm,
        hr_size=hr_size,
        fwhm_mm=fwhm_mm,
        num_angles=num_angles,
        scale_counts=scale_counts,
        device=device
    )


# =============================================================================
# Test code
# =============================================================================

if __name__ == '__main__':
    print("Testing PETSimulator...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create simulator
    simulator = create_pet_simulator(
        image_size=128,
        num_angles=180,
        dose_factor=0.1,
        use_psf=True,
        device=device
    )
    
    # Create test phantom
    size = 128
    phantom = torch.zeros(size, size, device=device)
    y, x = torch.meshgrid(
        torch.arange(size, device=device, dtype=torch.float32),
        torch.arange(size, device=device, dtype=torch.float32),
        indexing='ij'
    )
    center = size // 2
    mask = ((x - center)**2 + (y - center)**2) < (size//3)**2
    phantom[mask] = 1.0
    
    print(f"Phantom shape: {phantom.shape}")
    
    # Test forward projection
    print("\nTesting forward_project...")
    sinogram, aux = simulator.forward_project(phantom)
    print(f"  Sinogram shape: {sinogram.shape}")
    print(f"  Aux: {aux}")
    
    # Test back projection
    print("\nTesting back_project...")
    recon = simulator.back_project(sinogram)
    print(f"  Reconstruction shape: {recon.shape}")
    
    # Test FBP
    print("\nTesting fbp...")
    recon_fbp = simulator.fbp(sinogram)
    print(f"  FBP shape: {recon_fbp.shape}")
    
    # Test measurement simulation
    print("\nTesting simulate_measurement...")
    sino_noisy, aux = simulator.simulate_measurement(phantom, {'add_noise': True, 'return_clean': True})
    print(f"  Noisy sinogram shape: {sino_noisy.shape}")
    print(f"  Noise std: {aux['noise_std']:.4f}")
    
    # Test degrade_image
    print("\nTesting degrade_image...")
    degraded = simulator.degrade_image(phantom, {'lowres': True, 'lowres_factor': 4, 'noise': True})
    print(f"  Degraded shape: {degraded.shape}")
    
    print("\n✅ All tests passed!")
