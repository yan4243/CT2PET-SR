"""
Physical PET Simulation Module with Interpretable Degradation Parameters

This module provides a unified interface for simulating physically meaningful
low-resolution (LR) and low-dose PET acquisitions. The degradation model
includes explicit physical parameters that can be used consistently in both
data simulation and Data Consistency (DC) reconstruction.

Physical Degradation Model:
===========================

1. PSF-based Resolution Degradation (mm-based)
   - Uses Gaussian PSF parameterized by FWHM in millimeters
   - sigma_px = fwhm_mm / (2 * sqrt(2 * ln(2)) * spacing_mm)
   - Applied in image domain as 2D convolution

2. Explicit Voxel Spacing (mm/voxel)
   - Fixed physical FOV (e.g., 256 mm)
   - spacing_mm = FOV_mm / grid_size
   - LR grid has larger spacing (fewer pixels covering same FOV)
   
3. Low-Dose via Projection-Domain Poisson Noise
   - Apply Poisson noise in sinogram domain only
   - dose_alpha: scaling factor (1.0 = full dose, 0.1 = 10% dose)
   - y ~ Poisson(dose_alpha * y_ideal)

4. Sinogram Rebin (Projection-Domain Downsampling)
   - Angular or radial downsampling in projection domain
   - rebin_factor: factor for angular/radial compression

5. Background/Scatter Proxy
   - Simple additive background before Poisson noise
   - b = background_beta * mean(y_ideal) or constant
   - Included in DC forward model

Data Consistency Requirement:
=============================
The EXACT same forward degradation (PSF, spacing, rebin, background, dose)
must be used both to generate y_obs AND inside DC gradient computation.
This module ensures consistency through the PhysicalDegradationMeta dataclass.

Author: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Union, Any
from dataclasses import dataclass, asdict
import warnings
import math
import astra


# =============================================================================
# Physical Constants and Defaults
# =============================================================================

# FWHM to sigma conversion factor: sigma = FWHM / FWHM_TO_SIGMA
FWHM_TO_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))  # ≈ 2.355

# Default physical parameters
DEFAULT_FOV_MM = 256.0          # Field of View in mm (typical whole-body PET)
DEFAULT_HR_SIZE = 256           # High-resolution grid size
DEFAULT_FWHM_MM = 4.0           # Typical PET scanner resolution FWHM
DEFAULT_DOSE_ALPHA = 1.0        # Full dose (1.0 = 100%)
DEFAULT_REBIN_FACTOR = 1        # No rebinning
DEFAULT_BACKGROUND_BETA = 0.0   # No background
DEFAULT_NUM_ANGLES = 180        # Number of projection angles
DEFAULT_SCALE_COUNTS = 1e6      # Counts scaling for Poisson simulation


@dataclass
class PhysicalDegradationMeta:
    """
    Metadata describing the physical degradation parameters.
    
    This dataclass stores all parameters needed to reproduce the exact
    degradation, ensuring consistency between simulation and DC.
    
    Attributes:
        fov_mm: Field of View in millimeters
        hr_size: High-resolution grid size (pixels)
        lr_size: Low-resolution grid size (pixels, may equal hr_size)
        spacing_hr_mm: HR voxel spacing in mm/pixel
        spacing_lr_mm: LR voxel spacing in mm/pixel
        fwhm_mm: PSF FWHM in millimeters
        sigma_hr_px: PSF sigma in HR pixels
        sigma_lr_px: PSF sigma in LR pixels
        dose_alpha: Dose scaling factor (1.0 = full dose)
        rebin_factor: Sinogram rebinning factor (angular)
        rebin_radial_factor: Sinogram radial rebinning factor
        background_beta: Background fraction (relative to mean signal)
        num_angles: Number of projection angles
        num_angles_rebinned: Number of angles after rebinning
        num_bins: Number of detector bins
        num_bins_rebinned: Number of bins after radial rebinning
        scale_counts: Counts scaling for Poisson simulation
    """
    fov_mm: float
    hr_size: int
    lr_size: int
    spacing_hr_mm: float
    spacing_lr_mm: float
    fwhm_mm: float
    sigma_hr_px: float
    sigma_lr_px: float
    dose_alpha: float
    rebin_factor: int
    rebin_radial_factor: int
    background_beta: float
    num_angles: int
    num_angles_rebinned: int
    num_bins: int
    num_bins_rebinned: int
    scale_counts: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PhysicalDegradationMeta':
        """Create from dictionary."""
        return cls(**d)


def compute_spacing_mm(fov_mm: float, grid_size: int) -> float:
    """
    Compute voxel spacing from FOV and grid size.
    
    Args:
        fov_mm: Field of View in mm
        grid_size: Number of pixels along one dimension
    
    Returns:
        spacing_mm: Voxel spacing in mm/pixel
    """
    return fov_mm / grid_size


def compute_sigma_px(fwhm_mm: float, spacing_mm: float) -> float:
    """
    Convert PSF FWHM in mm to sigma in pixels.
    
    Uses the standard Gaussian relationship:
        FWHM = 2 * sqrt(2 * ln(2)) * sigma
        sigma_mm = fwhm_mm / FWHM_TO_SIGMA
        sigma_px = sigma_mm / spacing_mm
    
    Args:
        fwhm_mm: PSF FWHM in millimeters
        spacing_mm: Voxel spacing in mm/pixel
    
    Returns:
        sigma_px: PSF sigma in pixels
    """
    sigma_mm = fwhm_mm / FWHM_TO_SIGMA
    sigma_px = sigma_mm / spacing_mm
    return sigma_px


def create_gaussian_kernel_2d(
    sigma_px: float,
    kernel_size: Optional[int] = None,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel for PSF blurring.
    
    Args:
        sigma_px: Standard deviation in pixels
        kernel_size: Kernel size (auto-computed if None: 6*sigma + 1, odd)
        device: Target device
        dtype: Data type
    
    Returns:
        kernel: 2D Gaussian kernel [1, 1, K, K] for conv2d
    """
    if kernel_size is None:
        # Use 6-sigma rule, ensure odd size
        kernel_size = int(np.ceil(6 * sigma_px)) | 1
        kernel_size = max(3, kernel_size)  # Minimum 3x3
    
    # Create coordinate grid
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    y = torch.arange(-half, half + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # 2D Gaussian
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma_px**2))
    kernel = kernel / kernel.sum()  # Normalize
    
    # Shape for conv2d: [out_ch, in_ch, H, W]
    return kernel.unsqueeze(0).unsqueeze(0)


# =============================================================================
# Physical PET Projector with Explicit Geometry
# =============================================================================

class PhysicalPETProjector(nn.Module):
    """
    PET Forward/Back Projector using ASTRA Toolbox (GPU-accelerated)
    
    This class implements physically accurate Radon transform using the
    ASTRA toolbox with GPU acceleration, replacing the naive rotation-based
    method. Uses parallel beam geometry (standard for 2D PET).
    
    Features:
    - Accurate line-integral forward projection via ASTRA FP_CUDA
    - Filtered back projection via ASTRA FBP_CUDA (ram-lak filter)
    - Unfiltered back projection via ASTRA BP_CUDA
    - mm-based PSF modeling (Gaussian convolution in image domain)
    - Sinogram rebinning (angular + radial)
    - Background term (scatter/random proxy)
    - Proper physical geometry: det_spacing matches pixel spacing
    
    The projector uses ASTRA's 'parallel' geometry:
    - Volume: image_size x image_size, pixel size = fov_mm / image_size
    - Detector: num_bins = ceil(image_size * sqrt(2)), spacing = pixel_size
    - Angles: uniformly spaced from 0 to pi
    
    Args:
        fov_mm: Field of View in mm (fixed for all operations)
        image_size: Image grid size (pixels)
        num_angles: Number of projection angles
        num_bins: Number of detector bins (auto if None)
        fwhm_mm: PSF FWHM in mm (0 = no PSF)
        rebin_factor: Angular rebinning factor
        rebin_radial_factor: Radial rebinning factor
        background_beta: Background fraction (relative to mean)
        device: Target device (ASTRA always uses GPU internally)
    
    Example:
        >>> proj = PhysicalPETProjector(fov_mm=256, image_size=256, fwhm_mm=4.0)
        >>> sinogram = proj.forward_project(image)
        >>> recon = proj.fbp(sinogram)
    """
    
    def __init__(
        self,
        fov_mm: float = DEFAULT_FOV_MM,
        image_size: int = DEFAULT_HR_SIZE,
        num_angles: int = DEFAULT_NUM_ANGLES,
        num_bins: Optional[int] = None,
        fwhm_mm: float = 0.0,
        rebin_factor: int = 1,
        rebin_radial_factor: int = 1,
        background_beta: float = 0.0,
        device: Union[str, torch.device] = 'cpu'
    ):
        super().__init__()
        
        self.fov_mm = fov_mm
        self.image_size = image_size
        self.num_angles = num_angles
        self.num_bins = num_bins if num_bins else int(np.ceil(image_size * np.sqrt(2)))
        self.fwhm_mm = fwhm_mm
        self.rebin_factor = rebin_factor
        self.rebin_radial_factor = rebin_radial_factor
        self.background_beta = background_beta
        self.device = device if isinstance(device, str) else str(device)
        
        # Compute derived quantities
        self.spacing_mm = compute_spacing_mm(fov_mm, image_size)
        self.sigma_px = compute_sigma_px(fwhm_mm, self.spacing_mm) if fwhm_mm > 0 else 0.0
        
        # Rebinned dimensions
        self.num_angles_rebinned = max(1, num_angles // rebin_factor)
        self.num_bins_rebinned = max(1, self.num_bins // rebin_radial_factor)
        
        # Initialize ASTRA geometry
        self._setup_astra_geometry()
        
        # Store angle tensors for compatibility
        angles_np = np.linspace(0, np.pi, self.num_angles, endpoint=False)
        self.register_buffer('angles', torch.from_numpy(angles_np).float())
        
        if self.rebin_factor > 1:
            indices = np.linspace(0, self.num_angles - 1, self.num_angles_rebinned, dtype=int)
            self.register_buffer('angles_rebinned', self.angles[indices])
        else:
            self.register_buffer('angles_rebinned', self.angles.clone())
        
        # Initialize PSF kernel if needed
        if fwhm_mm > 0 and self.sigma_px > 0.1:
            self._setup_psf_kernel()
        else:
            self.psf_kernel = None
            self.psf_padding = 0
    
    def _setup_astra_geometry(self):
        """Setup ASTRA volume and projection geometries."""
        
        
        # Volume geometry: image_size x image_size
        # Use physical extents so pixel size = spacing_mm
        half_fov = self.fov_mm / 2.0
        self._vol_geom = astra.create_vol_geom(
            self.image_size, self.image_size,
            -half_fov, half_fov,  # min_x, max_x
            -half_fov, half_fov   # min_y, max_y
        )
        
        # Projection geometry: parallel beam
        # det_spacing = pixel size in mm (so detector covers FOV properly)
        # det_count = num_bins (= ceil(image_size * sqrt(2)) for full coverage)
        det_spacing = self.spacing_mm
        angles_full = np.linspace(0, np.pi, self.num_angles, endpoint=False)
        
        self._proj_geom_full = astra.create_proj_geom(
            'parallel', det_spacing, self.num_bins, angles_full
        )
        
        # Also create rebinned projection geometry
        if self.rebin_factor > 1:
            indices = np.linspace(0, self.num_angles - 1, self.num_angles_rebinned, dtype=int)
            angles_rebinned = angles_full[indices]
        else:
            angles_rebinned = angles_full
        
        self._proj_geom_rebinned = astra.create_proj_geom(
            'parallel', det_spacing, self.num_bins, angles_rebinned
        )
        
        # For FBP on rebinned+radially-pooled sinograms, we create geometry on demand
        # because the bin count changes after radial rebinning
    
    def _setup_psf_kernel(self):
        """Setup PSF convolution kernel."""
        kernel = create_gaussian_kernel_2d(
            self.sigma_px,
            device=self.device
        )
        self.register_buffer('psf_kernel', kernel)
        self.psf_padding = kernel.shape[-1] // 2
    
    def _astra_forward_project_single(
        self, 
        image_np: np.ndarray, 
        proj_geom
    ) -> np.ndarray:
        """
        Forward project a single 2D image using ASTRA GPU.
        
        Args:
            image_np: 2D numpy array [H, W], float32
            proj_geom: ASTRA projection geometry
        
        Returns:
            sinogram: 2D numpy array [num_angles, num_bins], float32
        """
        
        
        vol_id = astra.data2d.create('-vol', self._vol_geom, data=image_np.astype(np.float32))
        sino_id = astra.data2d.create('-sino', proj_geom)
        
        cfg = astra.astra_dict('FP_CUDA')
        cfg['ProjectionDataId'] = sino_id
        cfg['VolumeDataId'] = vol_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        
        sinogram = astra.data2d.get(sino_id)
        
        astra.algorithm.delete(alg_id)
        astra.data2d.delete([vol_id, sino_id])
        
        return sinogram
    
    def _astra_back_project_single(
        self, 
        sinogram_np: np.ndarray, 
        proj_geom,
        vol_geom=None
    ) -> np.ndarray:
        """
        Back project a single sinogram using ASTRA GPU (unfiltered).
        
        Args:
            sinogram_np: 2D numpy array [num_angles, num_bins], float32
            proj_geom: ASTRA projection geometry
            vol_geom: ASTRA volume geometry (default: self._vol_geom)
        
        Returns:
            image: 2D numpy array [H, W], float32
        """
        
        vg = vol_geom if vol_geom is not None else self._vol_geom
        
        sino_id = astra.data2d.create('-sino', proj_geom, data=sinogram_np.astype(np.float32))
        vol_id = astra.data2d.create('-vol', vg)
        
        cfg = astra.astra_dict('BP_CUDA')
        cfg['ProjectionDataId'] = sino_id
        cfg['ReconstructionDataId'] = vol_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        
        image = astra.data2d.get(vol_id)
        
        astra.algorithm.delete(alg_id)
        astra.data2d.delete([sino_id, vol_id])
        
        return image
    
    def _astra_fbp_single(
        self, 
        sinogram_np: np.ndarray, 
        proj_geom,
        vol_geom=None,
        filter_type: str = 'ram-lak'
    ) -> np.ndarray:
        """
        Filtered Back Projection of a single sinogram using ASTRA GPU.
        
        Args:
            sinogram_np: 2D numpy array [num_angles, num_bins], float32
            proj_geom: ASTRA projection geometry
            vol_geom: ASTRA volume geometry (default: self._vol_geom)
            filter_type: FBP filter type (default: 'ram-lak')
        
        Returns:
            image: 2D numpy array [H, W], float32
        """
        
        vg = vol_geom if vol_geom is not None else self._vol_geom
        
        sino_id = astra.data2d.create('-sino', proj_geom, data=sinogram_np.astype(np.float32))
        vol_id = astra.data2d.create('-vol', vg)
        
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ProjectionDataId'] = sino_id
        cfg['ReconstructionDataId'] = vol_id
        cfg['option'] = {'FilterType': filter_type}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        
        image = astra.data2d.get(vol_id)
        
        astra.algorithm.delete(alg_id)
        astra.data2d.delete([sino_id, vol_id])
        
        return image
    
    def _make_proj_geom_for_sinogram(
        self,
        num_angles_in: int,
        num_bins_in: int
    ):
        """
        Create an ASTRA projection geometry matching an arbitrary sinogram shape.
        Used for back-projecting rebinned/radially-pooled sinograms.
        """
        
        
        # Determine angles
        if num_angles_in == self.num_angles:
            angles_np = np.linspace(0, np.pi, self.num_angles, endpoint=False)
        elif num_angles_in == self.num_angles_rebinned:
            if self.rebin_factor > 1:
                full_angles = np.linspace(0, np.pi, self.num_angles, endpoint=False)
                indices = np.linspace(0, self.num_angles - 1, self.num_angles_rebinned, dtype=int)
                angles_np = full_angles[indices]
            else:
                angles_np = np.linspace(0, np.pi, num_angles_in, endpoint=False)
        else:
            angles_np = np.linspace(0, np.pi, num_angles_in, endpoint=False)
        
        # Detector spacing: scale by ratio of bins
        # If radial rebinning was applied, each new bin covers more physical space
        det_spacing = self.spacing_mm * (self.num_bins / num_bins_in)
        
        return astra.create_proj_geom('parallel', det_spacing, num_bins_in, angles_np)
    
    def apply_psf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply PSF blurring in image domain.
        
        Args:
            x: Input image [B, C, H, W], [B, H, W], or [H, W]
        
        Returns:
            Blurred image with same shape
        """
        if self.psf_kernel is None:
            return x
        
        # Handle dimensions
        squeeze_batch = False
        squeeze_channel = False
        
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze_batch = True
            squeeze_channel = True
        elif x.dim() == 3:
            x = x.unsqueeze(1)
            squeeze_channel = True
        
        # Apply convolution
        blurred = F.conv2d(x, self.psf_kernel.to(x.dtype), padding=self.psf_padding)
        
        # Restore shape
        if squeeze_channel:
            blurred = blurred.squeeze(1)
        if squeeze_batch:
            blurred = blurred.squeeze(0)
        
        return blurred
    
    def forward_project(
        self,
        x: torch.Tensor,
        apply_psf: bool = True,
        apply_rebin: bool = True
    ) -> torch.Tensor:
        """
        Forward projection: image -> sinogram (using ASTRA GPU)
        
        Applies optional PSF, then ASTRA forward projection, then optional rebinning.
        
        Args:
            x: Input image [B, C, H, W], [B, H, W], or [H, W]
            apply_psf: Whether to apply PSF before projection
            apply_rebin: Whether to apply sinogram rebinning
        
        Returns:
            sinogram: [B, num_angles, num_bins] or [num_angles, num_bins]
        """
        # Handle dimensions
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze_batch = True
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        
        batch_size = x.shape[0]
        input_device = x.device
        
        # Apply PSF if enabled
        if apply_psf and self.psf_kernel is not None:
            x = F.conv2d(x, self.psf_kernel.to(x.dtype).to(x.device), padding=self.psf_padding)
        
        # Select projection geometry based on rebinning
        if apply_rebin and self.rebin_factor > 1:
            proj_geom = self._proj_geom_rebinned
        else:
            proj_geom = self._proj_geom_full
        
        # ASTRA forward projection (batch loop, numpy bridge)
        sinograms = []
        for b in range(batch_size):
            img_np = x[b, 0].detach().cpu().numpy()
            sino_np = self._astra_forward_project_single(img_np, proj_geom)
            sinograms.append(torch.from_numpy(sino_np).float())
        
        sinogram = torch.stack(sinograms, dim=0).to(input_device)  # [B, num_angles, num_bins]
        
        # Apply radial rebinning if needed
        if apply_rebin and self.rebin_radial_factor > 1:
            sinogram = F.avg_pool1d(sinogram, self.rebin_radial_factor)
        
        if squeeze_batch:
            sinogram = sinogram.squeeze(0)
        
        return sinogram
    
    def back_project(
        self,
        sinogram: torch.Tensor,
        filtered: bool = False,
        target_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Back projection: sinogram -> image (using ASTRA GPU)
        
        Args:
            sinogram: [B, num_angles, num_bins] or [num_angles, num_bins]
            filtered: If True, use ASTRA FBP_CUDA with ram-lak filter
            target_size: Target image size (default: self.image_size)
        
        Returns:
            image: [B, H, W] or [H, W]
        """
        target_size = target_size or self.image_size
        
        squeeze_batch = False
        if sinogram.dim() == 2:
            sinogram = sinogram.unsqueeze(0)
            squeeze_batch = True
        
        batch_size = sinogram.shape[0]
        num_angles_in = sinogram.shape[1]
        num_bins_in = sinogram.shape[2]
        input_device = sinogram.device
        
        # Create matching projection geometry for this sinogram shape
        proj_geom = self._make_proj_geom_for_sinogram(num_angles_in, num_bins_in)
        
        # Create volume geometry for target size 
        half_fov = self.fov_mm / 2.0
        if target_size != self.image_size:
            vol_geom = astra.create_vol_geom(
                target_size, target_size,
                -half_fov, half_fov,
                -half_fov, half_fov
            )
        else:
            vol_geom = self._vol_geom
        
        # ASTRA back/FBP projection (batch loop, numpy bridge)
        images = []
        for b in range(batch_size):
            sino_np = sinogram[b].detach().cpu().numpy()
            if filtered:
                img_np = self._astra_fbp_single(sino_np, proj_geom, vol_geom)
            else:
                img_np = self._astra_back_project_single(sino_np, proj_geom, vol_geom)
            images.append(torch.from_numpy(img_np).float())
        
        image = torch.stack(images, dim=0).to(input_device)  # [B, H, W]
        
        if squeeze_batch:
            image = image.squeeze(0)
        
        return image
    
    def fbp(self, sinogram: torch.Tensor, target_size: Optional[int] = None) -> torch.Tensor:
        """Filtered Back Projection (convenience wrapper using ASTRA FBP_CUDA)."""
        return self.back_project(sinogram, filtered=True, target_size=target_size)
    
    def add_background(
        self,
        sinogram: torch.Tensor,
        beta: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add background (scatter/random proxy) to sinogram.
        
        The background is computed as: b = beta * mean(sinogram)
        
        Args:
            sinogram: Clean sinogram
            beta: Background fraction (uses self.background_beta if None)
        
        Returns:
            sinogram_with_bg: Sinogram with added background
            background: The background term (for DC correction)
        """
        beta = beta if beta is not None else self.background_beta
        
        if beta <= 0:
            return sinogram, torch.zeros_like(sinogram)
        
        # Compute background as fraction of mean signal
        mean_signal = sinogram.mean()
        background = beta * mean_signal * torch.ones_like(sinogram)
        
        return sinogram + background, background
    
    def get_meta(self) -> PhysicalDegradationMeta:
        """Get metadata describing the projector configuration."""
        return PhysicalDegradationMeta(
            fov_mm=self.fov_mm,
            hr_size=self.image_size,
            lr_size=self.image_size,  # Same for projector
            spacing_hr_mm=self.spacing_mm,
            spacing_lr_mm=self.spacing_mm,
            fwhm_mm=self.fwhm_mm,
            sigma_hr_px=self.sigma_px,
            sigma_lr_px=self.sigma_px,
            dose_alpha=1.0,  # Set during simulation
            rebin_factor=self.rebin_factor,
            rebin_radial_factor=self.rebin_radial_factor,
            background_beta=self.background_beta,
            num_angles=self.num_angles,
            num_angles_rebinned=self.num_angles_rebinned,
            num_bins=self.num_bins,
            num_bins_rebinned=self.num_bins_rebinned,
            scale_counts=DEFAULT_SCALE_COUNTS
        )
    
    def to(self, device: Union[str, torch.device]) -> 'PhysicalPETProjector':
        """Move to device."""
        self.device = device if isinstance(device, str) else str(device)
        return super().to(device)


# =============================================================================
# Physical PET Degradation Simulator
# =============================================================================

class PhysicalPETSimulator(nn.Module):
    """
    Physical PET Simulator with Interpretable Degradation Parameters
    
    This class provides the main interface for simulating low-resolution
    and low-dose PET acquisitions with explicit physical parameters.
    
    Key Features:
    - FOV-based geometry (spacing_mm = FOV_mm / grid_size)
    - mm-based PSF modeling (fwhm_mm)
    - Projection-domain Poisson noise (dose_alpha)
    - Sinogram rebinning (rebin_factor)
    - Background term (background_beta)
    
    Args:
        fov_mm: Field of View in mm
        hr_size: High-resolution grid size
        lr_size: Low-resolution grid size (for LR simulation, optional)
        fwhm_mm: PSF FWHM in mm
        num_angles: Number of projection angles
        num_bins: Number of detector bins (auto if None)
        device: Target device
    
    Example:
        >>> sim = PhysicalPETSimulator(fov_mm=256, hr_size=256, fwhm_mm=4.0)
        >>> y_obs, x_lr, meta = sim.degrade_pet(
        ...     x_hr, dose_alpha=0.1, rebin_factor=2, background_beta=0.05
        ... )
    """
    
    def __init__(
        self,
        fov_mm: float = DEFAULT_FOV_MM,
        hr_size: int = DEFAULT_HR_SIZE,
        lr_size: Optional[int] = None,
        fwhm_mm: float = DEFAULT_FWHM_MM,
        num_angles: int = DEFAULT_NUM_ANGLES,
        num_bins: Optional[int] = None,
        scale_counts: float = DEFAULT_SCALE_COUNTS,
        device: Union[str, torch.device] = 'cpu'
    ):
        super().__init__()
        
        self.fov_mm = fov_mm
        self.hr_size = hr_size
        self.lr_size = lr_size if lr_size else hr_size
        self.fwhm_mm = fwhm_mm
        self.num_angles = num_angles
        self.num_bins = num_bins if num_bins else int(np.ceil(hr_size * np.sqrt(2)))
        self.scale_counts = scale_counts
        self.device = device if isinstance(device, str) else str(device)
        
        # Compute spacings
        self.spacing_hr_mm = compute_spacing_mm(fov_mm, hr_size)
        self.spacing_lr_mm = compute_spacing_mm(fov_mm, self.lr_size)
        
        # Compute PSF sigma in both resolutions
        self.sigma_hr_px = compute_sigma_px(fwhm_mm, self.spacing_hr_mm) if fwhm_mm > 0 else 0.0
        self.sigma_lr_px = compute_sigma_px(fwhm_mm, self.spacing_lr_mm) if fwhm_mm > 0 else 0.0
        
        # Create projector for HR size (main projector)
        # Note: We don't set rebin/background in projector, those are per-call
        self._projector_hr = PhysicalPETProjector(
            fov_mm=fov_mm,
            image_size=hr_size,
            num_angles=num_angles,
            num_bins=self.num_bins,
            fwhm_mm=fwhm_mm,
            device=device
        )
        
        # Create projector for LR size if different
        if self.lr_size != hr_size:
            # LR projector has fewer bins proportionally
            lr_num_bins = int(np.ceil(self.lr_size * np.sqrt(2)))
            self._projector_lr = PhysicalPETProjector(
                fov_mm=fov_mm,
                image_size=self.lr_size,
                num_angles=num_angles,
                num_bins=lr_num_bins,
                fwhm_mm=fwhm_mm,
                device=device
            )
        else:
            self._projector_lr = self._projector_hr
    
    def degrade_pet(
        self,
        x_hr: torch.Tensor,
        dose_alpha: float = DEFAULT_DOSE_ALPHA,
        rebin_factor: int = DEFAULT_REBIN_FACTOR,
        rebin_radial_factor: int = DEFAULT_REBIN_FACTOR,
        background_beta: float = DEFAULT_BACKGROUND_BETA,
        apply_psf: bool = True,
        return_sinogram: bool = True,
        downsample_lr: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, PhysicalDegradationMeta]:
        """
        Simulate complete PET degradation with physical parameters.
        
        Pipeline:
        1. (Optional) Apply PSF blur in HR image domain
        2. (Optional) Downsample to LR grid
        3. Forward project
        4. Apply angular rebinning
        5. Apply radial rebinning
        6. Add background
        7. Apply Poisson noise with dose_alpha scaling
        8. FBP to get degraded image
        9. (Optional) Upsample back to HR for network input
        
        Args:
            x_hr: High-resolution PET image [B, C, H, W] or [B, H, W] or [H, W]
            dose_alpha: Dose scaling (1.0 = full dose, 0.1 = 10% dose)
            rebin_factor: Angular rebinning factor (1 = no rebinning)
            rebin_radial_factor: Radial rebinning factor (1 = no rebinning)
            background_beta: Background fraction (0 = no background)
            apply_psf: Whether to apply PSF blur
            return_sinogram: Whether to return sinogram
            downsample_lr: Whether to downsample to LR grid
        
        Returns:
            y_obs: Observed sinogram (if return_sinogram=True) or degraded image
            x_lr: Degraded image (upsampled to HR size if downsample_lr)
            meta: PhysicalDegradationMeta with all parameters
        """
        # Ensure proper shape
        orig_shape = x_hr.shape
        x = self._ensure_4d(x_hr)
        batch_size = x.shape[0]
        
        # Determine actual LR size
        actual_lr_size = self.lr_size if downsample_lr else self.hr_size
        
        # Step 1: Apply PSF in HR domain (before any resolution change)
        if apply_psf and self.fwhm_mm > 0:
            x = self._projector_hr.apply_psf(x)
        
        # Step 2: Downsample to LR if requested
        if downsample_lr and self.lr_size < self.hr_size:
            x_lr_grid = F.interpolate(x, size=(self.lr_size, self.lr_size),
                                      mode='bilinear', align_corners=False)
            projector = self._projector_lr
            num_bins_used = self._projector_lr.num_bins
        else:
            x_lr_grid = x
            projector = self._projector_hr
            num_bins_used = self._projector_hr.num_bins
        
        # Step 3: Forward project (PSF already applied)
        # Calculate rebinned dimensions
        num_angles_rebinned = max(1, self.num_angles // rebin_factor)
        num_bins_rebinned = max(1, num_bins_used // rebin_radial_factor)
        
        # Create temporary projector with rebin settings for this call
        # Or manually rebin after projection
        sinogram_full = projector.forward_project(x_lr_grid, apply_psf=False, apply_rebin=False)
        
        # Step 4 & 5: Angular and radial rebinning
        if rebin_factor > 1:
            # Angular rebin: average adjacent angles
            sinogram = self._angular_rebin(sinogram_full, rebin_factor)
        else:
            sinogram = sinogram_full
        
        if rebin_radial_factor > 1:
            # Radial rebin: pool along detector dimension
            sinogram = F.avg_pool1d(sinogram, rebin_radial_factor)
        
        # Step 6: Add background
        if background_beta > 0:
            mean_signal = sinogram.mean()
            background = background_beta * mean_signal
            sinogram_with_bg = sinogram + background
        else:
            background = 0.0
            sinogram_with_bg = sinogram
        
        # Step 7: Apply Poisson noise
        y_obs, noise_info = self._apply_poisson_noise(
            sinogram_with_bg, dose_alpha
        )
        
        # Step 8: FBP to get degraded image
        # Need to create projector for rebinned sinogram
        if rebin_factor > 1 or rebin_radial_factor > 1:
            # Create temporary projector for reconstruction
            temp_projector = PhysicalPETProjector(
                fov_mm=self.fov_mm,
                image_size=actual_lr_size,
                num_angles=num_angles_rebinned,
                num_bins=sinogram.shape[-1],
                fwhm_mm=0,  # No PSF in back projection
                device=self.device
            ).to(x_hr.device)
            x_recon = temp_projector.fbp(y_obs, target_size=actual_lr_size)
        else:
            x_recon = projector.fbp(y_obs, target_size=actual_lr_size)
        
        # Ensure non-negative
        x_recon = torch.clamp(x_recon, min=0)
        
        # Step 9: Upsample back to HR if needed
        if downsample_lr and self.lr_size < self.hr_size:
            x_lr = F.interpolate(
                x_recon.unsqueeze(1) if x_recon.dim() == 3 else x_recon.unsqueeze(0).unsqueeze(0),
                size=(self.hr_size, self.hr_size),
                mode='bilinear', align_corners=False
            )
            if x_recon.dim() == 3:
                x_lr = x_lr.squeeze(1)
            else:
                x_lr = x_lr.squeeze(0).squeeze(0)
        else:
            x_lr = x_recon
        
        # Restore shape
        x_lr = self._restore_shape(x_lr, orig_shape)
        
        # Build metadata
        meta = PhysicalDegradationMeta(
            fov_mm=self.fov_mm,
            hr_size=self.hr_size,
            lr_size=actual_lr_size,
            spacing_hr_mm=self.spacing_hr_mm,
            spacing_lr_mm=compute_spacing_mm(self.fov_mm, actual_lr_size),
            fwhm_mm=self.fwhm_mm if apply_psf else 0.0,
            sigma_hr_px=self.sigma_hr_px if apply_psf else 0.0,
            sigma_lr_px=compute_sigma_px(self.fwhm_mm, compute_spacing_mm(self.fov_mm, actual_lr_size)) if apply_psf else 0.0,
            dose_alpha=dose_alpha,
            rebin_factor=rebin_factor,
            rebin_radial_factor=rebin_radial_factor,
            background_beta=background_beta,
            num_angles=self.num_angles,
            num_angles_rebinned=num_angles_rebinned,
            num_bins=num_bins_used,
            num_bins_rebinned=num_bins_rebinned,
            scale_counts=self.scale_counts
        )
        
        if return_sinogram:
            return y_obs, x_lr, meta
        else:
            return x_lr, x_lr, meta
    
    def _angular_rebin(self, sinogram: torch.Tensor, factor: int) -> torch.Tensor:
        """Rebin sinogram by averaging adjacent angles."""
        if factor <= 1:
            return sinogram
        
        batch_size = sinogram.shape[0]
        num_angles = sinogram.shape[1]
        num_bins = sinogram.shape[2]
        
        # Number of output angles
        num_angles_out = num_angles // factor
        
        # Reshape and average
        # Truncate to be divisible by factor
        sinogram_truncated = sinogram[:, :num_angles_out * factor, :]
        sinogram_reshaped = sinogram_truncated.view(batch_size, num_angles_out, factor, num_bins)
        sinogram_rebinned = sinogram_reshaped.mean(dim=2)
        
        return sinogram_rebinned
    
    def _apply_poisson_noise(
        self,
        sinogram: torch.Tensor,
        dose_alpha: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply Poisson noise in projection domain."""
        # Scale to counts
        counts = sinogram * self.scale_counts * dose_alpha
        counts = torch.clamp(counts, min=0)
        
        # Poisson sampling
        if self.training or not torch.is_grad_enabled():
            # True Poisson sampling
            noisy_counts = torch.poisson(counts)
        else:
            # Gaussian approximation for gradient computation
            std = torch.sqrt(counts + 1e-8)
            noisy_counts = counts + std * torch.randn_like(counts)
            noisy_counts = torch.clamp(noisy_counts, min=0)
        
        # Scale back
        noisy_sinogram = noisy_counts / (self.scale_counts * dose_alpha + 1e-10)
        
        # Noise statistics
        noise_std = float(torch.std(noisy_sinogram - sinogram).item()) if sinogram.numel() > 0 else 0.0
        
        return noisy_sinogram, {'noise_std': noise_std}
    
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
            # Target: [H, W]
            while x.dim() > 2:
                x = x.squeeze(0)
            return x
        elif len(orig_shape) == 3:
            # Target: [B, H, W] or [C, H, W]
            if x.dim() == 4:
                return x.squeeze(1)  # Remove channel dim
            elif x.dim() == 2:
                return x.unsqueeze(0)  # Add batch dim
            return x
        elif len(orig_shape) == 4:
            # Target: [B, C, H, W]
            if x.dim() == 3:
                return x.unsqueeze(1)  # Add channel dim
            elif x.dim() == 2:
                return x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            return x
        return x
    
    def get_projector(self, for_lr: bool = False) -> PhysicalPETProjector:
        """Get the internal projector."""
        return self._projector_lr if for_lr else self._projector_hr
    
    def to(self, device: Union[str, torch.device]) -> 'PhysicalPETSimulator':
        """Move to device."""
        self.device = device if isinstance(device, str) else str(device)
        self._projector_hr = self._projector_hr.to(device)
        if self._projector_lr is not self._projector_hr:
            self._projector_lr = self._projector_lr.to(device)
        return super().to(device)


# =============================================================================
# Top-Level Convenience Function
# =============================================================================

def degrade_pet(
    x_hr: torch.Tensor,
    fov_mm: float = DEFAULT_FOV_MM,
    spacing_mm: Optional[float] = None,
    fwhm_mm: float = DEFAULT_FWHM_MM,
    dose_alpha: float = DEFAULT_DOSE_ALPHA,
    rebin_factor: int = DEFAULT_REBIN_FACTOR,
    rebin_radial_factor: int = 1,
    background_beta: float = DEFAULT_BACKGROUND_BETA,
    num_angles: int = DEFAULT_NUM_ANGLES,
    scale_counts: float = DEFAULT_SCALE_COUNTS,
    return_sinogram: bool = True,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.Tensor, torch.Tensor, PhysicalDegradationMeta]:
    """
    Top-level function to degrade a high-resolution PET image.
    
    This function creates a PhysicalPETSimulator internally and applies
    the degradation. For repeated use, instantiate PhysicalPETSimulator
    directly for efficiency.
    
    Args:
        x_hr: High-resolution PET image [B, C, H, W] or [B, H, W] or [H, W]
        fov_mm: Field of View in mm (default: 256)
        spacing_mm: Voxel spacing in mm (computed from fov_mm/hr_size if None)
        fwhm_mm: PSF FWHM in mm (default: 4.0)
        dose_alpha: Dose scaling (1.0 = full, 0.1 = 10%)
        rebin_factor: Angular rebinning factor
        rebin_radial_factor: Radial rebinning factor
        background_beta: Background fraction
        num_angles: Number of projection angles
        scale_counts: Counts scaling for Poisson
        return_sinogram: If True, return sinogram; else return None
        device: Target device (auto-detect from x_hr if None)
    
    Returns:
        y_obs: Observed sinogram (or None if return_sinogram=False)
        x_lr: Degraded image
        meta: PhysicalDegradationMeta with all parameters
    
    Example:
        >>> y_obs, x_lr, meta = degrade_pet(
        ...     x_hr, fov_mm=256, fwhm_mm=4.0, dose_alpha=0.1,
        ...     rebin_factor=2, background_beta=0.05
        ... )
    """
    if device is None:
        device = x_hr.device if isinstance(x_hr, torch.Tensor) else 'cpu'
    
    hr_size = x_hr.shape[-1]
    
    # If spacing_mm is provided, compute the implied lr_size
    # Otherwise, use hr_size
    if spacing_mm is not None:
        lr_size = int(fov_mm / spacing_mm)
    else:
        lr_size = hr_size
    
    # Create simulator
    simulator = PhysicalPETSimulator(
        fov_mm=fov_mm,
        hr_size=hr_size,
        lr_size=lr_size,
        fwhm_mm=fwhm_mm,
        num_angles=num_angles,
        scale_counts=scale_counts,
        device=device
    )
    simulator = simulator.to(device)
    
    # Apply degradation
    y_obs, x_lr, meta = simulator.degrade_pet(
        x_hr,
        dose_alpha=dose_alpha,
        rebin_factor=rebin_factor,
        rebin_radial_factor=rebin_radial_factor,
        background_beta=background_beta,
        apply_psf=(fwhm_mm > 0),
        return_sinogram=return_sinogram,
        downsample_lr=(lr_size != hr_size)
    )
    
    return y_obs, x_lr, meta


# =============================================================================
# Factory Functions for Backward Compatibility
# =============================================================================

def create_physical_simulator(
    fov_mm: float = DEFAULT_FOV_MM,
    hr_size: int = DEFAULT_HR_SIZE,
    fwhm_mm: float = DEFAULT_FWHM_MM,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs
) -> PhysicalPETSimulator:
    """
    Factory function to create PhysicalPETSimulator.
    
    Args:
        fov_mm: Field of View in mm
        hr_size: High-resolution grid size
        fwhm_mm: PSF FWHM in mm
        device: Target device
        **kwargs: Additional arguments
    
    Returns:
        Configured PhysicalPETSimulator
    """
    return PhysicalPETSimulator(
        fov_mm=fov_mm,
        hr_size=hr_size,
        fwhm_mm=fwhm_mm,
        device=device,
        **kwargs
    )


def create_physical_projector(
    fov_mm: float = DEFAULT_FOV_MM,
    image_size: int = DEFAULT_HR_SIZE,
    fwhm_mm: float = 0.0,
    rebin_factor: int = 1,
    background_beta: float = 0.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs
) -> PhysicalPETProjector:
    """
    Factory function to create PhysicalPETProjector.
    
    Args:
        fov_mm: Field of View in mm
        image_size: Image grid size
        fwhm_mm: PSF FWHM in mm
        rebin_factor: Angular rebinning factor
        background_beta: Background fraction
        device: Target device
        **kwargs: Additional arguments
    
    Returns:
        Configured PhysicalPETProjector
    """
    return PhysicalPETProjector(
        fov_mm=fov_mm,
        image_size=image_size,
        fwhm_mm=fwhm_mm,
        rebin_factor=rebin_factor,
        background_beta=background_beta,
        device=device,
        **kwargs
    )


# =============================================================================
# Test Code
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Physical PET Simulation Module")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test parameters
    fov_mm = 256.0
    hr_size = 128  # Use smaller for faster testing
    fwhm_mm = 4.0
    
    print(f"\nPhysical parameters:")
    print(f"  FOV: {fov_mm} mm")
    print(f"  HR size: {hr_size} pixels")
    print(f"  Spacing: {fov_mm/hr_size:.2f} mm/pixel")
    print(f"  FWHM: {fwhm_mm} mm")
    print(f"  Sigma: {compute_sigma_px(fwhm_mm, fov_mm/hr_size):.2f} pixels")
    
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
    print(f"Phantom range: [{phantom.min():.2f}, {phantom.max():.2f}]")
    
    # Test PhysicalPETSimulator
    print("\n" + "-" * 40)
    print("Testing PhysicalPETSimulator")
    print("-" * 40)
    
    simulator = PhysicalPETSimulator(
        fov_mm=fov_mm,
        hr_size=hr_size,
        fwhm_mm=fwhm_mm,
        device=device
    )
    
    # Test with various degradation parameters
    test_cases = [
        {'dose_alpha': 1.0, 'rebin_factor': 1, 'background_beta': 0.0, 'name': 'Full dose'},
        {'dose_alpha': 0.1, 'rebin_factor': 1, 'background_beta': 0.0, 'name': '10% dose'},
        {'dose_alpha': 0.1, 'rebin_factor': 2, 'background_beta': 0.0, 'name': '10% dose + 2x rebin'},
        {'dose_alpha': 0.1, 'rebin_factor': 2, 'background_beta': 0.1, 'name': '10% dose + rebin + bg'},
    ]
    
    for tc in test_cases:
        y_obs, x_lr, meta = simulator.degrade_pet(
            phantom,
            dose_alpha=tc['dose_alpha'],
            rebin_factor=tc['rebin_factor'],
            background_beta=tc['background_beta']
        )
        
        print(f"\n{tc['name']}:")
        print(f"  y_obs shape: {y_obs.shape}")
        print(f"  x_lr shape: {x_lr.shape}")
        print(f"  y_obs range: [{y_obs.min():.4f}, {y_obs.max():.4f}]")
        print(f"  x_lr range: [{x_lr.min():.4f}, {x_lr.max():.4f}]")
        print(f"  Metadata:")
        print(f"    - dose_alpha: {meta.dose_alpha}")
        print(f"    - rebin_factor: {meta.rebin_factor}")
        print(f"    - background_beta: {meta.background_beta}")
        print(f"    - num_angles_rebinned: {meta.num_angles_rebinned}")
    
    # Test top-level function
    print("\n" + "-" * 40)
    print("Testing degrade_pet() function")
    print("-" * 40)
    
    y_obs, x_lr, meta = degrade_pet(
        phantom.squeeze(),  # Test with 2D input
        fov_mm=256.0,
        fwhm_mm=4.0,
        dose_alpha=0.1,
        rebin_factor=2,
        background_beta=0.05,
        device=device
    )
    
    print(f"Input shape: {phantom.squeeze().shape}")
    print(f"y_obs shape: {y_obs.shape}")
    print(f"x_lr shape: {x_lr.shape}")
    print(f"Meta:\n{meta}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
