"""
Data Consistency (DC) Sampler Wrapper for Diffusion Models

This module provides the DCSamplerWrapper class that wraps an existing diffusion
sampler to add Data Consistency (DC) gradient steps during the sampling process.

Key Design Principles:
======================
1. Non-invasive: The original sampler code is NOT modified
2. Optional: DC is only applied when explicitly enabled
3. Configurable: Multiple parameters control DC behavior
4. Modular: Works with any sampler that follows the ConditionalSampler interface

Mathematical Background:
========================
In standard diffusion sampling, we iteratively denoise:
    x_{t-1} = denoise_step(x_t, t, condition)

With DC, after each denoise step, we optionally apply a gradient correction:
    x_{t-1}' = x_{t-1} - λ * ∇_x DC_loss(x_{t-1}, y_obs)

where:
    - DC_loss is the data consistency loss (L2 or Poisson NLL)
    - y_obs is the observed measurement (sinogram)
    - λ is the step size (potentially time-dependent)

The DC correction encourages the generated sample to be consistent with
the observed measurements while still following the learned prior.

DC Application Modes:
=====================
1. apply_on='x0_pred' (default, recommended): Apply DC to predicted clean image x0_pred.
   This is the Route 1 design where DC operates only on denoised estimates.
2. apply_on='x': [DEPRECATED] Apply DC to the noisy sample x_t at each step.
   This mode is deprecated; if set, it falls back to x0_pred with a warning.

Step Size Schedules:
====================
1. 'constant': Fixed step size throughout
2. 'linear': Linearly decay from step_size_start to step_size_end
3. 'cosine': Cosine annealing schedule

New in 2026: Physical Forward Model Support
===========================================
This module now supports the PhysicalPETForwardModel which provides
exact consistency between simulation and DC. To use:

1. Simulate degradation with simulate_physical_degradation()
2. Get the meta (PhysicalDegradationMeta) from simulation
3. Create PhysicalPETForwardModel with that meta
4. Pass to DCSamplerWrapper

This ensures the DC gradient uses the EXACT same forward operator
as used in simulation (PSF, rebinning, background, etc.).

Route 1 Refactor (2026):
=========================
Resolution mismatch between HR images and LR measurements is handled
ENTIRELY inside the forward model:
    y_hat = A_LR( H( x_HR ) ) + r
    grad  = H^T A^T ( residual )
The DC wrapper NO LONGER applies any lowres_factor / _apply_lowres hacks.
DC operates on x0_pred (denoised estimate) only, not on noisy x_t.

Author: 2026
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple, Callable, Union
from tqdm import tqdm
import warnings

from .pet_forward_model import PETForwardModel, create_pet_forward_model

# Import physical forward model for the new API
try:
    from .physical_forward_model import (
        PhysicalPETForwardModel,
        create_physical_forward_model,
    )
    from .physical_pet_simulation import (
        PhysicalDegradationMeta,
    )
    PHYSICAL_FORWARD_MODEL_AVAILABLE = True
except ImportError:
    PHYSICAL_FORWARD_MODEL_AVAILABLE = False
    PhysicalPETForwardModel = None
    PhysicalDegradationMeta = None


class DCSamplerWrapper:
    """
    Wrapper that adds Data Consistency (DC) to an existing diffusion sampler.
    
    This wrapper intercepts the sampling process and applies DC gradient steps
    after each diffusion denoise step. The original sampler is NOT modified.
    
    Args:
        base_sampler: The original ConditionalSampler instance
        forward_model: PETForwardModel for DC computation
        y_obs: Observed sinogram [B, num_angles, num_bins] or [num_angles, num_bins]
        dc_config: Configuration dict with DC parameters:
            - enable: bool, whether DC is enabled (default: True)
            - loss_type: 'l2' or 'poisson' (default: 'l2')
            - step_size: float or schedule config (default: 0.1)
            - step_size_schedule: 'constant', 'linear', 'cosine' (default: 'constant')
            - step_size_start: Starting step size for schedules
            - step_size_end: Ending step size for schedules
            - num_inner_steps: Number of DC gradient steps per diffusion step (default: 1)
            - apply_on: 'x0_pred' (default, recommended) or 'x' (deprecated)
            - start_step: Start applying DC after this diffusion step (default: 0)
            - end_step: Stop applying DC after this diffusion step (default: inf)
            - clamp_min: Minimum value after DC (default: 0.0)
            - clamp_max: Maximum value after DC (default: None)
        meta: Optional scanner/geometry metadata
    
    Example:
        >>> # Create base sampler (unchanged)
        >>> base_sampler = ConditionalSampler(diffusion, model, model_type, device)
        >>> 
        >>> # Create forward model
        >>> forward_model = PETForwardModel(image_size=256)
        >>> 
        >>> # Configure DC
        >>> dc_config = {
        ...     'enable': True,
        ...     'loss_type': 'l2',
        ...     'step_size': 0.1,
        ...     'num_inner_steps': 3
        ... }
        >>> 
        >>> # Wrap sampler
        >>> dc_sampler = DCSamplerWrapper(base_sampler, forward_model, y_obs, dc_config)
        >>> 
        >>> # Sample (same interface as base sampler)
        >>> samples = dc_sampler.sample(condition, shape=shape, method='ddim', ddim_steps=50)
    """
    
    def __init__(
        self,
        base_sampler,
        forward_model: PETForwardModel,
        y_obs: Optional[torch.Tensor] = None,
        dc_config: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None
    ):
        self.base_sampler = base_sampler
        self.forward_model = forward_model
        self.y_obs = y_obs
        self.meta = meta or {}
        
        # Parse DC configuration with defaults
        dc_config = dc_config or {}
        self.dc_enabled = dc_config.get('enable', True)
        self.loss_type = dc_config.get('loss_type', 'l2')
        self.num_inner_steps = dc_config.get('num_inner_steps', 1)
        # Route 1 default: DC operates on x0_pred only (not noisy x_t)
        self.apply_on = dc_config.get('apply_on', 'x0_pred')
        self.start_step = dc_config.get('start_step', 0) or 0
        # Handle None for end_step - use infinity as default
        end_step_val = dc_config.get('end_step', float('inf'))
        self.end_step = float('inf') if end_step_val is None else end_step_val
        self.clamp_min = dc_config.get('clamp_min', 0.0)
        self.clamp_max = dc_config.get('clamp_max', None)
        
        # Data range configuration
        # The diffusion model typically works in [-1, 1] normalized space
        # But the sinogram y_obs is in physical space (positive values)
        # We need to transform x to physical space before computing DC
        self.data_range = dc_config.get('data_range', 1.0)  # Max value in physical space
        self.normalize_to_model = dc_config.get('normalize_to_model', True)  # If True, expect model space in [-1, 1]
        
        # PET normalization parameters (arcsinh normalization from PET_CT_Datasets.py)
        # Physical space: MBq/mL (typically 0 ~ 0.05)
        # Model space: arcsinh normalized to [0, 1]
        self.pet_max_mbqml = dc_config.get('pet_max_mbqml', 0.05)
        self.pet_scale_mbqml = dc_config.get('pet_scale_mbqml', 0.01)
        self.use_arcsinh_norm = dc_config.get('use_arcsinh_norm', True)
        
        # [DEPRECATED / Route 1] lowres_factor is no longer used in the DC path.
        # Resolution mismatch is handled entirely inside the forward model.
        # Kept for backward-compat config parsing; value is ignored for DC.
        _lowres_val = dc_config.get('lowres_factor', 1)
        if _lowres_val > 1:
            warnings.warn(
                f"lowres_factor={_lowres_val} is set but IGNORED in Route 1 DC. "
                "Resolution mismatch is handled inside the forward model. "
                "Remove lowres_factor from dc_config."
            )
        self.lowres_factor = 1  # Always 1 — forward model handles LR
        
        # Step size configuration
        self.step_size_schedule = dc_config.get('step_size_schedule', 'constant')
        if self.step_size_schedule == 'constant':
            self.step_size = dc_config.get('step_size', 0.1)
            self.step_size_start = self.step_size
            self.step_size_end = self.step_size
        else:
            self.step_size_start = dc_config.get('step_size_start', 0.1)
            self.step_size_end = dc_config.get('step_size_end', 0.01)
            self.step_size = self.step_size_start
        
        # Storage for saving intermediate results
        self.save_intermediates = dc_config.get('save_intermediates', False)
        self.intermediate_results = []
        
        # Inherit device from base sampler
        self.device = base_sampler.device
        
        # Move forward model to device
        self.forward_model = self.forward_model.to(self.device)
    
    def set_y_obs(self, y_obs: torch.Tensor):
        """Set or update the observed sinogram."""
        self.y_obs = y_obs.to(self.device) if y_obs is not None else None
    
    def set_meta(self, meta: Dict[str, Any]):
        """Set or update the metadata."""
        self.meta = meta
    
    def _get_step_size(self, step_idx: int, total_steps: int) -> float:
        """
        Compute step size for given step index.
        
        Args:
            step_idx: Current step index (0-indexed)
            total_steps: Total number of sampling steps
        
        Returns:
            Step size for this step
        """
        if self.step_size_schedule == 'constant':
            return self.step_size
        
        # Normalized progress [0, 1]
        progress = step_idx / max(total_steps - 1, 1)
        
        if self.step_size_schedule == 'linear':
            # Linear interpolation from start to end
            return self.step_size_start + progress * (self.step_size_end - self.step_size_start)
        
        elif self.step_size_schedule == 'cosine':
            # Cosine annealing: starts slow, speeds up in middle, slows down at end
            cos_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return self.step_size_end + (self.step_size_start - self.step_size_end) * cos_decay
        
        else:
            warnings.warn(f"Unknown step_size_schedule: {self.step_size_schedule}, using constant")
            return self.step_size
    
    def _model_to_physical(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert from model space (arcsinh normalized, [0, 1]) to physical space (MBq/mL).
        
        For DC on x0_pred (predicted clean image), the model output is already in [0, 1].
        For DC on x (noisy intermediate), this may need different handling.
        
        The arcsinh normalization from PET_CT_Datasets.py:
            normalized = arcsinh(physical / SCALE) / arcsinh(MAX / SCALE)
        
        Inverse:
            physical = sinh(normalized * arcsinh(MAX / SCALE)) * SCALE
        
        Args:
            x: Image in model space [0, 1] (or approximately)
        
        Returns:
            Image in physical space (MBq/mL)
        """
        if not self.use_arcsinh_norm:
            # Simple linear scaling: assume x in [0, 1], scale to [0, data_range]
            return torch.clamp(x, 0, 1) * self.data_range
        
        # Clamp to valid range first
        x_clamped = torch.clamp(x, 0, 1)
        
        # arcsinh inverse transform
        norm_factor = np.arcsinh(self.pet_max_mbqml / self.pet_scale_mbqml)
        # x is in [0, 1], we need to apply sinh inverse
        x_physical = torch.sinh(x_clamped * norm_factor) * self.pet_scale_mbqml
        return x_physical
    
    def _physical_to_model(self, x_physical: torch.Tensor) -> torch.Tensor:
        """
        Convert from physical space (MBq/mL) to model space (arcsinh normalized, [0, 1]).
        
        Args:
            x_physical: Image in physical space (MBq/mL)
        
        Returns:
            Image in model space [0, 1]
        """
        if not self.use_arcsinh_norm:
            # Simple linear scaling
            return x_physical / self.data_range
        
        # arcsinh forward transform
        norm_factor = np.arcsinh(self.pet_max_mbqml / self.pet_scale_mbqml)
        x_normalized = torch.arcsinh(x_physical / self.pet_scale_mbqml) / norm_factor
        return x_normalized
    
    def _apply_lowres(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """
        [DEPRECATED - Route 1] Apply low-resolution degradation (downsample then upsample).
        
        WARNING: This method is no longer called in the DC path. Resolution mismatch
        is handled entirely by the forward model (Route 1). Kept for backward
        compatibility with legacy code paths outside DC.
        
        Args:
            x: Input image [B, C, H, W]
            factor: Downsampling factor
        
        Returns:
            Low-resolution image with SAME shape as input (not downsampled)
        """
        if factor <= 1:
            return x
        
        # Ensure 4D
        squeeze_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze_batch = True
        
        orig_size = x.shape[-1]
        lr_size = orig_size // factor
        
        # Downsample
        x_lowres = torch.nn.functional.interpolate(
            x, size=(lr_size, lr_size), mode='bilinear', align_corners=False
        )
        
        # Upsample back to original size (this is the KEY difference!)
        x_up = torch.nn.functional.interpolate(
            x_lowres, size=(orig_size, orig_size), mode='bilinear', align_corners=False
        )
        
        if squeeze_batch:
            x_up = x_up.squeeze(0)
        
        return x_up
    
    def _apply_dc_correction(
        self,
        x: torch.Tensor,
        step_idx: int,
        total_steps: int
    ) -> torch.Tensor:
        """
        Apply DC correction to x0_pred (Route 1).
        
        Route 1 design:
        - The forward model handles ALL resolution mismatch internally:
              y_hat = A_LR( H( x_hr ) ) + r
        - DC gradient propagates through the full adjoint chain:
              grad = H^T A^T (residual)
        - NO image-side lowres hacks in this method.
        - Input x should be x0_pred (denoised estimate), NOT noisy x_t.
        
        Pipeline:
        1. Convert x0_pred from model space to physical space (inverse arcsinh)
        2. Compute DC gradient via forward_model (HR in, HR grad out)
        3. Normalize gradient and update x_physical
        4. Convert back to model space
        
        Args:
            x: Current denoised estimate (x0_pred) in MODEL SPACE [B, C, H, W]
            step_idx: Current step index
            total_steps: Total number of steps
        
        Returns:
            DC-corrected image in MODEL SPACE
        """
        if not self.dc_enabled or self.y_obs is None:
            return x
        
        # Check if we should apply DC at this step
        if step_idx < self.start_step or step_idx > self.end_step:
            return x
        
        # Store original dtype
        orig_dtype = x.dtype
        
        # Get step size for this step
        step_size = self._get_step_size(step_idx, total_steps)
        
        # =====================================================================
        # Step 1: Convert from model space to physical space
        # Model space: arcsinh normalized [0, 1]
        # Physical space: MBq/mL (typically 0 ~ 0.05)
        # Non-negativity projection on x0_pred only (NOT on x_t).
        # =====================================================================
        x_clamped = torch.clamp(x, 0, 1)  # nonneg projection on denoised estimate
        x_physical = self._model_to_physical(x_clamped)
        
        # Apply multiple inner DC steps if configured
        for inner_idx in range(self.num_inner_steps):
            # =====================================================================
            # Step 2: Compute DC gradient in physical space (Route 1)
            # Forward model handles PSF + projection + rebinning + background.
            # Input is the full HR image — no lowres hack.
            # =====================================================================
            if PHYSICAL_FORWARD_MODEL_AVAILABLE and isinstance(self.forward_model, PhysicalPETForwardModel):
                # Route 1 API: gradient through full H^T A^T chain
                gradient = self.forward_model.dc_gradient_wrt_x_hr(
                    x_physical, self.y_obs, loss_type=self.loss_type
                )
            else:
                # Legacy API: pass meta parameter
                gradient = self.forward_model.dc_gradient(
                    x_physical, self.y_obs, self.meta, self.loss_type
                )
            
            # Debug: print gradient and image statistics (only for first few steps)
            if step_idx <= 1 and inner_idx == 0:
                print(f"\n[DC Route1] Step {step_idx}:")
                print(f"  x (model space): min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")
                print(f"  x_physical: min={x_physical.min().item():.6f}, max={x_physical.max().item():.6f}, mean={x_physical.mean().item():.6f}")
                print(f"  y_obs: min={self.y_obs.min().item():.4f}, max={self.y_obs.max().item():.4f}, mean={self.y_obs.mean().item():.4f}")
                
                # Forward project current HR to check y_hat shape / values
                if PHYSICAL_FORWARD_MODEL_AVAILABLE and isinstance(self.forward_model, PhysicalPETForwardModel):
                    y_hat = self.forward_model.forward_project_hr_to_lr_measurement(x_physical)
                    print(f"  y_hat shape: {y_hat.shape}, y_obs shape: {self.y_obs.shape}")
                    assert y_hat.shape == self.y_obs.shape, (
                        f"Shape mismatch: y_hat {y_hat.shape} vs y_obs {self.y_obs.shape}"
                    )
                    print(f"  y_hat: min={y_hat.min().item():.4f}, max={y_hat.max().item():.4f}, mean={y_hat.mean().item():.4f}")
                    print(f"  |y_hat - y_obs|: {(y_hat - self.y_obs).abs().mean().item():.6f}")
                
                print(f"  gradient (raw): min={gradient.min().item():.6f}, max={gradient.max().item():.6f}")
                if not torch.isfinite(gradient).all():
                    warnings.warn("[DC Route1] Gradient contains NaN/Inf!")
            
            # =====================================================================
            # Step 3: Normalize gradient and update in physical space
            # Use norm-based normalization (more robust than max-based)
            # =====================================================================
            grad_norm = torch.norm(gradient) + 1e-10
            image_norm = torch.norm(x_physical) + 0.1  # floor for stability
            scale_factor = image_norm / grad_norm
            gradient_normalized = gradient * scale_factor
            
            if step_idx <= 5 and inner_idx == 0:
                print(f"  grad_norm: {grad_norm.item():.6f}, image_norm: {image_norm.item():.6f}")
                print(f"  scale_factor: {scale_factor.item():.6f}")
                print(f"  gradient_normalized: norm={torch.norm(gradient_normalized).item():.6f}")
                print(f"  step_size: {step_size:.6f}")
                print(f"  max update: {(step_size * gradient_normalized.abs().max()).item():.6f}")
            
            # Update in physical space
            x_physical = x_physical - step_size * gradient_normalized
            
            # Clamp to valid physical range (non-negative for PET)
            x_physical = torch.clamp(x_physical, min=0.0)
            
            if step_idx <= 5 and inner_idx == 0:
                print(f"  x_physical after DC: min={x_physical.min().item():.6f}, max={x_physical.max().item():.6f}")
        
        # =====================================================================
        # Step 4: Convert back to model space
        # =====================================================================
        x_model = self._physical_to_model(x_physical)
        
        # Clamp to valid model range [0, 1]
        x_model = torch.clamp(x_model, 0.0, 1.0)
        
        if step_idx <= 5:
            print(f"  x (output model space): min={x_model.min().item():.4f}, max={x_model.max().item():.4f}, mean={x_model.mean().item():.4f}")
        
        # Restore original dtype
        if x_model.dtype != orig_dtype:
            x_model = x_model.to(orig_dtype)
        
        return x_model
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        shape: Optional[Tuple] = None,
        method: str = 'ddim',
        ddim_steps: int = 50,
        eta: float = 0.0,
        lr_guidance: Optional[torch.Tensor] = None,
        lr_blend: float = 0.0,
        ode_solver: str = 'heun',
        ode_steps: int = 50,
        ode_t_start: Optional[float] = None,
        ode_t_end: Optional[float] = None,
        ode_eps: float = 1e-3,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Generate samples with DC correction.
        
        This method mirrors the base sampler's sample() interface but adds
        DC correction steps during sampling.
        
        Args:
            condition: Condition tensor [B, C_cond, H, W]
            shape: Output shape (B, C, H, W)
            method: Sampling method ('ddim', 'ddpm', 'ode')
            ddim_steps: Number of DDIM steps
            eta: DDIM eta parameter
            lr_guidance: LR PET guidance tensor
            lr_blend: LR guidance blend weight
            ode_solver: ODE solver type
            ode_steps: Number of ODE steps
            ode_t_start: ODE start time
            ode_t_end: ODE end time
            ode_eps: ODE numerical epsilon
            verbose: Print DC progress info
        
        Returns:
            Generated samples [B, C, H, W]
        """
        # If DC is disabled, just use base sampler
        if not self.dc_enabled or self.y_obs is None:
            if verbose:
                print("[DC Sampler] DC disabled, using base sampler directly")
            return self.base_sampler.sample(
                condition=condition, shape=shape, method=method,
                ddim_steps=ddim_steps, eta=eta,
                lr_guidance=lr_guidance, lr_blend=lr_blend,
                ode_solver=ode_solver, ode_steps=ode_steps,
                ode_t_start=ode_t_start, ode_t_end=ode_t_end, ode_eps=ode_eps
            )
        
        # Route to appropriate DC sampling method
        if method == 'ddim':
            return self._sample_ddim_with_dc(
                condition, shape, ddim_steps, eta, lr_guidance, lr_blend, verbose
            )
        elif method == 'ddpm':
            return self._sample_ddpm_with_dc(
                condition, shape, lr_guidance, lr_blend, verbose
            )
        elif method == 'ode':
            return self._sample_ode_with_dc(
                condition, shape, ode_solver, ode_steps,
                ode_t_start, ode_t_end, ode_eps, lr_guidance, lr_blend, verbose
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def _sample_ddim_with_dc(
        self,
        condition: torch.Tensor,
        shape: Optional[Tuple],
        ddim_steps: int,
        eta: float,
        lr_guidance: Optional[torch.Tensor],
        lr_blend: float,
        verbose: bool
    ) -> torch.Tensor:
        """DDIM sampling with DC correction."""
        # Setup
        batch_size = condition.shape[0]
        H, W = condition.shape[2], condition.shape[3]
        if shape is None:
            shape = (batch_size, 1, H, W)
        
        condition = condition.float()
        if lr_guidance is not None:
            lr_guidance = lr_guidance.float()
        
        # Get diffusion parameters
        timesteps = self.base_sampler.timesteps
        c = timesteps // ddim_steps
        timesteps_seq = list(range(0, timesteps, c))
        alphas_cumprod = self.base_sampler.diffusion.alphas_cumprod.float()
        
        # Start from noise
        noise = torch.randn(shape, device=self.device, dtype=torch.float32)
        if lr_guidance is not None and lr_blend > 0:
            img = (1 - lr_blend) * noise + lr_blend * lr_guidance
        else:
            img = noise
        
        # Clear intermediate storage
        if self.save_intermediates:
            self.intermediate_results = []
        
        # DDIM sampling loop with DC
        total_steps = len(timesteps_seq)
        
        desc = "DDIM+DC Sampling" if verbose else "DDIM+DC"
        for step_idx, i in enumerate(tqdm(reversed(range(total_steps)), desc=desc, total=total_steps, leave=False)):
            t = timesteps_seq[i]
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            
            # Predict noise using base sampler's model forward
            pred_noise = self.base_sampler._model_forward(img, t_tensor, condition)
            
            # DDIM update
            alpha_t = alphas_cumprod[t]
            if i > 0:
                t_prev = timesteps_seq[i - 1]
                alpha_t_prev = alphas_cumprod[t_prev]
            else:
                alpha_t_prev = torch.tensor(1.0, device=self.device)
            
            # Predicted x0
            pred_x0 = (img - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, 0, 2)
            
            # Route 1: DC operates on x0_pred (denoised estimate) only.
            # Applying DC on noisy x_t is disabled to ensure mathematical consistency.
            if self.apply_on == 'x0_pred':
                pred_x0 = self._apply_dc_correction(pred_x0, step_idx, total_steps)
            elif self.apply_on == 'x' and step_idx == 0:
                warnings.warn(
                    "[DC Route1] apply_on='x' is deprecated. "
                    "DC should operate on x0_pred only. Falling back to x0_pred."
                )
                pred_x0 = self._apply_dc_correction(pred_x0, step_idx, total_steps)
            
            # LR guidance on x0
            if lr_guidance is not None and lr_blend > 0:
                guidance_weight = lr_blend * (t / timesteps)
                pred_x0 = (1 - guidance_weight) * pred_x0 + guidance_weight * lr_guidance
            
            # DDIM direction
            dir_xt = torch.sqrt(1 - alpha_t_prev - eta**2 * (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) * pred_noise
            
            # Noise (only if eta > 0)
            if eta > 0 and i > 0:
                noise = torch.randn_like(img)
                sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
            else:
                noise = 0
                sigma = 0
            
            # Update x_{t-1} using corrected x0_pred (no DC on x_t)
            img = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma * noise
            
            # Save intermediate if requested
            if self.save_intermediates:
                self.intermediate_results.append({
                    'step': step_idx,
                    't': t,
                    'x': img.clone(),
                    'x0_pred': pred_x0.clone()
                })
        
        return img
    
    def _sample_ddpm_with_dc(
        self,
        condition: torch.Tensor,
        shape: Optional[Tuple],
        lr_guidance: Optional[torch.Tensor],
        lr_blend: float,
        verbose: bool
    ) -> torch.Tensor:
        """DDPM sampling with DC correction."""
        # Setup
        batch_size = condition.shape[0]
        H, W = condition.shape[2], condition.shape[3]
        if shape is None:
            shape = (batch_size, 1, H, W)
        
        condition = condition.float()
        if lr_guidance is not None:
            lr_guidance = lr_guidance.float()
        
        # Get diffusion parameters
        timesteps = self.base_sampler.timesteps
        betas = self.base_sampler.diffusion.betas.float()
        alphas = 1. - betas
        alphas_cumprod = self.base_sampler.diffusion.alphas_cumprod.float()
        
        # Start from noise
        noise = torch.randn(shape, device=self.device, dtype=torch.float32)
        if lr_guidance is not None and lr_blend > 0:
            img = (1 - lr_blend) * noise + lr_blend * lr_guidance
        else:
            img = noise
        
        # DDPM sampling loop with DC
        desc = "DDPM+DC Sampling" if verbose else "DDPM+DC"
        for step_idx, t in enumerate(tqdm(reversed(range(timesteps)), desc=desc, total=timesteps, leave=False)):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            pred_noise = self.base_sampler._model_forward(img, t_tensor, condition)
            
            # Coefficients
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            beta_t = betas[t]
            
            # Mean
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1 - alpha_cumprod_t)
            mean = coef1 * (img - coef2 * pred_noise)
            
            # LR guidance
            if lr_guidance is not None and lr_blend > 0:
                guidance_weight = lr_blend * (t / timesteps)
                mean = (1 - guidance_weight) * mean + guidance_weight * lr_guidance
            
            # Variance
            if t > 0:
                noise = torch.randn_like(img)
                sigma = torch.sqrt(beta_t)
                img = mean + sigma * noise
            else:
                img = mean
            
            # Route 1: DC on denoised mean (x0_pred proxy), not on noisy x_t.
            # DDPM does not expose x0_pred directly; we approximate it from mean.
            # For strict Route 1, prefer DDIM/ODE over DDPM.
            if self.apply_on in ('x', 'x0_pred'):
                img = self._apply_dc_correction(img, step_idx, timesteps)
        
        return img
    
    def _sample_ode_with_dc(
        self,
        condition: torch.Tensor,
        shape: Optional[Tuple],
        ode_solver: str,
        ode_steps: int,
        ode_t_start: Optional[float],
        ode_t_end: Optional[float],
        ode_eps: float,
        lr_guidance: Optional[torch.Tensor],
        lr_blend: float,
        verbose: bool
    ) -> torch.Tensor:
        """ODE sampling with DC correction."""
        # Setup
        batch_size = condition.shape[0]
        H, W = condition.shape[2], condition.shape[3]
        if shape is None:
            shape = (batch_size, 1, H, W)
        
        condition = condition.float()
        if lr_guidance is not None:
            lr_guidance = lr_guidance.float()
        
        # Get diffusion parameters
        timesteps = self.base_sampler.timesteps
        alphas_cumprod = self.base_sampler.diffusion.alphas_cumprod.float()
        
        # Time grid
        t_start = int(ode_t_start if ode_t_start is not None else (timesteps - 1))
        t_end = int(ode_t_end if ode_t_end is not None else 0)
        t_end = max(t_end, int(ode_eps * timesteps))
        t_grid = torch.linspace(t_start, t_end, ode_steps + 1, device=self.device)
        t_grid = t_grid.round().long()
        
        # Start from noise
        img = torch.randn(shape, device=self.device, dtype=torch.float32)
        
        if verbose:
            print(f"[DC ODE] Solver: {ode_solver}, Steps: {ode_steps}")
        
        # ODE integration with DC
        desc = f"ODE+DC ({ode_solver})" if verbose else f"ODE+DC"
        for step_idx in tqdm(range(ode_steps), desc=desc, leave=False):
            t_cur = t_grid[step_idx].item()
            t_next = t_grid[step_idx + 1].item()
            
            if t_cur == t_next:
                continue
            
            # ODE step (using base sampler's methods)
            if ode_solver == 'euler':
                img = self.base_sampler._ode_euler_step_ddim(img, t_cur, t_next, condition, alphas_cumprod)
            elif ode_solver == 'heun':
                img = self.base_sampler._ode_heun_step_ddim(img, t_cur, t_next, condition, alphas_cumprod)
            elif ode_solver == 'rk4':
                img = self.base_sampler._ode_rk4_step_ddim(img, t_cur, t_next, condition, alphas_cumprod)
            else:
                raise ValueError(f"Unknown ODE solver: {ode_solver}")
            
            # Route 1: do NOT apply DC on noisy x_t; handled on x0_pred below
            
            # LR guidance
            if lr_guidance is not None and lr_blend > 0 and t_cur > t_end:
                t_tensor = torch.full((shape[0],), t_next, device=self.device, dtype=torch.long)
                eps_pred = self.base_sampler._model_forward(img, t_tensor, condition)
                alpha_bar_t = alphas_cumprod[t_next]
                pred_x0 = (img - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
                pred_x0 = torch.clamp(pred_x0, 0, 2)
                
                # Apply DC to x0_pred if configured
                if self.apply_on == 'x0_pred':
                    pred_x0 = self._apply_dc_correction(pred_x0, step_idx, ode_steps)
                
                guidance_weight = lr_blend * (t_next / timesteps)
                guided_x0 = (1 - guidance_weight) * pred_x0 + guidance_weight * lr_guidance
                
                if t_next > 0:
                    img = torch.sqrt(alpha_bar_t) * guided_x0 + torch.sqrt(1 - alpha_bar_t) * eps_pred
        
        # Final denoising step
        final_t = t_grid[-1].item()
        if final_t > 0:
            t_tensor = torch.full((shape[0],), final_t, device=self.device, dtype=torch.long)
            eps_pred = self.base_sampler._model_forward(img, t_tensor, condition)
            alpha_bar_t = alphas_cumprod[final_t]
            pred_x0 = (img - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            pred_x0 = torch.clamp(pred_x0, 0, 2)
            
            # Final DC correction
            if self.apply_on == 'x0_pred':
                pred_x0 = self._apply_dc_correction(pred_x0, ode_steps - 1, ode_steps)
            
            if lr_guidance is not None and lr_blend > 0:
                guidance_weight = lr_blend * 0.01
                pred_x0 = (1 - guidance_weight) * pred_x0 + guidance_weight * lr_guidance
            
            img = pred_x0
        
        return img
    
    def get_intermediate_results(self) -> list:
        """Return saved intermediate results (if save_intermediates was True)."""
        return self.intermediate_results


# =============================================================================
# Factory Functions
# =============================================================================

def create_dc_sampler_wrapper(
    base_sampler,
    image_size: int,
    y_obs: Optional[torch.Tensor] = None,
    dc_config: Optional[Dict[str, Any]] = None,
    device: str = 'cuda'
) -> DCSamplerWrapper:
    """
    Factory function to create DCSamplerWrapper with forward model.
    
    This is the legacy interface using PETForwardModel.
    For physical forward model (DC-consistent), use 
    create_dc_sampler_wrapper_physical() instead.
    
    Args:
        base_sampler: The base ConditionalSampler instance
        image_size: Size of images for forward model
        y_obs: Observed sinogram (can be set later)
        dc_config: DC configuration dict
        device: Target device
    
    Returns:
        Configured DCSamplerWrapper instance
    """
    # Create forward model
    forward_model = create_pet_forward_model(
        image_size=image_size,
        num_angles=dc_config.get('num_angles', 180) if dc_config else 180,
        use_psf=dc_config.get('use_psf', False) if dc_config else False,
        device=device
    )
    
    return DCSamplerWrapper(
        base_sampler=base_sampler,
        forward_model=forward_model,
        y_obs=y_obs,
        dc_config=dc_config
    )


def create_dc_sampler_wrapper_physical(
    base_sampler,
    meta,  # PhysicalDegradationMeta from simulation
    y_obs: Optional[torch.Tensor] = None,
    dc_config: Optional[Dict[str, Any]] = None,
    device: str = 'cuda'
) -> DCSamplerWrapper:
    """
    Create DCSamplerWrapper with PhysicalPETForwardModel for exact DC consistency.
    
    RECOMMENDED: Use this factory when you have a PhysicalDegradationMeta
    from simulate_physical_degradation(). This ensures the DC forward model
    uses the EXACT same parameters as were used in simulation.
    
    Args:
        base_sampler: The base ConditionalSampler instance
        meta: PhysicalDegradationMeta from simulation (contains all params)
        y_obs: Observed sinogram from simulation
        dc_config: DC configuration dict (step_size, num_inner_steps, etc.)
        device: Target device
    
    Returns:
        Configured DCSamplerWrapper with PhysicalPETForwardModel
    
    Example:
        >>> from utils.simulation import simulate_physical_degradation
        >>> y_obs, x_lr, meta = simulate_physical_degradation(
        ...     x_hr, fov_mm=256, fwhm_mm=4.0, dose_alpha=0.1,
        ...     rebin_factor=2, background_beta=0.05
        ... )
        >>> dc_sampler = create_dc_sampler_wrapper_physical(
        ...     base_sampler=sampler,
        ...     meta=meta,
        ...     y_obs=y_obs,
        ...     dc_config={'step_size': 0.1, 'loss_type': 'l2'},
        ...     device='cuda'
        ... )
        >>> samples = dc_sampler.sample(condition, method='ddim')
    """
    if not PHYSICAL_FORWARD_MODEL_AVAILABLE:
        raise ImportError(
            "Physical forward model not available. "
            "Check that utils/physical_forward_model.py exists."
        )
    
    # Create physical forward model from meta
    forward_model = create_physical_forward_model(meta=meta, device=device)
    
    return DCSamplerWrapper(
        base_sampler=base_sampler,
        forward_model=forward_model,
        y_obs=y_obs,
        dc_config=dc_config
    )


def create_dc_sampler_wrapper_from_params(
    base_sampler,
    fov_mm: float = 256.0,
    image_size: int = 256,
    fwhm_mm: float = 4.0,
    dose_alpha: float = 1.0,
    rebin_factor: int = 1,
    rebin_radial_factor: int = 1,
    background_beta: float = 0.0,
    num_angles: int = 180,
    y_obs: Optional[torch.Tensor] = None,
    dc_config: Optional[Dict[str, Any]] = None,
    device: str = 'cuda'
) -> DCSamplerWrapper:
    """
    Create DCSamplerWrapper with PhysicalPETForwardModel from explicit parameters.
    
    Use this when you don't have a meta but know the physical parameters.
    WARNING: Parameters must EXACTLY match the simulation used to generate y_obs!
    
    Args:
        base_sampler: The base ConditionalSampler instance
        fov_mm: Field of View in mm
        image_size: Image grid size
        fwhm_mm: PSF FWHM in mm
        dose_alpha: Dose scaling factor
        rebin_factor: Angular rebinning factor
        rebin_radial_factor: Radial rebinning factor
        background_beta: Background fraction
        num_angles: Number of projection angles
        y_obs: Observed sinogram
        dc_config: DC configuration dict
        device: Target device
    
    Returns:
        Configured DCSamplerWrapper with PhysicalPETForwardModel
    """
    if not PHYSICAL_FORWARD_MODEL_AVAILABLE:
        raise ImportError(
            "Physical forward model not available. "
            "Check that utils/physical_forward_model.py exists."
        )
    
    # Create physical forward model from explicit parameters
    forward_model = PhysicalPETForwardModel.from_params(
        fov_mm=fov_mm,
        image_size=image_size,
        fwhm_mm=fwhm_mm,
        dose_alpha=dose_alpha,
        rebin_factor=rebin_factor,
        rebin_radial_factor=rebin_radial_factor,
        background_beta=background_beta,
        num_angles=num_angles,
        device=device
    )
    
    return DCSamplerWrapper(
        base_sampler=base_sampler,
        forward_model=forward_model,
        y_obs=y_obs,
        dc_config=dc_config
    )


def is_physical_dc_available() -> bool:
    """Check if physical DC sampler is available."""
    return PHYSICAL_FORWARD_MODEL_AVAILABLE


# =============================================================================
# Utility Functions for Saving Simulated Data
# =============================================================================

def save_simulation_outputs(
    output_dir: str,
    sample_key: str,
    pet_image: torch.Tensor,
    sinogram: torch.Tensor,
    degraded_pet: Optional[torch.Tensor] = None,
    save_format: str = 'npz'
):
    """
    Save simulated PET data (sinogram and optionally degraded image).
    
    Args:
        output_dir: Output directory
        sample_key: Key for naming files
        pet_image: Generated PET image
        sinogram: Simulated sinogram
        degraded_pet: Optional degraded PET image
        save_format: 'npz' or 'npy'
    """
    import os
    import numpy as np
    
    sim_dir = os.path.join(output_dir, 'simulation')
    os.makedirs(sim_dir, exist_ok=True)
    
    # Convert to numpy
    pet_np = pet_image.cpu().numpy() if isinstance(pet_image, torch.Tensor) else pet_image
    sino_np = sinogram.cpu().numpy() if isinstance(sinogram, torch.Tensor) else sinogram
    
    if save_format == 'npz':
        data = {
            'pet_image': pet_np,
            'sinogram': sino_np
        }
        if degraded_pet is not None:
            deg_np = degraded_pet.cpu().numpy() if isinstance(degraded_pet, torch.Tensor) else degraded_pet
            data['degraded_pet'] = deg_np
        
        np.savez(os.path.join(sim_dir, f'{sample_key}_simulation.npz'), **data)
    else:
        np.save(os.path.join(sim_dir, f'{sample_key}_pet.npy'), pet_np)
        np.save(os.path.join(sim_dir, f'{sample_key}_sinogram.npy'), sino_np)
        if degraded_pet is not None:
            deg_np = degraded_pet.cpu().numpy() if isinstance(degraded_pet, torch.Tensor) else degraded_pet
            np.save(os.path.join(sim_dir, f'{sample_key}_degraded.npy'), deg_np)
