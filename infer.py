"""
CT to PET Conditional Diffusion Sampling Script (2D).

This script generates PET images from CT conditions using trained conditional
diffusion models. Supports both concat and cross-attention conditioning.

Condition Modes:
- ct: Only CT as condition
- ct+lrpet: CT + Low-Resolution PET (downsampled from HR PET)

Model Types:
- concat: Conditions concatenated with noisy input along channel dimension
- crossattn: Conditions passed via cross-attention mechanism

Sampling Methods:
- ddpm: Standard DDPM sampling (stochastic, 1000 steps)
- ddim: DDIM sampling (deterministic when eta=0, faster)
- ode: Probability Flow ODE sampling (deterministic, uses continuous-time interpretation)
        Mathematically equivalent to DDIM with eta=0, but implemented via explicit ODE solvers
        (Euler/Heun/RK4) for better numerical control and educational clarity.

Data Consistency (DC) Enhancement (Optional):
- When enabled via --dc_enable or config, applies physics-based DC correction
- Requires observed sinogram (y_obs) or simulates from GT PET
- Supports L2 and Poisson NLL loss for DC gradient computation
- See USAGE_DC section below for details

Channel Rules:
- concat + ct:       model_input = cat([x_t(1), ct(1)], dim=1) -> 2 channels
- concat + ct+lrpet: model_input = cat([x_t(1), ct(1), lr_up(1)], dim=1) -> 3 channels
- crossattn + ct:       x=x_t(1), context=ct(1) -> 1 channel context
- crossattn + ct+lrpet: x=x_t(1), context=cat([ct(1), lr_up(1)]) -> 2 channel context

Usage:
    # Concat model with CT only (DDIM)
    python sample_ct2pet_conditional_2d.py \\
        --config configs/ct2pet_concat_2d.yaml \\
        --checkpoint checkpoints/ct2pet_concat_2d/checkpoint_best.pth \\
        --model_type concat --cond ct --sampler ddim \\
        --num_samples 8 --out_dir samples/ct2pet_concat_ct

    # Crossattn model with CT + LR PET (DDIM)
    python sample_ct2pet_conditional_2d.py \\
        --config configs/ct2pet_crossattn_2d.yaml \\
        --checkpoint checkpoints/ct2pet_crossattn_2d/checkpoint_best.pth \\
        --model_type crossattn --cond ct+lrpet --lr_factor 4 --sampler ddim \\
        --num_samples 8 --out_dir samples/ct2pet_crossattn_ct_lrpet

    # ODE sampler with Heun solver (default)
    python sample_ct2pet_conditional_2d.py \\
        --config configs/ct2pet_concat_2d.yaml \\
        --checkpoint checkpoints/ct2pet_concat_2d/checkpoint_best.pth \\
        --model_type concat --cond ct+lrpet --sampler ode --ode_solver heun --ode_steps 50 \\
        --out_dir samples/ct2pet_ode

    # ODE sampler with RK4 solver
    python sample_ct2pet_conditional_2d.py \\
        --config configs/ct2pet_crossattn_2d.yaml \\
        --checkpoint checkpoints/ct2pet_crossattn_2d/checkpoint_best.pth \\
        --model_type crossattn --cond ct --sampler ode --ode_solver rk4 --ode_steps 100 \\
        --out_dir samples/ct2pet_ode_rk4

    # Debug mode
    python sample_ct2pet_conditional_2d.py \\
        --config configs/ct2pet_concat_2d.yaml \\
        --checkpoint checkpoints/ct2pet_concat_2d/checkpoint_best.pth \\
        --model_type concat --cond ct+lrpet --sampler ode \\
        --debug --out_dir samples/debug

USAGE_DC (Data Consistency Enhancement):
    # Enable DC with default settings (simulates y_obs from GT)
    python sample_ct2pet_conditional_2d.py \\
        --config configs/ct2pet_crossattn_2d.yaml \\
        --checkpoint checkpoints/ct2pet_crossattn_2d/checkpoint_best.pth \\
        --model_type crossattn --cond ct --sampler ddim \\
        --dc_enable --dc_loss_type l2 --dc_step_size 0.1 \\
        --out_dir samples/ct2pet_with_dc

    # DC with Poisson NLL and custom settings
    python sample_ct2pet_conditional_2d.py \\
        --config configs/ct2pet_crossattn_2d.yaml \\
        --checkpoint checkpoints/ct2pet_crossattn_2d/checkpoint_best.pth \\
        --model_type crossattn --cond ct --sampler ddim \\
        --dc_enable --dc_loss_type poisson --dc_step_size 0.05 --dc_num_inner 3 \\
        --dc_save_simulation \\
        --out_dir samples/ct2pet_dc_poisson
"""

import os
import sys
import argparse
import json
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from models import UNetModel, GaussianDiffusion
from models.diffusion_crossattn import UNetModelCrossAttn
from datasets.PET_CT_Datasets import PairedImageFolders
from datasets.PET_CT_Datasets_case import PairedImageFoldersByCase, get_case_dataloader
from torch.utils.data import DataLoader
from utils import (
    set_seed,
    calculate_psnr,
    calculate_ssim,
    calculate_mse,
    align_ct_to_pet_batch
)


# =============================================================================
# Physical DC Helper Functions
# =============================================================================

# PET normalization constants (from read_data.py)
# The Dataset uses arcsinh normalization for PET images:
#   normalized = arcsinh(physical / SCALE) / arcsinh(MAX / SCALE)
# Physical space: MBq/mL (typically 0 ~ 0.05)
# Model space: [0, 1] (arcsinh normalized)
PET_MAX_MBQML = 0.05
PET_SCALE_MBQML = 0.01
PET_NORM_FACTOR = np.arcsinh(PET_MAX_MBQML / PET_SCALE_MBQML)  # ≈ 2.312


def model_to_physical(x: torch.Tensor) -> torch.Tensor:
    """
    Convert PET from model space (arcsinh normalized [0,1]) to physical space (MBq/mL).
    
    Inverse of arcsinh normalization:
        physical = sinh(normalized * norm_factor) * SCALE
    
    Args:
        x: PET image in model space [0, 1]
    
    Returns:
        PET image in physical space (MBq/mL, typically 0 ~ 0.05)
    """
    x_clamped = torch.clamp(x, 0, 1)
    return torch.sinh(x_clamped * PET_NORM_FACTOR) * PET_SCALE_MBQML


def physical_to_model(x_physical: torch.Tensor) -> torch.Tensor:
    """
    Convert PET from physical space (MBq/mL) to model space (arcsinh normalized [0,1]).
    
    Forward arcsinh normalization:
        normalized = arcsinh(physical / SCALE) / norm_factor
    
    Args:
        x_physical: PET image in physical space (MBq/mL)
    
    Returns:
        PET image in model space [0, 1]
    """
    x_clamped = torch.clamp(x_physical, 0, None)  # PET must be non-negative
    return torch.arcsinh(x_clamped / PET_SCALE_MBQML) / PET_NORM_FACTOR


def extract_fov_from_affine(affine: np.ndarray, image_size: int) -> float:
    """
    Extract Field of View (FOV) in mm from affine matrix.
    
    The affine matrix (4x4) contains voxel spacing information.
    For DICOM/NIfTI format, the spacing is encoded in the first 3 columns:
    - affine[:3, 0] = row direction vector * row spacing
    - affine[:3, 1] = col direction vector * col spacing
    - affine[:3, 2] = slice direction vector * slice spacing
    
    For 2D slices, we extract the in-plane spacing and compute FOV.
    
    Args:
        affine: 4x4 affine transformation matrix (numpy array)
        image_size: Number of pixels in one dimension
    
    Returns:
        fov_mm: Field of View in mm (assuming square images)
    """
    if affine.ndim == 3:
        # Batch affine [B, 4, 4], use first one
        affine = affine[0]
    
    # Extract voxel spacing from affine matrix
    # The spacing is the norm of the direction vectors scaled by spacing
    spacing_row = np.linalg.norm(affine[:3, 0])
    spacing_col = np.linalg.norm(affine[:3, 1])
    
    # Use average of row and col spacing for 2D
    spacing_mm = (spacing_row + spacing_col) / 2.0
    
    # FOV = spacing * image_size
    fov_mm = spacing_mm * image_size
    
    return fov_mm


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_lr_pet(pet, lr_factor=4, mode='area', upsample_mode='bilinear'):
    """
    Create Low-Resolution PET from High-Resolution PET.
    
    Args:
        pet: HR PET tensor [B, 1, H, W]
        lr_factor: Downsampling factor (default: 4)
        mode: Interpolation mode for downsampling ('area' recommended for downsampling)
        upsample_mode: Upsampling mode ('nearest', 'bilinear', 'zero_pad')
            - 'nearest': Nearest neighbor interpolation
            - 'bilinear': Bilinear interpolation
            - 'zero_pad': Zero-padding expansion (each pixel expands to lr_factor x lr_factor 
                          block with value at top-left corner, rest filled with zeros)
    
    Returns:
        lr: Low-resolution PET [B, 1, H//lr_factor, W//lr_factor]
        lr_up: LR PET upsampled back to original size [B, 1, H, W]
    """
    B, C, H, W = pet.shape
    
    # Downsample: HR -> LR
    # Using 'area' mode for downsampling (average pooling effect, good for natural images)
    lr_h, lr_w = H // lr_factor, W // lr_factor
    lr = F.interpolate(pet, size=(lr_h, lr_w), mode=mode)
    
    # Upsample: LR -> original size (to match CT condition spatial dimensions)
    if upsample_mode == 'zero_pad':
        # Zero-padding expansion: each LR pixel becomes a lr_factor x lr_factor block
        # with the value at top-left corner and zeros elsewhere
        lr_up = torch.zeros(B, C, H, W, device=pet.device, dtype=pet.dtype)
        lr_up[:, :, ::lr_factor, ::lr_factor] = lr
    elif upsample_mode == 'nearest':
        lr_up = F.interpolate(lr, size=(H, W), mode='nearest')
    elif upsample_mode == 'bilinear':
        lr_up = F.interpolate(lr, size=(H, W), mode='bilinear', align_corners=False)
    else:
        raise ValueError(f"Unknown upsample_mode: {upsample_mode}. "
                         f"Choose from 'nearest', 'bilinear', 'zero_pad'")
    
    return lr, lr_up


def load_checkpoint_weights(model, checkpoint_path, device):
    """
    Load model weights from checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load weights on
    
    Returns:
        Loaded checkpoint dict (for metadata)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"[Checkpoint] Loading from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get state dict (prefer EMA weights if available)
    if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
        state_dict = checkpoint['ema_state_dict']
        print("[Checkpoint] Using EMA weights")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("[Checkpoint] Using model weights")
    else:
        raise ValueError("No valid state_dict found in checkpoint")
    
    # Remove 'module.' prefix if present (from DDP training)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"[Checkpoint] WARNING: Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"[Checkpoint] WARNING: Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    
    if 'epoch' in checkpoint:
        print(f"[Checkpoint] From epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"[Checkpoint] Val loss: {checkpoint['val_loss']:.4f}")
    
    print("[Checkpoint] Successfully loaded weights")
    return checkpoint


def create_model(config, model_type, cond_mode, device):
    """
    Create model based on type and condition mode.
    
    Args:
        config: Configuration dict
        model_type: 'concat' or 'crossattn'
        cond_mode: 'ct' or 'ct+lrpet'
        device: Device to create model on
    
    Returns:
        model: Created model
        expected_cond_channels: Expected number of condition channels (always 1, CT only)
    
    Note:
        For ct+lrpet mode, LR PET is blended into the sampling process (not concatenated to condition).
        This allows using models trained with CT-only condition.
    """
    unet_config = config['diffusion']['unet']
    
    # CT is always the only condition channel (LR PET is blended into input)
    expected_cond_channels = 1
    
    if model_type == 'concat':
        # For concat: in_channels = 1 (x_t) + 1 (ct) = 2
        config_in_channels = unet_config['in_channels']
        expected_in_channels = 2
        
        # Check if model config matches
        if config_in_channels != expected_in_channels:
            print(f"\n[WARNING] Config in_channels={config_in_channels}, expected {expected_in_channels}")
            print(f"  Using config value. May cause issues if checkpoint doesn't match.\n")
        
        model = UNetModel(
            in_channels=config_in_channels,  # Use config value
            model_channels=unet_config['model_channels'],
            out_channels=unet_config['out_channels'],
            num_res_blocks=unet_config['num_res_blocks'],
            attention_resolutions=tuple(unet_config['attention_resolutions']),
            dropout=unet_config['dropout'],
            channel_mult=tuple(unet_config['channel_mult']),
            num_heads=unet_config['num_heads']
        ).to(device)
        
    elif model_type == 'crossattn':
        # For crossattn: in_channels = 1 (always), context = 1 (CT only)
        model = UNetModelCrossAttn(
            in_channels=unet_config['in_channels'],  # Should be 1
            model_channels=unet_config['model_channels'],
            out_channels=unet_config['out_channels'],
            context_channels=unet_config.get('context_channels', 64),
            num_res_blocks=unet_config['num_res_blocks'],
            attention_resolutions=tuple(unet_config['attention_resolutions']),
            dropout=unet_config['dropout'],
            channel_mult=tuple(unet_config['channel_mult']),
            num_heads=unet_config['num_heads']
        ).to(device)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model, expected_cond_channels


class ConditionalSampler:
    """
    Conditional sampler supporting both concat and crossattn models.
    """
    
    def __init__(self, diffusion, model, model_type, device):
        self.diffusion = diffusion
        self.model = model
        self.model_type = model_type
        self.device = device
        self.timesteps = diffusion.timesteps
    
    @torch.no_grad()
    def sample(self, condition, shape=None, method='ddim', ddim_steps=50, eta=0.0,
               lr_guidance=None, lr_blend=0.0,
               ode_solver='heun', ode_steps=50, ode_t_start=None, ode_t_end=None, ode_eps=1e-3):
        """
        Generate samples conditioned on given condition tensor.
        
        Args:
            condition: Condition tensor (CT in PET space) [B, 1, H, W]
            shape: Output shape (B, 1, H, W). If None, inferred from condition
            method: 'ddim', 'ddpm', or 'ode'
            ddim_steps: Number of DDIM steps
            eta: DDIM eta parameter (0 = deterministic)
            lr_guidance: LR PET upsampled to original size [B, 1, H, W] for guidance
            lr_blend: Blend weight for LR guidance (0-1). x_init = (1-w)*noise + w*lr_up
            ode_solver: ODE solver type ('euler', 'heun', 'rk4') - only for method='ode'
            ode_steps: Number of ODE integration steps - only for method='ode'
            ode_t_start: ODE starting time (default: T-1) - only for method='ode'
            ode_t_end: ODE ending time (default: 0) - only for method='ode'
            ode_eps: Small epsilon for numerical stability - only for method='ode'
        
        Returns:
            Generated samples [B, 1, H, W]
        """
        batch_size = condition.shape[0]
        H, W = condition.shape[2], condition.shape[3]
        
        if shape is None:
            shape = (batch_size, 1, H, W)
        
        # Ensure float types
        condition = condition.float()
        if lr_guidance is not None:
            lr_guidance = lr_guidance.float()
        
        # Start from pure noise
        noise = torch.randn(shape, device=self.device, dtype=torch.float32)
        
        # Blend LR PET into initial noise if provided (for DDIM/DDPM)
        # For ODE, we do not blend into initial noise as the ODE is deterministic
        if method != 'ode' and lr_guidance is not None and lr_blend > 0:
            # Normalize lr_guidance to similar range as noise
            # Scale to roughly match noise statistics for better blending
            img = (1 - lr_blend) * noise + lr_blend * lr_guidance
        else:
            img = noise
        
        if method == 'ddim':
            return self._ddim_sample(img, condition, ddim_steps, eta, lr_guidance, lr_blend)
        elif method == 'ddpm':
            return self._ddpm_sample(img, condition, lr_guidance, lr_blend)
        elif method == 'ode':
            return self._ode_sample(img, condition, ode_steps, ode_solver,
                                    ode_t_start, ode_t_end, ode_eps,
                                    lr_guidance, lr_blend)
        else:
            raise ValueError(f"Unknown sampling method: {method}. Choose from 'ddim', 'ddpm', 'ode'")
    
    def _ddim_sample(self, img, condition, ddim_steps, eta, lr_guidance=None, lr_blend=0.0):
        """DDIM sampling loop with optional LR PET guidance."""
        # Create timestep sequence
        c = self.timesteps // ddim_steps
        timesteps_seq = list(range(0, self.timesteps, c))
        
        # Get diffusion parameters (ensure float for consistency)
        alphas_cumprod = self.diffusion.alphas_cumprod.float()
        
        # Ensure img is float
        img = img.float()
        
        for i in tqdm(reversed(range(len(timesteps_seq))), desc="DDIM Sampling", total=len(timesteps_seq), leave=False):
            t = timesteps_seq[i]
            t_tensor = torch.full((img.shape[0],), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            pred_noise = self._model_forward(img, t_tensor, condition)
            
            # DDIM update
            alpha_t = alphas_cumprod[t]
            
            if i > 0:
                t_prev = timesteps_seq[i - 1]
                alpha_t_prev = alphas_cumprod[t_prev]
            else:
                alpha_t_prev = torch.tensor(1.0, device=self.device)
            
            # Predicted x0
            pred_x0 = (img - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, 0, 2)  # Clip for stability (PET uses arcsinh normalization [0, ~1.2+])
            
            # LR guidance: blend predicted x0 with lr_guidance
            # This encourages the generation to stay close to LR structure
            if lr_guidance is not None and lr_blend > 0:
                # Apply guidance with decaying weight (stronger at early steps)
                guidance_weight = lr_blend * (t / self.timesteps)  # Decay as t decreases
                pred_x0 = (1 - guidance_weight) * pred_x0 + guidance_weight * lr_guidance
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev - eta**2 * (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) * pred_noise
            
            # Random noise (only if eta > 0 and not last step)
            if eta > 0 and i > 0:
                noise = torch.randn_like(img)
                sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
            else:
                noise = 0
                sigma = 0
            
            # Update
            img = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma * noise
        
        return img
    
    def _ddpm_sample(self, img, condition, lr_guidance=None, lr_blend=0.0):
        """DDPM sampling loop with optional LR PET guidance."""
        betas = self.diffusion.betas.float()
        alphas = 1. - betas
        alphas_cumprod = self.diffusion.alphas_cumprod.float()
        
        # Ensure img is float
        img = img.float()
        
        for t in tqdm(reversed(range(self.timesteps)), desc="DDPM Sampling", total=self.timesteps, leave=False):
            t_tensor = torch.full((img.shape[0],), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            pred_noise = self._model_forward(img, t_tensor, condition)
            
            # Coefficients
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            beta_t = betas[t]
            
            # Mean (predicted x0 implicitly)
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1 - alpha_cumprod_t)
            mean = coef1 * (img - coef2 * pred_noise)
            
            # LR guidance: blend mean with lr_guidance
            if lr_guidance is not None and lr_blend > 0:
                guidance_weight = lr_blend * (t / self.timesteps)
                mean = (1 - guidance_weight) * mean + guidance_weight * lr_guidance
            
            # Variance
            if t > 0:
                noise = torch.randn_like(img)
                sigma = torch.sqrt(beta_t)
                img = mean + sigma * noise
            else:
                img = mean
        
        return img
    
    def _ode_sample(self, img, condition, ode_steps, ode_solver='heun', 
                    ode_t_start=None, ode_t_end=None, ode_eps=1e-3,
                    lr_guidance=None, lr_blend=0.0):
        """
        ODE-based sampling using Probability Flow ODE.
        
        This implements deterministic sampling via an ODE formulation that is
        mathematically equivalent to DDIM with eta=0. We use the "alpha-based"
        parameterization which is more numerically stable than the continuous
        VP-SDE formulation.
        
        The key insight is that DDIM with eta=0 can be viewed as solving:
            x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1-alpha_{t-1}) * direction
        where direction = (x_t - sqrt(alpha_t) * pred_x0) / sqrt(1-alpha_t)
        
        This is the deterministic probability flow ODE in discrete form.
        We provide Euler/Heun/RK4 solvers for different accuracy/speed tradeoffs.
        
        Args:
            img: Initial noisy image [B, 1, H, W] (starts from pure noise)
            condition: Condition tensor [B, C_cond, H, W]
            ode_steps: Number of ODE integration steps
            ode_solver: Solver type ('euler', 'heun', 'rk4')
            ode_t_start: Starting time index (default: T-1, max noise)
            ode_t_end: Ending time index (default: 0, clean image)
            ode_eps: Small epsilon to avoid t=0 numerical issues
            lr_guidance: LR PET guidance tensor [B, 1, H, W]
            lr_blend: Blend weight for LR guidance
        
        Returns:
            Generated samples [B, 1, H, W]
        """
        # Get diffusion parameters (ensure float32 for numerical stability)
        alphas_cumprod = self.diffusion.alphas_cumprod.float()
        
        # Ensure img is float32
        img = img.float()
        
        # Time grid setup: ODE goes from high noise (t_start) to low noise (t_end)
        # We use discrete time indices [0, T-1] as in DDPM/DDIM
        t_start = int(ode_t_start if ode_t_start is not None else (self.timesteps - 1))
        t_end = int(ode_t_end if ode_t_end is not None else 0)
        
        # Ensure t_end is not exactly 0 for stability (same as DDIM)
        t_end = max(t_end, int(ode_eps * self.timesteps))
        
        # Create time grid (strictly decreasing integer indices)
        # Similar to DDIM's time subsequence but with uniform spacing for ODE
        t_grid = torch.linspace(t_start, t_end, ode_steps + 1, device=self.device)
        t_grid = t_grid.round().long()  # Round to integer indices
        
        # Log ODE setup info
        print(f"[ODE] Solver: {ode_solver}, Steps: {ode_steps}")
        print(f"[ODE] Time grid: t_start={t_start} -> t_end={t_end}")
        
        # ODE integration loop using the DDIM-equivalent formulation
        x = img
        for step_idx in tqdm(range(ode_steps), desc=f"ODE Sampling ({ode_solver})", leave=False):
            t_cur = t_grid[step_idx].item()
            t_next = t_grid[step_idx + 1].item()
            
            # Skip if no progress (can happen due to rounding)
            if t_cur == t_next:
                continue
            
            # Check for NaN
            if not torch.isfinite(x).all():
                print(f"[ODE ERROR] NaN detected at step {step_idx}, t={t_cur}")
                print(f"  x range: [{x.min():.4f}, {x.max():.4f}]")
                raise ValueError("NaN encountered in ODE sampling")
            
            # Apply ODE solver step
            if ode_solver == 'euler':
                x = self._ode_euler_step_ddim(x, t_cur, t_next, condition, alphas_cumprod)
            elif ode_solver == 'heun':
                x = self._ode_heun_step_ddim(x, t_cur, t_next, condition, alphas_cumprod)
            elif ode_solver == 'rk4':
                x = self._ode_rk4_step_ddim(x, t_cur, t_next, condition, alphas_cumprod)
            else:
                raise ValueError(f"Unknown ODE solver: {ode_solver}")
            
            # Apply LR guidance (blend pred_x0 with lr_guidance)
            if lr_guidance is not None and lr_blend > 0 and t_cur > t_end:
                # Compute predicted x0 from current x
                t_tensor = torch.full((x.shape[0],), t_next, device=self.device, dtype=torch.long)
                eps_pred = self._model_forward(x, t_tensor, condition)
                
                alpha_bar_t = alphas_cumprod[t_next]
                pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
                pred_x0 = torch.clamp(pred_x0, 0, 2)
                
                # Blend with LR guidance (decaying weight)
                guidance_weight = lr_blend * (t_next / self.timesteps)
                guided_x0 = (1 - guidance_weight) * pred_x0 + guidance_weight * lr_guidance
                
                # Re-construct x at t_next with guided x0
                if t_next > 0:
                    x = torch.sqrt(alpha_bar_t) * guided_x0 + torch.sqrt(1 - alpha_bar_t) * eps_pred
        
        # Final step: if we didn't reach t=0, do one final denoising step to get clean x0
        # This is crucial to remove residual noise in the background
        final_t = t_grid[-1].item()
        if final_t > 0:
            t_tensor = torch.full((x.shape[0],), final_t, device=self.device, dtype=torch.long)
            eps_pred = self._model_forward(x, t_tensor, condition)
            
            alpha_bar_t = alphas_cumprod[final_t]
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            pred_x0 = torch.clamp(pred_x0, 0, 2)
            
            # Apply final LR guidance if enabled
            if lr_guidance is not None and lr_blend > 0:
                guidance_weight = lr_blend * 0.01  # Very small weight at final step
                pred_x0 = (1 - guidance_weight) * pred_x0 + guidance_weight * lr_guidance
            
            x = pred_x0  # Return predicted clean image
        
        return x
    
    def _ddim_step(self, x, t_cur, t_next, condition, alphas_cumprod):
        """
        Single DDIM step (deterministic, eta=0).
        This is the fundamental ODE step that maps x_t -> x_{t-1}.
        
        DDIM update (eta=0):
            pred_x0 = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
            x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1-alpha_{t-1}) * eps
        
        This is equivalent to following the probability flow ODE.
        """
        t_tensor = torch.full((x.shape[0],), t_cur, device=self.device, dtype=torch.long)
        
        # Predict noise
        eps_pred = self._model_forward(x, t_tensor, condition)
        
        # Get alpha values
        alpha_t = alphas_cumprod[t_cur]
        alpha_t_next = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=self.device)
        
        # Predict x0
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
        pred_x0 = torch.clamp(pred_x0, 0, 2)  # Clip for stability (PET uses arcsinh normalization [0, ~1.2+])
        
        # DDIM update (eta=0, deterministic)
        x_next = torch.sqrt(alpha_t_next) * pred_x0 + torch.sqrt(1 - alpha_t_next) * eps_pred
        
        return x_next, pred_x0, eps_pred
    
    def _ode_euler_step_ddim(self, x, t_cur, t_next, condition, alphas_cumprod):
        """
        Euler method using DDIM formulation.
        Simply one DDIM step from t_cur to t_next.
        """
        x_next, _, _ = self._ddim_step(x, t_cur, t_next, condition, alphas_cumprod)
        return x_next
    
    def _ode_heun_step_ddim(self, x, t_cur, t_next, condition, alphas_cumprod):
        """
        Heun's method (improved Euler / 2nd order RK) using DDIM formulation.
        
        1. Euler prediction: x_euler = DDIM_step(x, t_cur -> t_next)
        2. Correction: use both x and x_euler to get better estimate
        
        For the DDIM-based ODE, Heun's method provides:
        - Better accuracy than Euler
        - Requires 2 model evaluations per step
        """
        # Step 1: Euler prediction
        t_tensor_cur = torch.full((x.shape[0],), t_cur, device=self.device, dtype=torch.long)
        eps_pred_cur = self._model_forward(x, t_tensor_cur, condition)
        
        alpha_t_cur = alphas_cumprod[t_cur]
        alpha_t_next = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=self.device)
        
        pred_x0_cur = (x - torch.sqrt(1 - alpha_t_cur) * eps_pred_cur) / torch.sqrt(alpha_t_cur)
        pred_x0_cur = torch.clamp(pred_x0_cur, 0, 2)
        
        x_euler = torch.sqrt(alpha_t_next) * pred_x0_cur + torch.sqrt(1 - alpha_t_next) * eps_pred_cur
        
        # Step 2: Correction using x_euler
        t_tensor_next = torch.full((x.shape[0],), t_next, device=self.device, dtype=torch.long)
        eps_pred_next = self._model_forward(x_euler, t_tensor_next, condition)
        
        pred_x0_next = (x_euler - torch.sqrt(1 - alpha_t_next) * eps_pred_next) / torch.sqrt(alpha_t_next)
        pred_x0_next = torch.clamp(pred_x0_next, 0, 2)
        
        # Average the two x0 predictions (Heun's method)
        pred_x0_avg = 0.5 * (pred_x0_cur + pred_x0_next)
        eps_avg = 0.5 * (eps_pred_cur + eps_pred_next)
        
        x_heun = torch.sqrt(alpha_t_next) * pred_x0_avg + torch.sqrt(1 - alpha_t_next) * eps_avg
        
        return x_heun
    
    def _ode_rk4_step_ddim(self, x, t_cur, t_next, condition, alphas_cumprod):
        """
        4th-order Runge-Kutta using DDIM formulation.
        
        This provides high accuracy but requires 4 model evaluations per step.
        For most cases, Heun (2nd order) is sufficient and more efficient.
        
        Implementation uses intermediate time points between t_cur and t_next.
        """
        # Intermediate time point
        t_mid = (t_cur + t_next) // 2
        
        alpha_t_cur = alphas_cumprod[t_cur]
        alpha_t_mid = alphas_cumprod[t_mid]
        alpha_t_next = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=self.device)
        
        # k1: slope at t_cur
        t_tensor = torch.full((x.shape[0],), t_cur, device=self.device, dtype=torch.long)
        eps_1 = self._model_forward(x, t_tensor, condition)
        pred_x0_1 = (x - torch.sqrt(1 - alpha_t_cur) * eps_1) / torch.sqrt(alpha_t_cur)
        pred_x0_1 = torch.clamp(pred_x0_1, 0, 2)
        
        # k2: slope at t_mid using k1
        x_k1 = torch.sqrt(alpha_t_mid) * pred_x0_1 + torch.sqrt(1 - alpha_t_mid) * eps_1
        t_tensor = torch.full((x.shape[0],), t_mid, device=self.device, dtype=torch.long)
        eps_2 = self._model_forward(x_k1, t_tensor, condition)
        pred_x0_2 = (x_k1 - torch.sqrt(1 - alpha_t_mid) * eps_2) / torch.sqrt(alpha_t_mid)
        pred_x0_2 = torch.clamp(pred_x0_2, 0, 2)
        
        # k3: slope at t_mid using k2
        x_k2 = torch.sqrt(alpha_t_mid) * pred_x0_2 + torch.sqrt(1 - alpha_t_mid) * eps_2
        eps_3 = self._model_forward(x_k2, t_tensor, condition)
        pred_x0_3 = (x_k2 - torch.sqrt(1 - alpha_t_mid) * eps_3) / torch.sqrt(alpha_t_mid)
        pred_x0_3 = torch.clamp(pred_x0_3, 0, 2)
        
        # k4: slope at t_next using k3
        x_k3 = torch.sqrt(alpha_t_next) * pred_x0_3 + torch.sqrt(1 - alpha_t_next) * eps_3
        t_tensor = torch.full((x.shape[0],), t_next, device=self.device, dtype=torch.long)
        eps_4 = self._model_forward(x_k3, t_tensor, condition)
        pred_x0_4 = (x_k3 - torch.sqrt(1 - alpha_t_next) * eps_4) / torch.sqrt(alpha_t_next)
        pred_x0_4 = torch.clamp(pred_x0_4, 0, 2)
        
        # RK4 combination
        pred_x0_rk4 = (pred_x0_1 + 2*pred_x0_2 + 2*pred_x0_3 + pred_x0_4) / 6.0
        eps_rk4 = (eps_1 + 2*eps_2 + 2*eps_3 + eps_4) / 6.0
        
        x_rk4 = torch.sqrt(alpha_t_next) * pred_x0_rk4 + torch.sqrt(1 - alpha_t_next) * eps_rk4
        
        return x_rk4
    
    def _model_forward(self, img, t, condition):
        """
        Forward pass through model with appropriate conditioning.
        
        Args:
            img: Noisy image [B, 1, H, W]
            t: Timesteps [B]
            condition: Condition tensor [B, C_cond, H, W]
        
        Returns:
            Predicted noise [B, 1, H, W]
        """
        if self.model_type == 'concat':
            # Concatenate condition with noisy image
            model_input = torch.cat([img, condition], dim=1)
            return self.model(model_input, t)
        else:  # crossattn
            # Pass condition as context
            return self.model(img, t, condition)


# ============================================================================
# Output saving functions
# ============================================================================

def save_sample_outputs(sample_data, out_dir, key, save_npy=True):
    """
    Save sample outputs (images and numpy arrays) for single image/slice.
    
    Args:
        sample_data: Dict containing sample tensors
        out_dir: Output directory
        key: Sample key for naming
        save_npy: Whether to save numpy arrays
    """
    # Create sample directory
    sample_dir = os.path.join(out_dir, f"sample_{key}")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Convert tensors to numpy
    data_np = {}
    for name, tensor in sample_data.items():
        if isinstance(tensor, torch.Tensor):
            data_np[name] = tensor.cpu().numpy()
        else:
            data_np[name] = tensor
    
    # Save numpy arrays
    if save_npy:
        npz_path = os.path.join(sample_dir, 'data.npz')
        np.savez(npz_path, **data_np)
    
    # Create visualization figure
    has_lr = 'lr' in data_np and 'lr_up' in data_np
    has_dc_lr = 'dc_lr_pet' in data_np
    
    # Determine if this is DC mode (lr and lr_up have same size as gt)
    is_dc_mode = False
    if has_lr and has_dc_lr:
        lr_shape = data_np['lr'].shape
        gt_shape = data_np['gt_pet'].shape
        # In DC mode, lr has the same spatial size as gt (degraded, not downsampled)
        if lr_shape[-2:] == gt_shape[-2:]:
            is_dc_mode = True
    
    # Calculate number of columns
    # DC mode: CT, DC_LR (degraded), GT, Pred, Diff = 5 cols
    # Non-DC mode: CT, LR, LR_up, GT, Pred, Diff = 6 cols (or without lr: 4 cols)
    n_cols = 4  # CT, GT, Pred, Diff (base)
    if is_dc_mode:
        n_cols += 1  # Only show DC degraded image (same size as GT)
    elif has_lr:
        n_cols += 2  # Show actual LR and upsampled version
    elif has_dc_lr:
        n_cols += 1  # Only DC LR available
    
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    idx = 0
    
    # CT in PET space
    ct = data_np['ct_in_pet_space'][0, 0]  # Remove batch and channel dims
    axes[idx].imshow(ct, cmap='gray')
    axes[idx].set_title('CT (condition)')
    axes[idx].axis('off')
    idx += 1
    
    # LR/Degraded PET visualization
    if is_dc_mode:
        # DC mode: show the degraded PET (physical space, stored in dc_lr_pet)
        dc_lr = data_np['dc_lr_pet']
        while dc_lr.ndim > 2:
            dc_lr = dc_lr[0]
        if dc_lr.ndim == 2:
            axes[idx].imshow(dc_lr, cmap='hot')
            axes[idx].set_title(f'Degraded PET ({dc_lr.shape[0]}x{dc_lr.shape[1]})')
            axes[idx].axis('off')
            idx += 1
    elif has_lr:
        # Non-DC mode: show actual LR and upsampled version
        lr = data_np['lr']
        while lr.ndim > 2:
            lr = lr[0]
        if lr.ndim == 2:
            axes[idx].imshow(lr, cmap='hot')
            axes[idx].set_title(f'LR PET ({lr.shape[0]}x{lr.shape[1]})')
            axes[idx].axis('off')
            idx += 1
        
        lr_up = data_np['lr_up']
        while lr_up.ndim > 2:
            lr_up = lr_up[0]
        if lr_up.ndim == 2:
            axes[idx].imshow(lr_up, cmap='hot')
            axes[idx].set_title('LR PET (upsampled)')
            axes[idx].axis('off')
            idx += 1
    elif has_dc_lr:
        # Only DC LR available (ct mode with DC)
        dc_lr = data_np['dc_lr_pet']
        while dc_lr.ndim > 2:
            dc_lr = dc_lr[0]
        if dc_lr.ndim == 2:
            axes[idx].imshow(dc_lr, cmap='hot')
            axes[idx].set_title(f'DC Degraded ({dc_lr.shape[0]}x{dc_lr.shape[1]})')
            axes[idx].axis('off')
            idx += 1
    
    # GT PET
    gt = data_np['gt_pet'][0, 0]  # Remove batch and channel dims
    axes[idx].imshow(gt, cmap='hot')
    axes[idx].set_title('GT PET')
    axes[idx].axis('off')
    idx += 1
    
    # Predicted PET
    pred = data_np['pred_pet'][0, 0]  # Remove batch and channel dims
    axes[idx].imshow(pred, cmap='hot')
    axes[idx].set_title('Pred PET')
    axes[idx].axis('off')
    idx += 1
    
    # Absolute difference
    diff = np.abs(gt - pred)
    im = axes[idx].imshow(diff, cmap='hot')
    axes[idx].set_title('|GT - Pred|')
    axes[idx].axis('off')
    plt.colorbar(im, ax=axes[idx], fraction=0.046)
    
    plt.tight_layout()
    fig_path = os.path.join(sample_dir, 'visualization.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return sample_dir


def save_case_outputs(case_dir, slice_data_list, case_metrics, save_npy=True):
    """
    Save outputs for a single case (all slices).
    
    Directory structure:
        case_dir/
            ct/          - CT images (aligned to PET space)
            lrpet/       - LR PET and upsampled (if enabled)
            pred/        - Predicted PET
            gt/          - Ground truth PET
            diff/        - Absolute difference
            metrics.csv  - Per-slice metrics
            data.npz     - All slice data (optional)
    
    Args:
        case_dir: Output directory for this case
        slice_data_list: List of dicts, each containing slice data
        case_metrics: List of metrics dicts for each slice
        save_npy: Whether to save numpy arrays
    """
    # Create subdirectories
    subdirs = ['ct', 'pred', 'gt', 'diff']
    has_lr = any('lr' in s for s in slice_data_list)
    if has_lr:
        subdirs.append('lrpet')
    
    for subdir in subdirs:
        os.makedirs(os.path.join(case_dir, subdir), exist_ok=True)
    
    # Save each slice
    for i, (slice_data, metrics) in enumerate(zip(slice_data_list, case_metrics)):
        key = slice_data.get('key', str(i))
        
        # Convert tensors to numpy
        data_np = {}
        for name, tensor in slice_data.items():
            if isinstance(tensor, torch.Tensor):
                data_np[name] = tensor.cpu().numpy()
            else:
                data_np[name] = tensor
        
        # Save CT (aligned to PET space)
        if 'ct_in_pet_space' in data_np:
            ct_img = data_np['ct_in_pet_space']
            if ct_img.ndim == 4:  # [B, C, H, W]
                ct_img = ct_img[0, 0]
            elif ct_img.ndim == 3:  # [C, H, W]
                ct_img = ct_img[0]
            np.save(os.path.join(case_dir, 'ct', f'slice_{key}.npy'), ct_img)
        
        # Save LR PET (if available)
        if has_lr and 'lr' in data_np:
            lr_img = data_np['lr']
            # Handle different dimensions robustly
            while lr_img.ndim > 2:
                lr_img = lr_img[0]
            np.save(os.path.join(case_dir, 'lrpet', f'slice_{key}_lr.npy'), lr_img)
            
            if 'lr_up' in data_np:
                lr_up_img = data_np['lr_up']
                while lr_up_img.ndim > 2:
                    lr_up_img = lr_up_img[0]
                np.save(os.path.join(case_dir, 'lrpet', f'slice_{key}_lr_up.npy'), lr_up_img)
        
        # Save GT PET
        if 'gt_pet' in data_np:
            gt_img = data_np['gt_pet']
            if gt_img.ndim == 4:
                gt_img = gt_img[0, 0]
            elif gt_img.ndim == 3:
                gt_img = gt_img[0]
            np.save(os.path.join(case_dir, 'gt', f'slice_{key}.npy'), gt_img)
        
        # Save Predicted PET
        if 'pred_pet' in data_np:
            pred_img = data_np['pred_pet']
            if pred_img.ndim == 4:
                pred_img = pred_img[0, 0]
            elif pred_img.ndim == 3:
                pred_img = pred_img[0]
            np.save(os.path.join(case_dir, 'pred', f'slice_{key}.npy'), pred_img)
        
        # Save absolute difference
        if 'gt_pet' in data_np and 'pred_pet' in data_np:
            gt_img = data_np['gt_pet']
            pred_img = data_np['pred_pet']
            if gt_img.ndim == 4:
                gt_img = gt_img[0, 0]
                pred_img = pred_img[0, 0]
            elif gt_img.ndim == 3:
                gt_img = gt_img[0]
                pred_img = pred_img[0]
            diff_img = np.abs(gt_img - pred_img)
            np.save(os.path.join(case_dir, 'diff', f'slice_{key}.npy'), diff_img)
    
    # Save per-slice metrics CSV
    csv_path = os.path.join(case_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['slice_key', 'mse', 'psnr', 'ssim'])
        writer.writeheader()
        for m in case_metrics:
            writer.writerow({
                'slice_key': m['key'],
                'mse': m['mse'],
                'psnr': m['psnr'],
                'ssim': m['ssim']
            })
        
        # Add case summary row
        if case_metrics:
            avg_mse = np.mean([m['mse'] for m in case_metrics])
            avg_psnr = np.mean([m['psnr'] for m in case_metrics])
            avg_ssim = np.mean([m['ssim'] for m in case_metrics])
            writer.writerow({
                'slice_key': 'CASE_MEAN',
                'mse': avg_mse,
                'psnr': avg_psnr,
                'ssim': avg_ssim
            })
    
    # Optionally save all data as npz
    if save_npy:
        all_data = {
            'num_slices': len(slice_data_list),
            'slice_keys': [s.get('key', str(i)) for i, s in enumerate(slice_data_list)],
            'metrics': case_metrics
        }
        # Save compact summary (not full images to save space)
        np.savez(os.path.join(case_dir, 'case_summary.npz'), **all_data)
    
    return csv_path


def create_case_visualization(case_dir, slice_data_list, slices_per_image=10):
    """
    Create visualizations for a case, saving multiple images with batches of slices.
    
    Each image contains up to `slices_per_image` slices. All images are saved
    to a 'vis' subdirectory within the case directory.
    
    Args:
        case_dir: Output directory for this case
        slice_data_list: List of dicts, each containing slice data
        slices_per_image: Number of slices per visualization image (default: 10)
    """
    n_total = len(slice_data_list)
    if n_total == 0:
        return
    
    # Create vis directory
    vis_dir = os.path.join(case_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    
    has_lr = 'lr' in slice_data_list[0] if slice_data_list else False
    n_cols = 5 if has_lr else 4  # CT, (LR_up), GT, Pred, Diff
    
    # Calculate number of images needed
    n_images = (n_total + slices_per_image - 1) // slices_per_image
    
    for img_idx in range(n_images):
        start_idx = img_idx * slices_per_image
        end_idx = min(start_idx + slices_per_image, n_total)
        batch_slices = slice_data_list[start_idx:end_idx]
        n_rows = len(batch_slices)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for row, slice_data in enumerate(batch_slices):
            key = slice_data.get('key', str(start_idx + row))
            
            # Convert to numpy
            data_np = {}
            for name, tensor in slice_data.items():
                if isinstance(tensor, torch.Tensor):
                    data_np[name] = tensor.cpu().numpy()
                else:
                    data_np[name] = tensor
            
            col = 0
            
            # CT
            if 'ct_in_pet_space' in data_np:
                ct = data_np['ct_in_pet_space']
                if ct.ndim == 4:
                    ct = ct[0, 0]
                elif ct.ndim == 3:
                    ct = ct[0]
                axes[row, col].imshow(ct, cmap='gray')
                axes[row, col].set_title(f'CT (slice {key})')
                axes[row, col].axis('off')
            col += 1
            
            # LR PET (if available)
            if has_lr and 'lr_up' in data_np:
                lr_up = data_np['lr_up']
                while lr_up.ndim > 2:
                    lr_up = lr_up[0]
                axes[row, col].imshow(lr_up, cmap='hot')
                axes[row, col].set_title('LR PET (up)')
                axes[row, col].axis('off')
                col += 1
            
            # GT PET
            if 'gt_pet' in data_np:
                gt = data_np['gt_pet']
                if gt.ndim == 4:
                    gt = gt[0, 0]
                elif gt.ndim == 3:
                    gt = gt[0]
                axes[row, col].imshow(gt, cmap='hot')
                axes[row, col].set_title('GT PET')
                axes[row, col].axis('off')
            col += 1
            
            # Pred PET
            if 'pred_pet' in data_np:
                pred = data_np['pred_pet']
                if pred.ndim == 4:
                    pred = pred[0, 0]
                elif pred.ndim == 3:
                    pred = pred[0]
                axes[row, col].imshow(pred, cmap='hot')
                axes[row, col].set_title('Pred PET')
                axes[row, col].axis('off')
            col += 1
            
            # Difference
            if 'gt_pet' in data_np and 'pred_pet' in data_np:
                gt = data_np['gt_pet']
                pred = data_np['pred_pet']
                if gt.ndim == 4:
                    gt = gt[0, 0]
                    pred = pred[0, 0]
                elif gt.ndim == 3:
                    gt = gt[0]
                    pred = pred[0]
                diff = np.abs(gt - pred)
                im = axes[row, col].imshow(diff, cmap='hot')
                axes[row, col].set_title('|GT - Pred|')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046)
        
        plt.tight_layout()
        # Save with slice range in filename
        fig_path = os.path.join(vis_dir, f'slices_{start_idx+1:03d}-{end_idx:03d}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# Utility functions
# ============================================================================

def sanitize_for_json(obj):
    """Convert non-JSON-serializable values to serializable equivalents.
    
    Handles:
    - float('inf') / float('-inf') -> None
    - numpy scalar types (float32, float64, int32, int64, etc.) -> native Python types
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if obj == float('inf'):
            return None  # Convert infinity to None for JSON compatibility
        elif obj == float('-inf'):
            return None
        return obj
    elif isinstance(obj, np.floating):  # Handles float32, float64, etc.
        val = float(obj)
        if val == float('inf') or val == float('-inf'):
            return None
        return val
    elif isinstance(obj, np.integer):  # Handles int32, int64, etc.
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# Main function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CT to PET Conditional Diffusion Sampling (2D)')
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    
    # Model and condition settings
    parser.add_argument('--model_type', type=str, choices=['concat', 'crossattn'], required=True,
                        help='Model type: concat or crossattn')
    parser.add_argument('--cond', type=str, choices=['ct', 'ct+lrpet'], default='ct',
                        help='Condition mode: ct or ct+lrpet')
    parser.add_argument('--lr_factor', type=int, default=4,
                        help='LR PET downsampling factor (default: 4)')
    parser.add_argument('--lr_blend', type=float, default=0.3,
                        help='Blend weight for LR PET into noisy input (0-1, default: 0.3). '
                             'x_init = (1-w)*noise + w*lr_up. Only used when cond=ct+lrpet')
    parser.add_argument('--lr_upsample', type=str, default='bilinear',
                        choices=['nearest', 'bilinear', 'zero_pad'],
                        help='Upsampling mode for LR PET (default: bilinear). '
                             'nearest: nearest neighbor, bilinear: bilinear interpolation, '
                             'zero_pad: zero-padding expansion (lr pixel at top-left, rest zeros)')
    
    # Sampling settings
    parser.add_argument('--num_samples', type=int, default=8,
                        help='Number of samples to generate')
    parser.add_argument('--shuffle', action='store_true',
                        help='Randomly shuffle samples before selection (default: sequential)')
    parser.add_argument('--sampler', type=str, choices=['ddpm', 'ddim', 'ode'], default=None,
                        help='Sampling method: ddpm, ddim, or ode (default: from config, usually ddim)')
    parser.add_argument('--method', type=str, choices=['ddim', 'ddpm', 'ode'], default=None,
                        help='(Deprecated, use --sampler) Sampling method')
    parser.add_argument('--ddim_steps', type=int, default=None,
                        help='DDIM steps (default: from config)')
    parser.add_argument('--eta', type=float, default=None,
                        help='DDIM eta (default: from config)')
    parser.add_argument('--data_range', type=float, default=None,
                        help='Data range for PSNR/SSIM (default: from config)')
    
    # ODE sampler settings
    parser.add_argument('--ode_solver', type=str, choices=['euler', 'heun', 'rk4'], default='heun',
                        help='ODE solver type (default: heun). '
                             'euler: 1st order, heun: 2nd order (improved Euler), rk4: 4th order Runge-Kutta')
    parser.add_argument('--ode_steps', type=int, default=50,
                        help='Number of ODE integration steps (default: 50)')
    parser.add_argument('--ode_t_start', type=float, default=None,
                        help='ODE starting time (default: T-1, max noise level)')
    parser.add_argument('--ode_t_end', type=float, default=None,
                        help='ODE ending time (default: 0, clean image)')
    parser.add_argument('--ode_eps', type=float, default=1e-3,
                        help='Small epsilon to avoid t=0 numerical issues (default: 1e-3)')
    
    # Output settings
    parser.add_argument('--out_dir', type=str, default='samples/ct2pet_conditional',
                        help='Output directory')
    parser.add_argument('--phase', type=str, choices=['train', 'val', 'test'], default='val',
                        help='Dataset phase to sample from')
    parser.add_argument('--unit', type=str, choices=['image', 'case'], default='image',
                        help='Output unit: image (per-slice) or case (per-patient)')
    parser.add_argument('--num_cases', type=int, default=None,
                        help='Number of cases to process (only for --unit case)')
    parser.add_argument('--max_slices_per_case', type=int, default=None,
                        help='Max slices per case (only for --unit case, None=all)')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (only 4 samples, verbose output)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID to use (default: use CUDA_VISIBLE_DEVICES or GPU 0)')
    
    # =========================================================================
    # Data Consistency (DC) Arguments - Physical PET Simulation
    # Priority: CLI > Config file > Hardcoded defaults (in dc_config builder)
    # All DC args default to None so config file values can take effect
    # =========================================================================
    # DC control
    parser.add_argument('--dc_enable', action='store_true',
                        help='Enable Data Consistency (DC) correction during sampling')
    parser.add_argument('--dc_loss_type', type=str, choices=['l2', 'poisson'], default=None,
                        help='DC loss type: l2 (Gaussian) or poisson (Poisson NLL). Default from config.')
    parser.add_argument('--dc_step_size', type=float, default=None,
                        help='DC gradient step size. Default from config.')
    parser.add_argument('--dc_num_inner', type=int, default=None,
                        help='Number of inner DC gradient steps per diffusion step. Default from config.')
    parser.add_argument('--dc_start_step', type=int, default=None,
                        help='Start applying DC after this diffusion step. Default from config.')
    parser.add_argument('--dc_end_step', type=int, default=None,
                        help='Stop applying DC after this step (None = until end). Default from config.')
    
    # Physical degradation parameters
    parser.add_argument('--dc_fov_mm', type=float, default=None,
                        help='FOV in mm (default: auto from image affine)')
    parser.add_argument('--dc_fwhm_mm', type=float, default=None,
                        help='PSF FWHM in mm. Default from config.')
    parser.add_argument('--dc_dose_alpha', type=float, default=None,
                        help='Dose scaling: 1.0=full, 0.1=10%% dose. Default from config.')
    parser.add_argument('--dc_num_angles', type=int, default=None,
                        help='Number of projection angles. Default from config.')
    parser.add_argument('--dc_rebin_factor', type=int, default=None,
                        help='Angular rebinning factor. Default from config.')
    parser.add_argument('--dc_rebin_radial_factor', type=int, default=None,
                        help='Radial rebinning factor. Default from config.')
    parser.add_argument('--dc_background_beta', type=float, default=None,
                        help='Background fraction. Default from config.')
    
    # Debug/save options
    parser.add_argument('--dc_save_simulation', action='store_true',
                        help='Save simulated sinograms and degraded PET images')
    parser.add_argument('--dc_save_every', type=int, default=None,
                        help='Save simulation outputs every N samples. Default from config.')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set random seed
    set_seed(args.seed)
    
    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Device selection
    if args.gpu is not None:
        if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
            device = torch.device(f'cuda:{args.gpu}')
            torch.cuda.set_device(args.gpu)
            print(f"[Device] Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        else:
            print(f"[Warning] GPU {args.gpu} not available, falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Get sampling parameters (CLI overrides config)
    # --sampler takes precedence over --method (deprecated)
    sampling_config = config['diffusion']['sampling']
    if args.sampler is not None:
        method = args.sampler
    elif args.method is not None:
        method = args.method
        print("[WARNING] --method is deprecated, use --sampler instead")
    else:
        method = sampling_config.get('method', 'ddim')
    
    ddim_steps = args.ddim_steps or sampling_config.get('ddim_steps', 50)
    eta = args.eta if args.eta is not None else sampling_config.get('eta', 0.0)
    data_range = args.data_range or config['evaluation']['metrics']['data_range']
    
    # ODE parameters
    ode_solver = args.ode_solver
    ode_steps = args.ode_steps
    ode_t_start = args.ode_t_start
    ode_t_end = args.ode_t_end
    ode_eps = args.ode_eps
    
    # Number of samples
    num_samples = 4 if args.debug else args.num_samples
    
    # Create output directory with timestamp
    # Include sampler info in directory name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if method == 'ode':
        out_dir_name = f"{args.model_type}_{args.cond}_ode_{ode_solver}_{ode_steps}steps_{timestamp}"
    else:
        out_dir_name = f"{args.model_type}_{args.cond}_{method}_{timestamp}"
    out_dir = os.path.join(args.out_dir, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 70)
    print("CT to PET Conditional Diffusion Sampling (2D)")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {args.model_type}")
    print(f"Condition mode: {args.cond}")
    if args.cond == 'ct+lrpet':
        print(f"LR factor: {args.lr_factor}")
        print(f"LR upsample: {args.lr_upsample}")
        print(f"LR blend: {args.lr_blend}")
    print(f"Sampler: {method}")
    if method == 'ddim':
        print(f"  DDIM steps: {ddim_steps}")
        print(f"  Eta: {eta}")
    elif method == 'ode':
        print(f"  ODE solver: {ode_solver}")
        print(f"  ODE steps: {ode_steps}")
        if ode_t_start is not None:
            print(f"  ODE t_start: {ode_t_start}")
        if ode_t_end is not None:
            print(f"  ODE t_end: {ode_t_end}")
        print(f"  ODE eps: {ode_eps}")
    print(f"Data range: {data_range}")
    print(f"Num samples: {num_samples}")
    print(f"Output dir: {out_dir}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Output unit: {args.unit}")
    if args.unit == 'case':
        print(f"  Num cases: {args.num_cases or 'all'}")
        print(f"  Max slices/case: {args.max_slices_per_case or 'all'}")
    if args.debug:
        print("[DEBUG MODE]")
    print("=" * 70)
    
    # Create model
    print("\n[Model] Creating model...")
    model, expected_cond_channels = create_model(config, args.model_type, args.cond, device)
    
    # Load checkpoint
    print("\n[Checkpoint] Loading weights...")
    load_checkpoint_weights(model, args.checkpoint, device)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] Total parameters: {total_params:,}")
    print(f"[Model] Expected condition channels: {expected_cond_channels}")
    
    # Create diffusion
    print("\n[Diffusion] Creating diffusion process...")
    diffusion = GaussianDiffusion(
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
        beta_start=config['diffusion'].get('beta_start', 0.0001),
        beta_end=config['diffusion'].get('beta_end', 0.02),
        device=device
    )
    
    # Create sampler
    sampler = ConditionalSampler(diffusion, model, args.model_type, device)
    
    # =========================================================================
    # Data Consistency (DC) Setup (Optional) - Using Physical PET Simulation
    # =========================================================================
    dc_enabled = args.dc_enable or config.get('dc', {}).get('enable', False)
    dc_wrapper = None
    dc_forward_model = None
    dc_config = None  # Will be set if DC is enabled
    
    if dc_enabled:
        print("\n[DC] Data Consistency enabled (Physical PET Simulation)")
        
        # Build DC config: CLI > Config file > Hardcoded defaults
        # Helper function for priority resolution
        def get_dc_param(cli_val, cfg_key, default):
            """CLI > Config > Default"""
            if cli_val is not None:
                return cli_val
            return dc_cfg.get(cfg_key, default)
        
        dc_cfg = config.get('dc', {})
        dc_config = {
            # DC algorithm parameters
            'enable': True,  # Explicitly enable since we're in this block
            'loss_type': get_dc_param(args.dc_loss_type, 'loss_type', 'l2'),
            'step_size': get_dc_param(args.dc_step_size, 'step_size', 0.1),
            'step_size_schedule': dc_cfg.get('step_size_schedule', 'constant'),
            'step_size_start': dc_cfg.get('step_size_start', 0.1),
            'step_size_end': dc_cfg.get('step_size_end', 0.01),
            'num_inner_steps': get_dc_param(args.dc_num_inner, 'num_inner_steps', 1),
            'start_step': get_dc_param(args.dc_start_step, 'start_step', 0),
            'end_step': get_dc_param(args.dc_end_step, 'end_step', None),
            # Physical simulation parameters
            'fov_mm': get_dc_param(args.dc_fov_mm, 'fov_mm', None),  # None = auto from affine
            'fwhm_mm': get_dc_param(args.dc_fwhm_mm, 'fwhm_mm', 6.0),
            'dose_alpha': get_dc_param(args.dc_dose_alpha, 'dose_alpha', 0.1),
            'num_angles': get_dc_param(args.dc_num_angles, 'num_angles', 180),
            'rebin_factor': get_dc_param(args.dc_rebin_factor, 'rebin_factor', 1),
            'rebin_radial_factor': get_dc_param(args.dc_rebin_radial_factor, 'rebin_radial_factor', 1),
            'background_beta': get_dc_param(args.dc_background_beta, 'background_beta', 0.0),
            # PET normalization parameters (must match read_data.py)
            'use_arcsinh_norm': True,
            'pet_max_mbqml': PET_MAX_MBQML,
            'pet_scale_mbqml': PET_SCALE_MBQML,
            # Route 1: DC operates on x0_pred; forward model handles resolution
            'apply_on': 'x0_pred',
            # Other DC parameters
            'clamp_min': dc_cfg.get('clamp_min', 0.0),
            'clamp_max': dc_cfg.get('clamp_max', None),
        }
        
        # Handle end_step: None means until end (use infinity for comparison)
        if dc_config['end_step'] is None:
            dc_config['end_step'] = float('inf')
        
        # Print effective config with source indication
        print(f"[DC] Effective config (CLI > Config > Default):")
        print(f"     loss_type={dc_config['loss_type']}, step_size={dc_config['step_size']}, "
              f"inner_steps={dc_config['num_inner_steps']}")
        print(f"     Physical: fwhm={dc_config['fwhm_mm']}mm, dose={dc_config['dose_alpha']}, "
              f"angles={dc_config['num_angles']}, rebin={dc_config['rebin_factor']}, "
              f"rebin_radial={dc_config['rebin_radial_factor']}")
    
    
    # =========================================================================
    # Branch by output unit: image vs case
    # =========================================================================
    if args.unit == 'image':
        # Original per-image mode
        _run_image_mode(args, config, sampler, out_dir, device, expected_cond_channels,
                        method, ddim_steps, eta, data_range, num_samples,
                        ode_solver, ode_steps, ode_t_start, ode_t_end, ode_eps,
                        dc_enabled=dc_enabled, dc_config=dc_config if dc_enabled else None)
    else:
        # New per-case mode
        _run_case_mode(args, config, sampler, out_dir, device, expected_cond_channels,
                       method, ddim_steps, eta, data_range,
                       ode_solver, ode_steps, ode_t_start, ode_t_end, ode_eps,
                       dc_enabled=dc_enabled, dc_config=dc_config if dc_enabled else None)


def _run_image_mode(args, config, sampler, out_dir, device, expected_cond_channels,
                    method, ddim_steps, eta, data_range, num_samples,
                    ode_solver, ode_steps, ode_t_start, ode_t_end, ode_eps,
                    dc_enabled=False, dc_config=None):
    """Run sampling in per-image mode (original behavior), with optional Physical DC."""
    
    # =========================================================================
    # DC Setup (if enabled) - Using Physical PET Simulation
    # =========================================================================
    dc_wrapper = None
    dc_forward_model = None
    dc_simulator = None
    dc_forward_model = None
    
    # Create dataloader
    print("\n[Data] Loading dataset (image mode)...")
    dataset = PairedImageFolders(
        data_path=config['data']['data_path'],
        list_path=config['data']['list_path'],
        phase=args.phase,
        df=config['data'].get('df', None),
        mini_data=args.debug  # Use mini data in debug mode
    )
    
    # Limit samples (with optional shuffle)
    if num_samples < len(dataset):
        if args.shuffle:
            import random
            indices = list(range(len(dataset.paired_files)))
            random.shuffle(indices)
            dataset.paired_files = [dataset.paired_files[i] for i in indices[:num_samples]]
            print(f"[Data] Randomly selected {num_samples} samples")
        else:
            dataset.paired_files = dataset.paired_files[:num_samples]
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one at a time for clean output
        shuffle=False,
        num_workers=0 if args.debug else 4
    )
    
    print(f"[Data] Dataset size: {len(dataset)} samples")
    
    # Metrics storage
    all_metrics = []
    
    # Process samples
    print("\n[Sampling] Starting generation (image mode)...")
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Generating")):
        # Unpack batch
        pet, pet_affine, ct, ct_affine, keys = batch_data
        key = keys[0] if isinstance(keys, (list, tuple)) else keys
        
        # Move to device
        pet = pet.to(device)
        ct = ct.to(device)
        
        # Convert affines to numpy
        pet_affine_np = pet_affine.numpy()
        ct_affine_np = ct_affine.numpy()
        
        # Align CT to PET space
        pet_h, pet_w = pet.shape[2], pet.shape[3]
        ct_in_pet_space = align_ct_to_pet_batch(
            ct_batch=ct,
            ct_affines=ct_affine_np,
            pet_affines=pet_affine_np,
            pet_shape=(pet_h, pet_w),
            device=device,
            mode='bilinear'
        ).float()
        
        # =====================================================================
        # DC: Initialize simulator on first batch (need image size and FOV)
        # Must happen BEFORE LR PET creation so we can use physical simulation
        # =====================================================================
        if dc_enabled and dc_config is not None and dc_simulator is None:
            from utils.physical_pet_simulation import PhysicalPETSimulator
            from utils.physical_forward_model import PhysicalPETForwardModel
            from utils.dc_sampler_wrapper import DCSamplerWrapper
            
            image_size = pet_h
            fov_mm = dc_config.get('fov_mm') or extract_fov_from_affine(pet_affine_np, image_size)
            dc_config['fov_mm'] = fov_mm  # Store for consistency
            
            dc_simulator = PhysicalPETSimulator(
                fov_mm=fov_mm,
                hr_size=image_size,
                fwhm_mm=dc_config['fwhm_mm'],
                num_angles=dc_config['num_angles'],
                device=device
            )
            print(f"\n[DC] Simulator initialized: FOV={fov_mm:.1f}mm, size={image_size}, "
                  f"FWHM={dc_config['fwhm_mm']}mm, angles={dc_config['num_angles']}")
        
        # =====================================================================
        # DC: Generate y_obs (noisy measurement) for this sample
        # CRITICAL: y_obs must be generated in PHYSICAL SPACE (MBq/mL)
        # so that DC gradient is consistent with physical forward model
        # =====================================================================
        y_obs = None
        lr_pet_for_dc = None  # Physical LR PET (in physical space)
        lr_pet_for_dc_model = None  # Physical LR PET converted to model space
        if dc_enabled and dc_simulator is not None:
            # Convert GT PET from model space to physical space BEFORE simulation
            pet_physical = model_to_physical(pet)
            
            # Degrade PET in physical space: returns (y_obs_noisy, degraded_image, meta)
            y_obs, lr_pet_for_dc, dc_meta = dc_simulator.degrade_pet(
                pet_physical,  # Use physical space PET
                dose_alpha=dc_config['dose_alpha'],
                rebin_factor=dc_config['rebin_factor'],
                rebin_radial_factor=dc_config['rebin_radial_factor'],
                background_beta=dc_config['background_beta'],
                apply_psf=(dc_config['fwhm_mm'] > 0),
                return_sinogram=True
            )
            
            # Convert LR PET back to model space for conditioning
            lr_pet_for_dc_model = physical_to_model(lr_pet_for_dc)
            
            # Create forward model from meta (ensures exact consistency)
            if dc_forward_model is None:
                dc_forward_model = PhysicalPETForwardModel(dc_meta, device=device)
            
            if batch_idx == 0 and args.debug:
                print(f"[DC] GT PET physical: min={pet_physical.min():.6f}, max={pet_physical.max():.6f}")
                print(f"[DC] y_obs shape={y_obs.shape}, range=[{y_obs.min():.4f}, {y_obs.max():.4f}]")
                print(f"[DC] LR PET (physical): min={lr_pet_for_dc.min():.6f}, max={lr_pet_for_dc.max():.6f}")
                print(f"[DC] LR PET (model): min={lr_pet_for_dc_model.min():.4f}, max={lr_pet_for_dc_model.max():.4f}")
        
        # =====================================================================
        # Create LR PET for ct+lrpet mode
        # Priority: Use DC physical LR if available, else simple downsampling
        # 
        # Note on terminology:
        # - Simple downsampling: lr is actual low-res (e.g., 100x100), lr_up is upsampled back (400x400)
        # - DC physical: lr_up is the degraded HR image (400x400, same size as original)
        #                lr is also 400x400 for DC (no actual resolution reduction, just degradation)
        # =====================================================================
        lr, lr_up = None, None
        if args.cond == 'ct+lrpet':
            if dc_enabled and lr_pet_for_dc_model is not None:
                # Use physically simulated degraded PET (PSF blur + Poisson noise + FBP recon)
                # Note: DC degradation does NOT reduce spatial resolution, it's the same size as HR
                # Both lr and lr_up are 400x400 (degraded quality, not reduced resolution)
                lr_up = lr_pet_for_dc_model  # [B, 1, H, W] - for blending into noisy input
                lr = lr_pet_for_dc_model.clone()  # [B, 1, H, W] - same size for consistency
                if batch_idx == 0:
                    print(f"[LR PET] Using DC physical degradation (FWHM={dc_config['fwhm_mm']}mm, "
                          f"dose={dc_config['dose_alpha']}, size={lr.shape[-2]}x{lr.shape[-1]})")
            else:
                # Fallback: simple area downsampling (true low resolution)
                # lr is actual low-res (e.g., 100x100), lr_up is upsampled back (400x400)
                lr, lr_up = create_lr_pet(pet, lr_factor=args.lr_factor, mode='area',
                                          upsample_mode=args.lr_upsample)
                if batch_idx == 0:
                    print(f"[LR PET] Using simple downsampling (factor={args.lr_factor}, "
                          f"lr_size={lr.shape[-2]}x{lr.shape[-1]})")
        
        # Debug: print shapes on first batch
        if args.debug and batch_idx == 0:
            print(f"\n[DEBUG] Shape check (first batch):")
            print(f"  PET shape: {pet.shape}")
            print(f"  CT shape: {ct.shape}")
            print(f"  CT in PET space shape: {ct_in_pet_space.shape}")
            if lr is not None:
                print(f"  LR PET shape: {lr.shape}")
                print(f"  LR PET (upsampled) shape: {lr_up.shape}")
            print(f"\n[DEBUG] Value ranges:")
            print(f"  PET: [{pet.min():.4f}, {pet.max():.4f}]")
            print(f"  CT: [{ct.min():.4f}, {ct.max():.4f}]")
            print(f"  CT in PET space: [{ct_in_pet_space.min():.4f}, {ct_in_pet_space.max():.4f}]")
            if lr is not None:
                print(f"  LR PET: [{lr.min():.4f}, {lr.max():.4f}]")
                print(f"  LR PET (upsampled): [{lr_up.min():.4f}, {lr_up.max():.4f}]")
        
        # Build condition tensor
        condition = ct_in_pet_space
        
        # =====================================================================
        # Create DC wrapper for this sample
        # =====================================================================
        active_sampler = sampler
        if dc_enabled and y_obs is not None:
            from utils.dc_sampler_wrapper import DCSamplerWrapper
            active_sampler = DCSamplerWrapper(
                base_sampler=sampler,
                forward_model=dc_forward_model,
                y_obs=y_obs,
                dc_config=dc_config
            )
        
        # Generate sample
        lr_guidance = lr_up if args.cond == 'ct+lrpet' else None
        pred_pet = active_sampler.sample(
            condition,
            shape=(1, 1, pet_h, pet_w),
            method=method,
            ddim_steps=ddim_steps,
            eta=eta,
            lr_guidance=lr_guidance,
            lr_blend=args.lr_blend if args.cond == 'ct+lrpet' else 0.0,
            ode_solver=ode_solver,
            ode_steps=ode_steps,
            ode_t_start=ode_t_start,
            ode_t_end=ode_t_end,
            ode_eps=ode_eps
        )
        
        # Post-processing: ensure non-negative values for PET
        pred_pet = torch.clamp(pred_pet, 0, None)
        
        # Calculate metrics
        mse_val = calculate_mse(pred_pet, pet)
        psnr_val = calculate_psnr(pred_pet, pet, data_range=data_range)
        ssim_val = calculate_ssim(pred_pet, pet, data_range=data_range)
        
        metrics = {
            'key': str(key),
            'mse': float(mse_val),
            'psnr': float(psnr_val),
            'ssim': float(ssim_val)
        }
        all_metrics.append(metrics)
        
        # Save outputs
        sample_data = {
            'ct_in_pet_space': ct_in_pet_space.cpu(),
            'gt_pet': pet.cpu(),
            'pred_pet': pred_pet.cpu(),
            'key': str(key)
        }
        if lr is not None:
            sample_data['lr'] = lr.cpu()
            sample_data['lr_up'] = lr_up.cpu()
        # Add DC degraded LR PET if available
        if lr_pet_for_dc is not None:
            sample_data['dc_lr_pet'] = lr_pet_for_dc.cpu()
        
        save_sample_outputs(sample_data, out_dir, key)
        
        # Save DC simulation outputs if enabled (using Physical Simulator)
        if dc_enabled and args.dc_save_simulation and y_obs is not None and dc_simulator is not None:
            if batch_idx % args.dc_save_every == 0:
                from utils.dc_sampler_wrapper import save_simulation_outputs
                # Generate sinogram from predicted PET for comparison
                # Using Physical PET Simulator's projector
                pred_sinogram = dc_simulator._projector_hr.forward_project(
                    pred_pet, 
                    apply_psf=(dc_config.get('fwhm_mm', 4.0) > 0),
                    apply_rebin=(dc_config.get('rebin_factor', 1) > 1)
                )
                save_simulation_outputs(
                    output_dir=out_dir,
                    sample_key=str(key),
                    pet_image=pred_pet,
                    sinogram=pred_sinogram,
                    degraded_pet=lr_pet_for_dc
                )
        
        if args.debug:
            print(f"\n[Sample {batch_idx}] Key: {key}")
            print(f"  MSE:  {mse_val:.6f}")
            print(f"  PSNR: {psnr_val:.2f} dB")
            print(f"  SSIM: {ssim_val:.4f}")
    
    # Calculate and save aggregate metrics
    avg_mse = np.mean([m['mse'] for m in all_metrics])
    avg_psnr = np.mean([m['psnr'] for m in all_metrics])
    avg_ssim = np.mean([m['ssim'] for m in all_metrics])
    
    print("\n" + "=" * 70)
    print("Sampling Results (Image Mode)")
    print("=" * 70)
    print(f"Total samples: {len(all_metrics)}")
    print(f"Average MSE:  {avg_mse:.6f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    # Save metrics to CSV
    csv_path = os.path.join(out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['key', 'mse', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nMetrics saved to: {csv_path}")
    
    # Save metrics to JSON (with summary)
    json_path = os.path.join(out_dir, 'metrics.json')
    summary = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'model_type': args.model_type,
        'cond_mode': args.cond,
        'lr_factor': args.lr_factor if args.cond == 'ct+lrpet' else None,
        'lr_upsample': args.lr_upsample if args.cond == 'ct+lrpet' else None,
        'lr_blend': args.lr_blend if args.cond == 'ct+lrpet' else None,
        'sampler': method,
        'seed': args.seed,
        'data_range': data_range,
        'num_samples': len(all_metrics),
        'unit': 'image',
        # DC settings
        'dc_enabled': dc_enabled,
    }
    
    # Add DC-specific parameters if enabled
    if dc_enabled and dc_config is not None:
        summary['dc_config'] = {
            # DC algorithm parameters
            'enable': dc_config.get('enable'),
            'loss_type': dc_config.get('loss_type'),
            'step_size': dc_config.get('step_size'),
            'step_size_schedule': dc_config.get('step_size_schedule'),
            'step_size_start': dc_config.get('step_size_start'),
            'step_size_end': dc_config.get('step_size_end'),
            'num_inner_steps': dc_config.get('num_inner_steps'),
            'start_step': dc_config.get('start_step'),
            'end_step': dc_config.get('end_step'),
            # Physical simulation parameters
            'fov_mm': dc_config.get('fov_mm'),
            'fwhm_mm': dc_config.get('fwhm_mm'),
            'dose_alpha': dc_config.get('dose_alpha'),
            'num_angles': dc_config.get('num_angles'),
            'rebin_factor': dc_config.get('rebin_factor'),
            'rebin_radial_factor': dc_config.get('rebin_radial_factor'),
            'background_beta': dc_config.get('background_beta'),
            # PET normalization parameters
            'use_arcsinh_norm': dc_config.get('use_arcsinh_norm'),
            'pet_max_mbqml': dc_config.get('pet_max_mbqml'),
            'pet_scale_mbqml': dc_config.get('pet_scale_mbqml'),
            # Route 1 configuration
            'apply_on': dc_config.get('apply_on'),
            # Other DC parameters
            'clamp_min': dc_config.get('clamp_min'),
            'clamp_max': dc_config.get('clamp_max'),
        }
    
    # Add sampler-specific parameters
    if method == 'ddim':
        summary['ddim_steps'] = ddim_steps
        summary['eta'] = eta
    elif method == 'ode':
        summary['ode_solver'] = ode_solver
        summary['ode_steps'] = ode_steps
        summary['ode_t_start'] = ode_t_start
        summary['ode_t_end'] = ode_t_end
        summary['ode_eps'] = ode_eps
    
    summary['average_metrics'] = {
        'mse': avg_mse,
        'psnr': avg_psnr,
        'ssim': avg_ssim
    }
    summary['per_sample_metrics'] = all_metrics
    
    # Sanitize for JSON serialization
    summary = sanitize_for_json(summary)
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {json_path}")
    
    print(f"\nOutput directory: {out_dir}")
    print("=" * 70)
    print("Sampling completed!")
    print("=" * 70)


def _run_case_mode(args, config, sampler, out_dir, device, expected_cond_channels,
                   method, ddim_steps, eta, data_range,
                   ode_solver, ode_steps, ode_t_start, ode_t_end, ode_eps,
                   dc_enabled=False, dc_config=None):
    """
    Run sampling in per-case mode, with optional Physical DC.
    
    Organizes output by case (patient), with subdirectories for each modality.
    """
    # =========================================================================
    # DC Setup (if enabled) - Using Physical PET Simulation
    # =========================================================================
    dc_wrapper = None
    dc_forward_model = None
    dc_simulator = None
    dc_meta = None
    _dc_initialized = False
    
    if dc_enabled and dc_config is not None:
        from utils.physical_pet_simulation import PhysicalPETSimulator, PhysicalDegradationMeta
        from utils.physical_forward_model import PhysicalPETForwardModel
        from utils.dc_sampler_wrapper import DCSamplerWrapper, save_simulation_outputs
    
    # Create case-based dataset
    print("\n[Data] Loading dataset (case mode)...")
    
    max_slices = args.max_slices_per_case
    if args.debug and max_slices is None:
        max_slices = 3  # Limit slices per case in debug mode
    
    dataset = PairedImageFoldersByCase(
        data_path=config['data']['data_path'],
        list_path=config['data']['list_path'],
        phase=args.phase,
        df=config['data'].get('df', None),
        max_slices_per_case=max_slices,
        mini_data=args.debug
    )
    
    # Limit number of cases
    num_cases = args.num_cases
    if args.debug and num_cases is None:
        num_cases = 2  # Only 2 cases in debug mode
    
    if num_cases is not None and num_cases < len(dataset):
        dataset.cases = dataset.cases[:num_cases]
    
    print(f"[Data] Number of cases: {len(dataset)}")
    total_slices = sum(len(c['slices']) for c in dataset.cases)
    print(f"[Data] Total slices: {total_slices}")
    
    # Storage for global metrics
    all_case_metrics = []  # Per-case summaries
    all_slice_metrics = []  # All slices across all cases
    
    # Process each case
    print("\n[Sampling] Starting generation (case mode)...")
    for case_idx in tqdm(range(len(dataset)), desc="Processing cases"):
        case_data = dataset[case_idx]
        case_id = case_data['case_id']
        num_slices = case_data['num_slices']
        slices = case_data['slices']
        
        if num_slices == 0:
            print(f"\n[Warning] Case {case_id} has no valid slices, skipping...")
            continue
        
        # Create case directory
        case_dir = os.path.join(out_dir, f"case_{case_id}")
        os.makedirs(case_dir, exist_ok=True)
        
        # Debug info
        if args.debug:
            print(f"\n[DEBUG] Case {case_idx}: {case_id}")
            print(f"  Number of slices: {num_slices}")
            if slices:
                first_slice = slices[0]
                print(f"  First slice shape: PET={first_slice['pet'].shape}, CT={first_slice['ct'].shape}")
                print(f"  PET range: [{first_slice['pet'].min():.4f}, {first_slice['pet'].max():.4f}]")
        
        # Process slices for this case
        case_slice_data = []
        case_slice_metrics = []
        
        for slice_idx, slice_info in enumerate(tqdm(slices, desc=f"Case {case_id}", leave=False)):
            key = slice_info['key']
            
            # Convert numpy to tensor and add batch dimension
            pet = torch.from_numpy(slice_info['pet']).unsqueeze(0).to(device)  # [1, 1, H, W]
            ct = torch.from_numpy(slice_info['ct']).unsqueeze(0).to(device)
            pet_affine = slice_info['pet_affine']
            ct_affine = slice_info['ct_affine']
            
            # Align CT to PET space
            pet_h, pet_w = pet.shape[2], pet.shape[3]
            ct_in_pet_space = align_ct_to_pet_batch(
                ct_batch=ct,
                ct_affines=np.expand_dims(ct_affine, 0),
                pet_affines=np.expand_dims(pet_affine, 0),
                pet_shape=(pet_h, pet_w),
                device=device,
                mode='bilinear'
            ).float()
            
            # =================================================================
            # DC: Initialize simulator on first slice
            # Must happen BEFORE LR PET creation so we can use physical simulation
            # =================================================================
            if dc_enabled and dc_config is not None and not _dc_initialized:
                from utils.physical_pet_simulation import PhysicalPETSimulator
                from utils.physical_forward_model import PhysicalPETForwardModel
                from utils.dc_sampler_wrapper import DCSamplerWrapper
                
                image_size = pet_h
                fov_mm = dc_config.get('fov_mm') or extract_fov_from_affine(pet_affine, image_size)
                dc_config['fov_mm'] = fov_mm
                
                dc_simulator = PhysicalPETSimulator(
                    fov_mm=fov_mm,
                    hr_size=image_size,
                    fwhm_mm=dc_config['fwhm_mm'],
                    num_angles=dc_config['num_angles'],
                    device=device
                )
                _dc_initialized = True
                if args.debug:
                    print(f"\n[DC] Simulator initialized: FOV={fov_mm:.1f}mm, FWHM={dc_config['fwhm_mm']}mm")
            
            # =================================================================
            # DC: Generate y_obs (noisy measurement) in PHYSICAL SPACE
            # =================================================================
            y_obs = None
            lr_pet_for_dc = None  # Physical LR PET (in physical space)
            lr_pet_for_dc_model = None  # Physical LR PET converted to model space
            active_sampler = sampler
            if dc_enabled and _dc_initialized:
                # Convert GT PET to physical space BEFORE simulation
                pet_physical = model_to_physical(pet)
                
                y_obs, lr_pet_for_dc, dc_meta = dc_simulator.degrade_pet(
                    pet_physical,  # Use physical space PET
                    dose_alpha=dc_config['dose_alpha'],
                    rebin_factor=dc_config['rebin_factor'],
                    rebin_radial_factor=dc_config['rebin_radial_factor'],
                    background_beta=dc_config['background_beta'],
                    apply_psf=(dc_config['fwhm_mm'] > 0),
                    return_sinogram=True
                )
                
                # Convert LR PET back to model space for conditioning
                lr_pet_for_dc_model = physical_to_model(lr_pet_for_dc)
                
                if dc_forward_model is None:
                    dc_forward_model = PhysicalPETForwardModel(dc_meta, device=device)
                
                active_sampler = DCSamplerWrapper(
                    base_sampler=sampler,
                    forward_model=dc_forward_model,
                    y_obs=y_obs,
                    dc_config=dc_config
                )
            
            # =================================================================
            # Create LR PET for ct+lrpet mode
            # Priority: Use DC physical LR if available, else simple downsampling
            # =================================================================
            lr, lr_up = None, None
            if args.cond == 'ct+lrpet':
                if dc_enabled and lr_pet_for_dc_model is not None:
                    # Use physically simulated degraded PET (PSF blur + Poisson noise + FBP recon)
                    # Note: DC degradation does NOT reduce spatial resolution
                    lr_up = lr_pet_for_dc_model  # [B, 1, H, W]
                    lr = lr_pet_for_dc_model.clone()  # Same size for consistency
                else:
                    # Fallback: simple area downsampling
                    lr, lr_up = create_lr_pet(pet, lr_factor=args.lr_factor, mode='area',
                                              upsample_mode=args.lr_upsample)
            
            # Build condition tensor
            condition = ct_in_pet_space
            
            # Generate sample
            lr_guidance = lr_up if args.cond == 'ct+lrpet' else None
            pred_pet = active_sampler.sample(
                condition,
                shape=(1, 1, pet_h, pet_w),
                method=method,
                ddim_steps=ddim_steps,
                eta=eta,
                lr_guidance=lr_guidance,
                lr_blend=args.lr_blend if args.cond == 'ct+lrpet' else 0.0,
                ode_solver=ode_solver,
                ode_steps=ode_steps,
                ode_t_start=ode_t_start,
                ode_t_end=ode_t_end,
                ode_eps=ode_eps
            )
            
            # Post-processing
            pred_pet = torch.clamp(pred_pet, 0, None)
            
            # Calculate metrics
            mse_val = calculate_mse(pred_pet, pet)
            psnr_val = calculate_psnr(pred_pet, pet, data_range=data_range)
            ssim_val = calculate_ssim(pred_pet, pet, data_range=data_range)
            
            metrics = {
                'key': str(key),
                'case_id': case_id,
                'mse': float(mse_val),
                'psnr': float(psnr_val),
                'ssim': float(ssim_val)
            }
            case_slice_metrics.append(metrics)
            all_slice_metrics.append(metrics)
            
            # Collect slice data for saving
            slice_output = {
                'ct_in_pet_space': ct_in_pet_space.cpu(),
                'gt_pet': pet.cpu(),
                'pred_pet': pred_pet.cpu(),
                'key': str(key)
            }
            if lr is not None:
                slice_output['lr'] = lr.cpu()
                slice_output['lr_up'] = lr_up.cpu()
            case_slice_data.append(slice_output)
        
        # Save case outputs
        save_case_outputs(case_dir, case_slice_data, case_slice_metrics, save_npy=True)
        
        # Create case visualization (batched into multiple images)
        create_case_visualization(case_dir, case_slice_data, slices_per_image=10)
        
        # Calculate case-level metrics
        if case_slice_metrics:
            case_avg_mse = np.mean([m['mse'] for m in case_slice_metrics])
            case_avg_psnr = np.mean([m['psnr'] for m in case_slice_metrics])
            case_avg_ssim = np.mean([m['ssim'] for m in case_slice_metrics])
            case_std_psnr = np.std([m['psnr'] for m in case_slice_metrics])
            case_std_ssim = np.std([m['ssim'] for m in case_slice_metrics])
            
            case_summary = {
                'case_id': case_id,
                'num_slices': len(case_slice_metrics),
                'avg_mse': case_avg_mse,
                'avg_psnr': case_avg_psnr,
                'avg_ssim': case_avg_ssim,
                'std_psnr': case_std_psnr,
                'std_ssim': case_std_ssim
            }
            all_case_metrics.append(case_summary)
            
            if args.debug:
                print(f"\n[Case {case_id}] {len(case_slice_metrics)} slices")
                print(f"  Avg MSE:  {case_avg_mse:.6f}")
                print(f"  Avg PSNR: {case_avg_psnr:.2f} ± {case_std_psnr:.2f} dB")
                print(f"  Avg SSIM: {case_avg_ssim:.4f} ± {case_std_ssim:.4f}")
    
    # =========================================================================
    # Save global metrics
    # =========================================================================
    
    # Global case-level summary
    if all_case_metrics:
        global_avg_psnr = np.mean([c['avg_psnr'] for c in all_case_metrics])
        global_avg_ssim = np.mean([c['avg_ssim'] for c in all_case_metrics])
        global_avg_mse = np.mean([c['avg_mse'] for c in all_case_metrics])
    else:
        global_avg_psnr = global_avg_ssim = global_avg_mse = 0.0
    
    print("\n" + "=" * 70)
    print("Sampling Results (Case Mode)")
    print("=" * 70)
    print(f"Total cases: {len(all_case_metrics)}")
    print(f"Total slices: {len(all_slice_metrics)}")
    print(f"Global Average MSE:  {global_avg_mse:.6f}")
    print(f"Global Average PSNR: {global_avg_psnr:.2f} dB")
    print(f"Global Average SSIM: {global_avg_ssim:.4f}")
    
    # Save per-case metrics CSV (metrics_cases.csv)
    cases_csv_path = os.path.join(out_dir, 'metrics_cases.csv')
    with open(cases_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'case_id', 'num_slices', 'avg_mse', 'avg_psnr', 'avg_ssim', 'std_psnr', 'std_ssim'
        ])
        writer.writeheader()
        writer.writerows(all_case_metrics)
    print(f"\nCase metrics saved to: {cases_csv_path}")
    
    # Save all slices metrics CSV (metrics_all_slices.csv)
    slices_csv_path = os.path.join(out_dir, 'metrics_all_slices.csv')
    with open(slices_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'key', 'mse', 'psnr', 'ssim'])
        writer.writeheader()
        for m in all_slice_metrics:
            writer.writerow({
                'case_id': m['case_id'],
                'key': m['key'],
                'mse': m['mse'],
                'psnr': m['psnr'],
                'ssim': m['ssim']
            })
    print(f"Slice metrics saved to: {slices_csv_path}")
    
    # Save global summary JSON
    json_path = os.path.join(out_dir, 'metrics.json')
    summary = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'model_type': args.model_type,
        'cond_mode': args.cond,
        'lr_factor': args.lr_factor if args.cond == 'ct+lrpet' else None,
        'lr_upsample': args.lr_upsample if args.cond == 'ct+lrpet' else None,
        'lr_blend': args.lr_blend if args.cond == 'ct+lrpet' else None,
        'sampler': method,
        'seed': args.seed,
        'data_range': data_range,
        'unit': 'case',
        'num_cases': len(all_case_metrics),
        'num_slices': len(all_slice_metrics),
        'max_slices_per_case': args.max_slices_per_case,
        # DC settings
        'dc_enabled': dc_enabled,
    }
    
    # Add DC-specific parameters if enabled
    if dc_enabled and dc_config is not None:
        summary['dc_config'] = {
            # DC algorithm parameters
            'enable': dc_config.get('enable'),
            'loss_type': dc_config.get('loss_type'),
            'step_size': dc_config.get('step_size'),
            'step_size_schedule': dc_config.get('step_size_schedule'),
            'step_size_start': dc_config.get('step_size_start'),
            'step_size_end': dc_config.get('step_size_end'),
            'num_inner_steps': dc_config.get('num_inner_steps'),
            'start_step': dc_config.get('start_step'),
            'end_step': dc_config.get('end_step'),
            # Physical simulation parameters
            'fov_mm': dc_config.get('fov_mm'),
            'fwhm_mm': dc_config.get('fwhm_mm'),
            'dose_alpha': dc_config.get('dose_alpha'),
            'num_angles': dc_config.get('num_angles'),
            'rebin_factor': dc_config.get('rebin_factor'),
            'rebin_radial_factor': dc_config.get('rebin_radial_factor'),
            'background_beta': dc_config.get('background_beta'),
            # PET normalization parameters
            'use_arcsinh_norm': dc_config.get('use_arcsinh_norm'),
            'pet_max_mbqml': dc_config.get('pet_max_mbqml'),
            'pet_scale_mbqml': dc_config.get('pet_scale_mbqml'),
            # Route 1 configuration
            'apply_on': dc_config.get('apply_on'),
            # Other DC parameters
            'clamp_min': dc_config.get('clamp_min'),
            'clamp_max': dc_config.get('clamp_max'),
        }
    
    # Add sampler-specific parameters
    if method == 'ddim':
        summary['ddim_steps'] = ddim_steps
        summary['eta'] = eta
    elif method == 'ode':
        summary['ode_solver'] = ode_solver
        summary['ode_steps'] = ode_steps
        summary['ode_t_start'] = ode_t_start
        summary['ode_t_end'] = ode_t_end
        summary['ode_eps'] = ode_eps
    
    summary['global_metrics'] = {
        'avg_mse': global_avg_mse,
        'avg_psnr': global_avg_psnr,
        'avg_ssim': global_avg_ssim
    }
    summary['per_case_metrics'] = all_case_metrics
    
    # Sanitize for JSON serialization
    summary = sanitize_for_json(summary)
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {json_path}")
    
    print(f"\nOutput directory: {out_dir}")
    print("=" * 70)
    print("Sampling completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()