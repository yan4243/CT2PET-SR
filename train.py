"""
CT to PET Conditional Generation Training Script (2D, Cross-Attention Condition).

This script trains a conditional diffusion model that generates 2D PET images 
from 2D CT images. The CT conditions the model via cross-attention mechanism.

Model Input: x_noisy [B, 1, H, W], context: ct_in_pet_space [B, 1, H, W]
Model Output: noise prediction (1 channel)

Features:
- DDP multi-GPU training support
- EMA for model weights
- Pretrained weight loading from LDPET pretraining (partial)
- TensorBoard + WandB logging
- Checkpoint save/resume with optimizer, scheduler, EMA
- Debug mode for quick validation
- 2D metrics: MSE, PSNR, SSIM

Usage:
    Single GPU:
        python train_ct2pet_crossattn_2d.py --config configs/ct2pet_crossattn_2d.yaml --debug
    
    Multi-GPU (DDP):
        torchrun --nproc_per_node=4 train_ct2pet_crossattn_2d.py --config configs/ct2pet_crossattn_2d.yaml
    
    With custom pretrained weights:
        python train_ct2pet_crossattn_2d.py --config configs/ct2pet_crossattn_2d.yaml --pretrained path/to/ckpt.pth
    
    Without pretrained weights:
        python train_ct2pet_crossattn_2d.py --config configs/ct2pet_crossattn_2d.yaml --pretrained none
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import local modules
from models import UNetModelCrossAttn, GaussianDiffusion
from datasets.PET_CT_Datasets import PairedImageFolders
from utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    get_cosine_schedule_with_warmup,
    AverageMeter,
    EMA,
    calculate_psnr,
    calculate_ssim,
    calculate_mse,
    align_ct_to_pet_batch
)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        num_gpus = torch.cuda.device_count()
        if local_rank >= num_gpus:
            raise RuntimeError(f"LOCAL_RANK={local_rank} invalid. Only {num_gpus} GPUs available.")
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def create_dataloaders(config, rank=0, world_size=1, debug=False, debug_samples=None):
    """Create train and validation dataloaders with DDP support."""
    
    # Build kwargs
    kwargs = {
        'mini_data': config['data'].get('mini_data', False),
        'mid_data': config['data'].get('mid_data', False)
    }
    
    # Override for debug mode
    if debug:
        kwargs['mini_data'] = True  # Use mini dataset in debug mode
    
    # Training dataset
    train_dataset = PairedImageFolders(
        data_path=config['data']['data_path'],
        list_path=config['data']['list_path'],
        phase='train',
        df=config['data'].get('df', None),
        **kwargs
    )
    
    # Validation dataset
    val_dataset = PairedImageFolders(
        data_path=config['data']['data_path'],
        list_path=config['data']['list_path'],
        phase='val',
        df=config['data'].get('df', None),
        **kwargs
    )
    
    # Limit dataset size for debug mode
    if debug and debug_samples:
        train_dataset.paired_files = train_dataset.paired_files[:debug_samples]
        val_dataset.paired_files = val_dataset.paired_files[:max(debug_samples // 5, 4)]
    
    # Create samplers for DDP
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Debug mode: reduce workers
    num_workers = 0 if debug else config['data']['num_workers']
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=config['data']['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=config['data']['pin_memory']
    )
    
    return train_loader, val_loader, train_sampler, val_sampler


def load_pretrained_weights(unet, pretrained_path, device, is_main=True):
    """
    Load pretrained UNet weights for cross-attention model.
    
    The cross-attention UNet has in_channels=1 (same as pretrained), but has additional
    cross-attention and condition encoder layers that will be randomly initialized.
    
    Args:
        unet: UNetModelCrossAttn
        pretrained_path: Path to pretrained checkpoint
        device: Device to load weights on
        is_main: Whether this is the main process (for logging)
    
    Returns:
        True if loaded successfully, False otherwise
    """
    if pretrained_path is None or pretrained_path.lower() == 'none':
        if is_main:
            print("[Pretrained] Skipping pretrained weight loading (--pretrained none)")
        return False
    
    if not os.path.exists(pretrained_path):
        if is_main:
            print(f"[Pretrained] WARNING: Checkpoint not found: {pretrained_path}")
        return False
    
    if is_main:
        print(f"[Pretrained] Loading weights from: {pretrained_path}")
    
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
    
    # Get state dict
    if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
        state_dict = checkpoint['ema_state_dict']
        if is_main:
            print("[Pretrained] Using EMA weights")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if is_main:
            print("[Pretrained] Using model weights")
    else:
        if is_main:
            print("[Pretrained] ERROR: No valid state_dict found in checkpoint")
        return False
    
    # Remove 'module.' prefix if present (from DDP training)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict
    
    # Load with strict=False since cross-attention and condition_encoder layers are new
    missing_keys, unexpected_keys = unet.load_state_dict(state_dict, strict=False)
    
    if is_main:
        # Cross-attention related keys that are expected to be missing
        crossattn_patterns = ['cross_attn', 'condition_encoder']
        
        if missing_keys:
            # Separate expected missing keys (cross-attention layers) from unexpected ones
            expected_missing = [k for k in missing_keys if any(p in k for p in crossattn_patterns)]
            actual_missing = [k for k in missing_keys if not any(p in k for p in crossattn_patterns)]
            
            if expected_missing:
                print(f"[Pretrained] New layers to train ({len(expected_missing)}): cross-attention & condition encoder")
            if actual_missing:
                print(f"[Pretrained] WARNING: Unexpected missing keys ({len(actual_missing)}): {actual_missing[:5]}{'...' if len(actual_missing) > 5 else ''}")
        
        if unexpected_keys:
            print(f"[Pretrained] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        
        if 'epoch' in checkpoint:
            print(f"[Pretrained] Checkpoint from epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"[Pretrained] Checkpoint val_loss: {checkpoint['val_loss']:.4f}")
        
        print("[Pretrained] Successfully loaded pretrained weights (cross-attention layers randomly initialized)")
    
    return True


class ConditionalDiffusionWrapper:
    """
    Wrapper for GaussianDiffusion to support conditional generation.
    CT conditions the model via cross-attention (not concatenation).
    """
    
    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion
        self.timesteps = diffusion.timesteps
    
    def p_losses(self, denoise_model, x_start, t, condition, noise=None):
        """
        Training loss with condition (CT) via cross-attention.
        
        Args:
            denoise_model: UNetModelCrossAttn
            x_start: clean PET images [B, 1, H, W]
            t: timesteps [B]
            condition: CT images in PET space [B, 1, H, W]
            noise: optional noise to use
        
        Returns:
            Loss value
        """
        # Ensure float32 type
        x_start = x_start.float()
        condition = condition.float()
        
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward diffusion on PET
        x_noisy = self.diffusion.q_sample(x_start, t, noise=noise)
        
        # Cross-attention: pass x_noisy and condition separately
        # Model signature: forward(x, timesteps, context)
        predicted_noise = denoise_model(x_noisy, t, condition)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    @torch.no_grad()
    def conditional_sample(self, denoise_model, condition, shape=None, 
                          method='ddim', ddim_steps=50, eta=0.0):
        """
        Generate PET samples conditioned on CT via cross-attention.
        
        Args:
            denoise_model: UNetModelCrossAttn
            condition: CT images in PET space [B, 1, H, W]
            shape: Output shape (if None, inferred from condition)
            method: Sampling method ('ddpm' or 'ddim')
            ddim_steps: Number of DDIM steps
            eta: DDIM eta parameter
        
        Returns:
            Generated PET images [B, 1, H, W]
        """
        device = condition.device
        batch_size = condition.shape[0]
        
        if shape is None:
            shape = (batch_size, 1, condition.shape[2], condition.shape[3])
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        if method == 'ddim':
            # DDIM sampling with cross-attention condition
            c = self.timesteps // ddim_steps
            timesteps_seq = np.asarray(list(range(0, self.timesteps, c)))
            
            for i in reversed(range(len(timesteps_seq))):
                t = torch.full((batch_size,), timesteps_seq[i], device=device, dtype=torch.long)
                
                # Cross-attention: pass condition as context
                pred_noise = denoise_model(img, t, condition)
                
                # Get alpha values
                alpha_t = self.diffusion.alphas_cumprod[timesteps_seq[i]]
                if i > 0:
                    alpha_t_prev = self.diffusion.alphas_cumprod[timesteps_seq[i - 1]]
                else:
                    alpha_t_prev = torch.tensor(1.0, device=device)
                
                # Predict x0
                pred_x0 = (img - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
                
                # Direction pointing to x_t
                dir_xt = torch.sqrt(1.0 - alpha_t_prev - eta ** 2 * (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) * pred_noise
                
                # Noise
                noise = torch.randn_like(img) if i > 0 else torch.zeros_like(img)
                sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
                
                img = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma_t * noise
        else:
            # DDPM sampling with cross-attention condition
            for i in reversed(range(self.timesteps)):
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                
                betas_t = self.diffusion.betas[t][:, None, None, None]
                sqrt_one_minus_alphas_cumprod_t = self.diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
                sqrt_recip_alphas_t = torch.sqrt(1.0 / self.diffusion.alphas[t])[:, None, None, None]
                
                # Cross-attention: pass condition as context
                pred_noise = denoise_model(img, t, condition)
                
                model_mean = sqrt_recip_alphas_t * (
                    img - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
                )
                
                if i == 0:
                    img = model_mean
                else:
                    posterior_variance_t = self.diffusion.posterior_variance[t][:, None, None, None]
                    noise = torch.randn_like(img)
                    img = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        return img


def train_epoch(cond_diffusion, unet, train_loader, optimizer, scheduler, device, config, 
                epoch, writer, global_step, ema=None, rank=0, debug=False):
    """Train conditional diffusion model for one epoch."""
    unet.train()
    
    loss_meter = AverageMeter()
    
    # Only show progress bar on main process
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for batch_idx, batch_data in enumerate(pbar):
        # Debug mode: only train few batches
        if debug and batch_idx >= 10:
            if rank == 0:
                print(f"[DEBUG] Stopping training after {batch_idx} batches")
            break
        
        # Unpack batch: pet, pet_affine, ct, ct_affine, key
        pet, pet_affine, ct, ct_affine, keys = batch_data
        
        # Move to device
        pet = pet.to(device)
        ct = ct.to(device)
        
        # Convert affines to numpy for spatial transform
        pet_affine_np = pet_affine.numpy()
        ct_affine_np = ct_affine.numpy()
        
        # Align CT to PET space (2D)
        # pet shape: [B, 1, H, W], ct shape: [B, 1, H', W']
        pet_h, pet_w = pet.shape[2], pet.shape[3]
        ct_in_pet_space = align_ct_to_pet_batch(
            ct_batch=ct,
            ct_affines=ct_affine_np,
            pet_affines=pet_affine_np,
            pet_shape=(pet_h, pet_w),
            device=device,
            mode='bilinear'
        )
        
        # Ensure float32 type (affine operations may produce float64)
        ct_in_pet_space = ct_in_pet_space.float()
        
        # Debug: print shapes once
        if debug and batch_idx == 0 and rank == 0:
            print(f"\n[DEBUG] Shape check:")
            print(f"  PET shape: {pet.shape}")
            print(f"  CT shape: {ct.shape}")
            print(f"  CT in PET space shape: {ct_in_pet_space.shape}")
            print(f"  Model input (x_noisy): [B, 1, {pet_h}, {pet_w}]")
            print(f"  Context (CT): [B, 1, {pet_h}, {pet_w}] -> encoded for cross-attention")
            print(f"  PET affine[0]:\n{pet_affine_np[0]}")
            print(f"  CT affine[0]:\n{ct_affine_np[0]}")
            print(f"  PET value range: [{pet.min():.4f}, {pet.max():.4f}]")
            print(f"  CT value range: [{ct.min():.4f}, {ct.max():.4f}]")
            print(f"  CT in PET space range: [{ct_in_pet_space.min():.4f}, {ct_in_pet_space.max():.4f}]")
        
        batch_size = pet.shape[0]
        
        # Sample timesteps
        t = torch.randint(0, cond_diffusion.timesteps, (batch_size,), device=device).long()
        
        # Calculate conditional diffusion loss
        loss = cond_diffusion.p_losses(unet, pet, t, ct_in_pet_space)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config['training']['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), config['training']['grad_clip'])
        
        optimizer.step()
        scheduler.step()
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        # Update meters
        loss_meter.update(loss.item(), batch_size)
        
        # Logging (only on main process)
        if rank == 0 and batch_idx % config['training']['log_interval'] == 0:
            writer.add_scalar('train/loss', loss.item(), global_step[0])
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step[0])
            
            # WandB logging
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/step': global_step[0],
                    'train/epoch': epoch
                }, step=global_step[0])
            
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'loss': f"{loss_meter.avg:.4f}"
                })
        
        global_step[0] += 1
    
    return loss_meter.avg


@torch.no_grad()
def evaluate(cond_diffusion, unet, val_loader, device, config, epoch, writer, 
             samples_dir, rank=0, debug=False):
    """
    Evaluate conditional diffusion model on validation set.
    Includes loss computation, conditional sampling, and metrics calculation.
    """
    unet.eval()
    
    loss_meter = AverageMeter()
    mse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    # Get data_range from config
    data_range = config['evaluation']['metrics']['data_range']
    
    if rank == 0:
        print(f"\n[Evaluation] Using data_range={data_range} for PSNR/SSIM")
    
    # Only show progress bar on main process
    if rank == 0:
        pbar = tqdm(val_loader, desc="Validation")
    else:
        pbar = val_loader
    
    # Collect samples for visualization
    vis_samples = []
    num_vis_samples = config['evaluation']['num_samples']
    
    for batch_idx, batch_data in enumerate(pbar):
        # Debug mode: only validate few batches
        if debug and batch_idx >= 5:
            if rank == 0:
                print(f"[DEBUG] Stopping validation after {batch_idx} batches")
            break
        
        # Unpack batch
        pet, pet_affine, ct, ct_affine, keys = batch_data
        
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
        )
        
        # Ensure float32 type (affine operations may produce float64)
        ct_in_pet_space = ct_in_pet_space.float()
        
        batch_size = pet.shape[0]
        
        # Calculate validation loss
        t = torch.randint(0, cond_diffusion.timesteps, (batch_size,), device=device).long()
        loss = cond_diffusion.p_losses(unet, pet, t, ct_in_pet_space)
        loss_meter.update(loss.item(), batch_size)
        
        # Conditional sampling for metrics (only on main process to save compute)
        if rank == 0 and len(vis_samples) < num_vis_samples:
            # Generate samples
            pred_pet = cond_diffusion.conditional_sample(
                unet,
                ct_in_pet_space,
                method=config['diffusion']['sampling']['method'],
                ddim_steps=config['diffusion']['sampling']['ddim_steps'],
                eta=config['diffusion']['sampling']['eta']
            )
            
            # Calculate metrics (2D)
            mse_val = calculate_mse(pred_pet, pet)
            psnr_val = calculate_psnr(pred_pet, pet, data_range=data_range)
            ssim_val = calculate_ssim(pred_pet, pet, data_range=data_range)
            
            mse_meter.update(mse_val, batch_size)
            psnr_meter.update(psnr_val, batch_size)
            ssim_meter.update(ssim_val, batch_size)
            
            # Collect for visualization
            for i in range(min(batch_size, num_vis_samples - len(vis_samples))):
                vis_samples.append({
                    'ct': ct_in_pet_space[i].cpu(),
                    'gt_pet': pet[i].cpu(),
                    'pred_pet': pred_pet[i].cpu(),
                    'key': keys[i] if isinstance(keys, list) else keys
                })
    
    # Log results (only on main process)
    if rank == 0:
        print(f"\n[Validation] Epoch {epoch}:")
        print(f"  Loss: {loss_meter.avg:.4f}")
        print(f"  MSE:  {mse_meter.avg:.6f}")
        print(f"  PSNR: {psnr_meter.avg:.2f} dB")
        print(f"  SSIM: {ssim_meter.avg:.4f}")
        
        # TensorBoard
        writer.add_scalar('val/loss', loss_meter.avg, epoch)
        writer.add_scalar('val/mse', mse_meter.avg, epoch)
        writer.add_scalar('val/psnr', psnr_meter.avg, epoch)
        writer.add_scalar('val/ssim', ssim_meter.avg, epoch)
        
        # WandB
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                'val/loss': loss_meter.avg,
                'val/mse': mse_meter.avg,
                'val/psnr': psnr_meter.avg,
                'val/ssim': ssim_meter.avg,
                'val/epoch': epoch
            })
        
        # Save visualizations
        if vis_samples:
            save_visualizations(vis_samples, samples_dir, epoch, writer)
    
    return loss_meter.avg, mse_meter.avg, psnr_meter.avg, ssim_meter.avg


def save_visualizations(vis_samples, samples_dir, epoch, writer):
    """Save visualization images."""
    os.makedirs(samples_dir, exist_ok=True)
    
    n_samples = len(vis_samples)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(vis_samples):
        ct = sample['ct'][0].numpy()
        gt = sample['gt_pet'][0].numpy()
        pred = sample['pred_pet'][0].numpy()
        diff = np.abs(gt - pred)
        
        # CT
        axes[i, 0].imshow(ct, cmap='gray')
        axes[i, 0].set_title(f'CT (condition)')
        axes[i, 0].axis('off')
        
        # GT PET
        axes[i, 1].imshow(gt, cmap='hot')
        axes[i, 1].set_title(f'GT PET')
        axes[i, 1].axis('off')
        
        # Predicted PET
        axes[i, 2].imshow(pred, cmap='hot')
        axes[i, 2].set_title(f'Pred PET')
        axes[i, 2].axis('off')
        
        # Difference
        im = axes[i, 3].imshow(diff, cmap='hot')
        axes[i, 3].set_title(f'|GT - Pred|')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(samples_dir, f'samples_epoch{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Log to WandB
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({'val/samples': wandb.Image(fig)}, step=epoch)
    
    plt.close()
    
    # Save individual samples as npy for detailed analysis
    for i, sample in enumerate(vis_samples[:4]):  # Save first 4
        npy_path = os.path.join(samples_dir, f'epoch{epoch}_sample{i}.npz')
        np.savez(npy_path,
                 ct=sample['ct'].numpy(),
                 gt_pet=sample['gt_pet'].numpy(),
                 pred_pet=sample['pred_pet'].numpy(),
                 key=str(sample['key']))


def main():
    parser = argparse.ArgumentParser(description='Train CT to PET Conditional Diffusion Model (2D)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained', type=str, default=None, 
                        help='Path to pretrained weights (default: from config, "none" to skip)')
    parser.add_argument('--debug', action='store_true', help='Debug mode (quick test)')
    parser.add_argument('--debug_samples', type=int, default=20, help='Number of samples for debug mode')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override resume path if provided
    if args.resume is not None:
        config['resume']['checkpoint'] = args.resume
    
    # Override pretrained path if provided
    if args.pretrained is not None:
        config['pretrained']['checkpoint'] = args.pretrained
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Setup device
    if world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed(config['experiment']['seed'] + rank)
    
    # Create directories (only on main process)
    # FIX: Ensure output_dir/name directory exists (bug in original train_ldpet_pretrain.py)
    checkpoint_dir = os.path.join(config['experiment']['output_dir'], config['experiment']['name'])
    samples_dir = os.path.join(config['experiment']['log_dir'], 'samples')
    
    if is_main_process():
        os.makedirs(config['experiment']['output_dir'], exist_ok=True)
        os.makedirs(config['experiment']['log_dir'], exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure this exists!
        os.makedirs(samples_dir, exist_ok=True)
        
        print("="*70)
        print("CT to PET Conditional Diffusion Training (2D Cross-Attention)")
        print("="*70)
        print(f"Config: {args.config}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"Samples dir: {samples_dir}")
        print(f"Data range for metrics: {config['evaluation']['metrics']['data_range']}")
        if args.debug:
            print(f"DEBUG MODE: Using {args.debug_samples} samples")
        print("="*70)
    
    # Wait for main process to create directories
    if world_size > 1:
        dist.barrier()
    
    # Create dataloaders
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
        config, rank, world_size, args.debug, args.debug_samples
    )
    
    if is_main_process():
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_loader.dataset)} samples")
        print(f"  Val: {len(val_loader.dataset)} samples")
    
    # Create conditional UNet model with cross-attention (in_channels=1, context_channels for CT)
    unet = UNetModelCrossAttn(
        in_channels=config['diffusion']['unet']['in_channels'],  # 1
        model_channels=config['diffusion']['unet']['model_channels'],
        out_channels=config['diffusion']['unet']['out_channels'],  # 1
        num_res_blocks=config['diffusion']['unet']['num_res_blocks'],
        attention_resolutions=tuple(config['diffusion']['unet']['attention_resolutions']),
        dropout=config['diffusion']['unet']['dropout'],
        channel_mult=tuple(config['diffusion']['unet']['channel_mult']),
        num_heads=config['diffusion']['unet']['num_heads'],
        context_channels=config['diffusion']['unet']['context_channels']  # CT condition encoded dim
    ).to(device)
    
    if is_main_process():
        total_params = sum(p.numel() for p in unet.parameters())
        trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        print(f"\nModel Summary:")
        print(f"  UNet in_channels: {config['diffusion']['unet']['in_channels']}")
        print(f"  UNet out_channels: {config['diffusion']['unet']['out_channels']}")
        print(f"  UNet context_channels: {config['diffusion']['unet']['context_channels']}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    # Load pretrained weights (before DDP wrapping)
    pretrained_path = config['pretrained']['checkpoint']
    load_pretrained_weights(unet, pretrained_path, device, is_main=is_main_process())
    
    # Create Gaussian diffusion
    base_diffusion = GaussianDiffusion(
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
        beta_start=config['diffusion'].get('beta_start', 0.0001),
        beta_end=config['diffusion'].get('beta_end', 0.02),
        device=device
    )
    
    # Wrap with conditional diffusion
    cond_diffusion = ConditionalDiffusionWrapper(base_diffusion)
    
    # Wrap model with DDP
    if world_size > 1:
        unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config['training']['learning_rate'],
        betas=tuple(config['training']['betas']),
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    total_steps = len(train_loader) * config['training']['epochs']
    warmup_steps = len(train_loader) * config['training']['scheduler']['warmup_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr=config['training']['scheduler']['min_lr']
    )
    
    # Create EMA (using the underlying model for DDP)
    ema = None
    if config['training']['use_ema']:
        ema = EMA(unet, decay=config['training']['ema_decay'])
    
    # TensorBoard writer
    writer = None
    if is_main_process():
        writer = SummaryWriter(config['experiment']['log_dir'])
    
    # Initialize wandb (skip in debug mode)
    if is_main_process() and config['training']['use_wandb'] and WANDB_AVAILABLE and not args.debug:
        wandb.init(
            project=config['training']['wandb_project'],
            entity=config['training'].get('wandb_entity', None),
            name=config['experiment']['name'],
            config=config
        )
    elif is_main_process() and args.debug:
        print("[DEBUG] WandB disabled in debug mode")
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    global_step = [0]
    
    resume_path = config['resume']['checkpoint']
    if resume_path is not None and os.path.exists(resume_path):
        checkpoint = load_checkpoint(resume_path, unet, optimizer)
        if checkpoint is not None:
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            global_step[0] = checkpoint.get('global_step', 0)
            
            # Load scheduler state
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load EMA state
            if ema is not None and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
                ema.shadow = checkpoint['ema_state_dict']
            
            if is_main_process():
                print(f"\nResumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    
    # Training loop
    for epoch in range(start_epoch, config['training']['epochs']):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(
            cond_diffusion, unet, train_loader, optimizer, scheduler, device, config,
            epoch, writer, global_step, ema, rank, args.debug
        )
        
        # Validate
        if (epoch + 1) % config['training']['eval_interval'] == 0:
            # Apply EMA for evaluation
            if ema is not None:
                ema.apply_shadow()
            
            val_loss, mse, psnr_val, ssim_val = evaluate(
                cond_diffusion,
                unet,
                val_loader, device, config, epoch, writer, samples_dir, rank, args.debug
            )
            
            if ema is not None:
                ema.restore()
            
            if is_main_process():
                print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                print(f"  Metrics: MSE={mse:.6f}, PSNR={psnr_val:.2f}dB, SSIM={ssim_val:.4f}")
                
                # Save checkpoint at intervals
                if (epoch + 1) % config['training']['save_interval'] == 0:
                    save_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                    model_state = unet.module.state_dict() if world_size > 1 else unet.state_dict()
                    save_checkpoint({
                        'epoch': epoch,
                        'global_step': global_step[0],
                        'model_state_dict': model_state,
                        'ema_state_dict': ema.shadow if ema else None,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss,
                        'mse': mse,
                        'psnr': psnr_val,
                        'ssim': ssim_val,
                        'config': config
                    }, save_path)
                    print(f"  Saved checkpoint: {save_path}")
                
                # Save best checkpoint
                if config['training']['save_best'] and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
                    model_state = unet.module.state_dict() if world_size > 1 else unet.state_dict()
                    save_checkpoint({
                        'epoch': epoch,
                        'global_step': global_step[0],
                        'model_state_dict': model_state,
                        'ema_state_dict': ema.shadow if ema else None,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss,
                        'mse': mse,
                        'psnr': psnr_val,
                        'ssim': ssim_val,
                        'config': config
                    }, save_path)
                    print(f"  Saved BEST checkpoint with val loss: {val_loss:.4f}")
        
        # Debug mode: only run 1 epoch
        if args.debug:
            print(f"\n[DEBUG] Completed 1 epoch, exiting debug mode")
            break
    
    # Cleanup
    if is_main_process() and writer is not None:
        writer.close()
    
    if is_main_process() and config['training']['use_wandb'] and WANDB_AVAILABLE and not args.debug:
        wandb.finish()
    
    cleanup_distributed()
    
    if is_main_process():
        print("\n" + "="*70)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
        print(f"Logs saved to: {config['experiment']['log_dir']}")
        print("="*70)


if __name__ == '__main__':
    main()
