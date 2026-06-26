"""
Utility functions for training, evaluation, and visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_psnr(img1, img2, data_range=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1: First image tensor [B, C, H, W]
        img2: Second image tensor [B, C, H, W]
        data_range: Maximum possible pixel value
    
    Returns:
        Average PSNR across batch
    """
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    
    psnr_values = []
    for i in range(img1_np.shape[0]):
        # Convert to [H, W, C] format for skimage
        im1 = np.transpose(img1_np[i], (1, 2, 0))
        im2 = np.transpose(img2_np[i], (1, 2, 0))
        psnr_val = psnr(im1, im2, data_range=data_range)
        psnr_values.append(psnr_val)
    
    return np.mean(psnr_values)


def calculate_ssim(img1, img2, data_range=1.0):
    """
    Calculate Structural Similarity Index between two images.
    
    Args:
        img1: First image tensor [B, C, H, W]
        img2: Second image tensor [B, C, H, W]
        data_range: Maximum possible pixel value
    
    Returns:
        Average SSIM across batch
    """
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    
    ssim_values = []
    for i in range(img1_np.shape[0]):
        # Convert to [H, W, C] format for skimage
        im1 = np.transpose(img1_np[i], (1, 2, 0))
        im2 = np.transpose(img2_np[i], (1, 2, 0))
        ssim_val = ssim(im1, im2, data_range=data_range, channel_axis=2)
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)


def calculate_mse(img1, img2):
    """
    Calculate Mean Squared Error between two images.
    
    Args:
        img1: First image tensor [B, C, H, W]
        img2: Second image tensor [B, C, H, W]
    
    Returns:
        MSE value
    """
    return F.mse_loss(img1, img2).item()


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename):
    """Save checkpoint to disk."""
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None):
    """
    Load checkpoint from disk.
    
    Args:
        filename: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
    
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filename, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Handle DDP model state_dict mismatch
    # If model is DDP but state_dict doesn't have 'module.' prefix, add it
    # If model is not DDP but state_dict has 'module.' prefix, remove it
    model_is_ddp = hasattr(model, 'module')
    state_dict_is_ddp = any(k.startswith('module.') for k in state_dict.keys())
    
    if model_is_ddp and not state_dict_is_ddp:
        # Add 'module.' prefix
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    elif not model_is_ddp and state_dict_is_ddp:
        # Remove 'module.' prefix
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def find_latest_checkpoint(checkpoint_dir, prefix='autoencoder'):
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Checkpoint file prefix (e.g., 'autoencoder' or 'diffusion')
    
    Returns:
        Path to latest checkpoint or None if not found
    """
    import os
    import glob
    import re
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for epoch checkpoints (e.g., autoencoder_epoch10.pth)
    pattern = os.path.join(checkpoint_dir, f'{prefix}_epoch*.pth')
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # Extract epoch numbers and find the latest
    epoch_nums = []
    for ckpt in checkpoints:
        match = re.search(r'epoch(\d+)\.pth', ckpt)
        if match:
            epoch_nums.append((int(match.group(1)), ckpt))
    
    if not epoch_nums:
        return None
    
    # Sort by epoch number and return the latest
    epoch_nums.sort(key=lambda x: x[0])
    return epoch_nums[-1][1]


def find_best_checkpoint(checkpoint_dir, prefix='autoencoder'):
    """
    Find the best checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Checkpoint file prefix (e.g., 'autoencoder' or 'diffusion')
    
    Returns:
        Path to best checkpoint or None if not found
    """
    import os
    
    best_path = os.path.join(checkpoint_dir, f'{prefix}_best.pth')
    
    if os.path.exists(best_path):
        return best_path
    
    return None


def resolve_checkpoint_path(checkpoint_path, checkpoint_dir, stage='autoencoder'):
    """
    Resolve checkpoint path from various formats.
    
    Args:
        checkpoint_path: Can be:
            - Specific file path
            - "latest" to find latest checkpoint
            - "best" to find best checkpoint
            - None to skip loading
        checkpoint_dir: Directory containing checkpoints
        stage: Training stage ('autoencoder' or 'diffusion')
    
    Returns:
        Resolved checkpoint path or None
    """
    import os
    
    if checkpoint_path is None:
        return None
    
    if checkpoint_path == 'latest':
        return find_latest_checkpoint(checkpoint_dir, prefix=stage)
    
    elif checkpoint_path == 'best':
        return find_best_checkpoint(checkpoint_dir, prefix=stage)
    
    elif os.path.exists(checkpoint_path):
        return checkpoint_path
    
    # Check if it's a relative path under checkpoint_dir
    full_path = os.path.join(checkpoint_dir, checkpoint_path)
    if os.path.exists(full_path):
        return full_path
    
    return None


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    min_lr=0.0
):
    """
    Create a schedule with a learning rate that decreases following a cosine
    function after a warmup period.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr: Minimum learning rate
    
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr, cosine_decay)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def visualize_reconstruction(original, reconstruction, save_path=None):
    """
    Visualize original and reconstructed images side by side.
    
    Args:
        original: Original image tensor [B, C, H, W]
        reconstruction: Reconstructed image tensor [B, C, H, W]
        save_path: Optional path to save the figure
    
    Returns:
        Figure object
    """
    import matplotlib.pyplot as plt
    
    # Take first sample from batch
    orig = original[0, 0].detach().cpu().numpy()
    recon = reconstruction[0, 0].detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(orig, cmap='hot')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Reconstruction
    axes[1].imshow(recon, cmap='hot')
    axes[1].set_title('Reconstruction')
    axes[1].axis('off')
    
    # Difference
    diff = np.abs(orig - recon)
    im = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    
    return fig


def visualize_diffusion_samples(samples, save_path=None, nrow=4):
    """
    Visualize samples generated by the diffusion model.
    
    Args:
        samples: Sample tensor [B, C, H, W]
        save_path: Optional path to save the figure
        nrow: Number of images per row
    
    Returns:
        Figure object
    """
    import matplotlib.pyplot as plt
    
    n_samples = min(samples.shape[0], nrow * nrow)
    samples_np = samples[:n_samples, 0].detach().cpu().numpy()
    
    fig, axes = plt.subplots(nrow, nrow, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(n_samples):
        axes[i].imshow(samples_np[i], cmap='hot')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_samples, nrow * nrow):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    
    return fig


class EMA:
    """
    Exponential Moving Average of model parameters.
    Useful for stabilizing diffusion model training.
    """
    
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # Ensure shadow is on the same device as param
                shadow = self.shadow[name].to(param.device)
                new_average = (1.0 - self.decay) * param.data + self.decay * shadow
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                # Ensure shadow is on the same device as param
                param.data = self.shadow[name].to(param.device)
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def count_parameters(model):
    """Count number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, name="Model"):
    """Print a summary of the model architecture."""
    print(f"\n{'=' * 60}")
    print(f"{name} Summary")
    print(f"{'=' * 60}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"{'=' * 60}\n")
