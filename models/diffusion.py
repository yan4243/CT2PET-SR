"""
Diffusion models for latent space denoising.
Includes UNet architecture and Gaussian diffusion process.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: 1D tensor of timestep values
        dim: embedding dimension
        max_period: controls the minimum frequency of the embeddings
    
    Returns:
        Tensor of shape [batch_size, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class AttentionBlock(nn.Module):
    """Self-attention block."""
    
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        
        # Attention
        q = q.permute(0, 1, 3, 2)  # (b, heads, hw, c_per_head)
        k = k.permute(0, 1, 2, 3)  # (b, heads, c_per_head, hw)
        
        attn = torch.matmul(q, k) * (int(c) ** (-0.5))
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        v = v.permute(0, 1, 3, 2)  # (b, heads, hw, c_per_head)
        out = torch.matmul(attn, v)  # (b, heads, hw, c_per_head)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        
        out = self.proj_out(out)
        return residual + out


class ResBlock(nn.Module):
    """Residual block with timestep embedding."""
    
    def __init__(self, in_channels, out_channels, temb_channels, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add timestep embedding
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return self.skip_connection(x) + h


class UNetModel(nn.Module):
    """
    UNet architecture for diffusion model in latent space.
    
    Args:
        in_channels: number of input channels (latent dimension)
        model_channels: base channel count for the model
        out_channels: number of output channels (same as in_channels)
        num_res_blocks: number of residual blocks per resolution level
        attention_resolutions: resolutions at which to apply attention
        dropout: dropout probability
        channel_mult: channel multiplier for each resolution level
        num_heads: number of attention heads
    """
    
    def __init__(
        self,
        in_channels=4,
        model_channels=128,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0.0,
        channel_mult=(1, 2, 4, 4),
        num_heads=4
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input convolution
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        # Downsampling
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                )
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle
        self.middle_block = nn.Sequential(
            ResBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(ch, ch, time_embed_dim, dropout),
        )
        
        # Upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(
                        nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
                    )
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
        
        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )
    
    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        
        Args:
            x: [batch_size, in_channels, height, width] tensor of inputs
            timesteps: [batch_size] tensor of timestep values
        
        Returns:
            [batch_size, out_channels, height, width] tensor of outputs
        """
        # Time embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # Downsampling
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
            else:
                h = layer(h)
        
        # Upsampling
        for module in self.output_blocks:
            skip_connection = hs.pop()
            # Handle size mismatch due to odd dimensions
            if h.shape[-2:] != skip_connection.shape[-2:]:
                # Resize h to match skip_connection
                h = F.interpolate(h, size=skip_connection.shape[-2:], mode='nearest')
            h = torch.cat([h, skip_connection], dim=1)
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)
        
        # Output
        h = self.out(h)
        return h


class GaussianDiffusion:
    """
    Gaussian diffusion process for training and sampling.
    
    Implements DDPM (Denoising Diffusion Probabilistic Models).
    """
    
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear',
        beta_start=0.0001,
        beta_end=0.02,
        device='cuda'
    ):
        self.timesteps = timesteps
        self.device = device
        
        # Generate beta schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == 'cosine':
            betas = self.cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (forward process).
        
        Args:
            x_start: initial data [batch_size, channels, height, width]
            t: timestep [batch_size]
            noise: if specified, the noise to add
        
        Returns:
            Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, denoise_model, x_start, t, noise=None):
        """
        Training loss (simplified DDPM objective).
        
        Args:
            denoise_model: the denoising network
            x_start: clean latent [batch_size, channels, height, width]
            t: timestep [batch_size]
            noise: if specified, the noise to predict
        
        Returns:
            Loss value
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    @torch.no_grad()
    def p_sample(self, denoise_model, x, t, t_index):
        """
        Sample from p(x_{t-1} | x_t) - single denoising step.
        """
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])[:, None, None, None]
        
        # Predict noise
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * denoise_model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, denoise_model, shape):
        """
        Generate samples from the model (reverse process).
        
        Args:
            denoise_model: the denoising network
            shape: shape of samples to generate [batch_size, channels, height, width]
        
        Returns:
            Generated samples
        """
        device = next(denoise_model.parameters()).device
        b = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(
                denoise_model,
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                i
            )
        
        return img
    
    @torch.no_grad()
    def ddim_sample(self, denoise_model, shape, ddim_steps=50, eta=0.0):
        """
        DDIM sampling for faster generation.
        
        Args:
            denoise_model: the denoising network
            shape: shape of samples to generate
            ddim_steps: number of DDIM steps (less than full timesteps)
            eta: controls stochasticity (0 = deterministic)
        
        Returns:
            Generated samples
        """
        device = next(denoise_model.parameters()).device
        b = shape[0]
        
        # Generate subsequence of timesteps
        c = self.timesteps // ddim_steps
        timesteps_seq = np.asarray(list(range(0, self.timesteps, c)))
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, len(timesteps_seq))):
            t = torch.full((b,), timesteps_seq[i], device=device, dtype=torch.long)
            
            # Predict noise
            pred_noise = denoise_model(img, t)
            
            # Get alpha values
            alpha_t = self.alphas_cumprod[timesteps_seq[i]]
            if i > 0:
                alpha_t_prev = self.alphas_cumprod[timesteps_seq[i - 1]]
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
        
        return img
    
    def sample(self, denoise_model, shape, method='ddpm', **kwargs):
        """
        Unified sampling interface.
        
        Args:
            denoise_model: the denoising network
            shape: shape of samples to generate
            method: 'ddpm' or 'ddim'
            **kwargs: additional arguments for specific sampling methods
        
        Returns:
            Generated samples
        """
        if method == 'ddpm':
            return self.p_sample_loop(denoise_model, shape)
        elif method == 'ddim':
            return self.ddim_sample(denoise_model, shape, **kwargs)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
