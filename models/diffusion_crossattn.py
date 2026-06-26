"""
Conditional UNet with Cross-Attention for CT-conditioned PET generation.
This module extends the base UNet with cross-attention mechanism for conditioning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import base components from original diffusion module
from .diffusion import timestep_embedding, ResBlock, AttentionBlock


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for conditioning on external context (e.g., CT images)."""
    
    def __init__(self, channels, context_channels, num_heads=1, context_level=0):
        super().__init__()
        self.channels = channels
        self.context_channels = context_channels
        self.num_heads = num_heads
        self.context_level = context_level  # Which level of context features to use
        
        self.norm = nn.GroupNorm(32, channels)
        self.norm_context = nn.GroupNorm(32, context_channels)
        
        # Query from x, Key/Value from context
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Conv2d(context_channels, channels * 2, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x, context):
        """
        Args:
            x: input tensor [B, C, H, W]
            context: conditioning tensor [B, context_C, H', W'] (will be resized if needed)
        """
        b, c, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        context = self.norm_context(context)
        
        # Resize context to match x if needed
        if context.shape[-2:] != x.shape[-2:]:
            context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=True)
        
        q = self.q(x)
        kv = self.kv(context)
        k, v = kv.chunk(2, dim=1)
        
        # Reshape for attention
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        
        # Attention: Q from x, K/V from context
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


class ConditionEncoder(nn.Module):
    """
    Encoder for conditioning input (CT image).
    Extracts multi-scale features for cross-attention at different UNet levels.
    """
    
    def __init__(self, in_channels=1, base_channels=64, channel_mult=(1, 2, 4, 4)):
        super().__init__()
        self.channel_mult = channel_mult
        
        # Initial conv
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling blocks to extract multi-scale features
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for mult in channel_mult:
            out_ch = base_channels * mult
            self.down_blocks.append(nn.Sequential(
                nn.GroupNorm(32, ch),
                nn.SiLU(),
                nn.Conv2d(ch, out_ch, 3, stride=2, padding=1),
                nn.GroupNorm(32, out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
            ))
            ch = out_ch
        
        self.out_channels = [base_channels * m for m in channel_mult]
    
    def forward(self, x):
        """
        Returns multi-scale features.
        
        Args:
            x: [B, 1, H, W] conditioning image
            
        Returns:
            List of feature maps at different scales
        """
        features = []
        h = self.input_conv(x)
        
        for block in self.down_blocks:
            h = block(h)
            features.append(h)
        
        return features


class UNetModelCrossAttn(nn.Module):
    """
    UNet with Cross-Attention conditioning for diffusion model.
    
    Conditioning is applied via cross-attention at specified resolution levels,
    rather than concatenation at input.
    
    Args:
        in_channels: number of input channels (1 for PET)
        model_channels: base channel count for the model
        out_channels: number of output channels (1 for noise prediction)
        context_channels: channels in conditioning encoder output
        num_res_blocks: number of residual blocks per resolution level
        attention_resolutions: resolutions at which to apply attention
        dropout: dropout probability
        channel_mult: channel multiplier for each resolution level
        num_heads: number of attention heads
    """
    
    def __init__(
        self,
        in_channels=1,
        model_channels=128,
        out_channels=1,
        context_channels=64,  # Base channels for condition encoder
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0.0,
        channel_mult=(1, 2, 3, 4),
        num_heads=8
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
        
        # Condition encoder for CT (use same channel_mult as UNet)
        self.condition_encoder = ConditionEncoder(
            in_channels=1,
            base_channels=context_channels,
            channel_mult=channel_mult  # Use same as UNet
        )
        
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
        
        # Track which levels have cross-attention
        self.cross_attn_levels = []
        
        # Downsampling
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        level_idx = 0
        
        for level, mult in enumerate(channel_mult):
            for block_idx in range(num_res_blocks):
                layers = nn.ModuleList([
                    ResBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ])
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    # Self-attention
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                    # Cross-attention with condition
                    # Map ds to context level: ds=8 -> level 3 (after 3 downsamples)
                    # context features are indexed from 0, after each downsample block
                    # ds=2 -> idx 0, ds=4 -> idx 1, ds=8 -> idx 2, ds=16 -> idx 3
                    ctx_level = max(0, int(math.log2(ds)) - 1)
                    ctx_level = min(ctx_level, len(channel_mult) - 1)
                    context_ch = context_channels * channel_mult[ctx_level]
                    layers.append(CrossAttentionBlock(ch, context_ch, num_heads=num_heads, context_level=ctx_level))
                    self.cross_attn_levels.append((len(self.input_blocks), level))
                
                self.input_blocks.append(layers)
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.ModuleList([
                    nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                ]))
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle block (uses the deepest context features)
        ctx_level = len(channel_mult) - 1
        context_ch = context_channels * channel_mult[ctx_level]
        self.middle_block = nn.ModuleList([
            ResBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            CrossAttentionBlock(ch, context_ch, num_heads=num_heads, context_level=ctx_level),
            ResBlock(ch, ch, time_embed_dim, dropout),
        ])
        
        # Upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = nn.ModuleList([
                    ResBlock(
                        ch + input_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ])
                ch = model_channels * mult
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                    # Same context level mapping as downsampling
                    ctx_level = max(0, int(math.log2(ds)) - 1)
                    ctx_level = min(ctx_level, len(channel_mult) - 1)
                    context_ch = context_channels * channel_mult[ctx_level]
                    layers.append(CrossAttentionBlock(ch, context_ch, num_heads=num_heads, context_level=ctx_level))
                
                if level and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                    ds //= 2
                
                self.output_blocks.append(layers)
        
        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )
        
        self._input_block_chans = input_block_chans
    
    def forward(self, x, timesteps, context):
        """
        Apply the model to an input batch with conditioning.
        
        Args:
            x: [B, in_channels, H, W] noisy input
            timesteps: [B] timestep values
            context: [B, 1, H, W] conditioning image (CT)
        
        Returns:
            [B, out_channels, H, W] predicted noise
        """
        # Encode conditioning - produces multi-scale features
        context_features = self.condition_encoder(context)
        
        # Time embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # Downsampling
        hs = []
        h = x
        ds = 1
        
        for module in self.input_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb)
                    elif isinstance(layer, CrossAttentionBlock):
                        # Use the context_level stored in the layer
                        ctx = context_features[layer.context_level]
                        h = layer(h, ctx)
                    elif isinstance(layer, AttentionBlock):
                        h = layer(h)
                    elif isinstance(layer, nn.Conv2d) and layer.stride[0] == 2:
                        h = layer(h)
                        ds *= 2
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
            elif isinstance(layer, CrossAttentionBlock):
                ctx = context_features[layer.context_level]
                h = layer(h, ctx)
            elif isinstance(layer, AttentionBlock):
                h = layer(h)
            else:
                h = layer(h)
        
        # Upsampling
        for module in self.output_blocks:
            skip_connection = hs.pop()
            if h.shape[-2:] != skip_connection.shape[-2:]:
                h = F.interpolate(h, size=skip_connection.shape[-2:], mode='nearest')
            h = torch.cat([h, skip_connection], dim=1)
            
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                elif isinstance(layer, CrossAttentionBlock):
                    ctx = context_features[layer.context_level]
                    h = layer(h, ctx)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                elif isinstance(layer, nn.ConvTranspose2d):
                    h = layer(h)
                    ds //= 2
                else:
                    h = layer(h)
        
        # Output
        h = self.out(h)
        return h


# Export for models/__init__.py
__all__ = ['UNetModelCrossAttn', 'CrossAttentionBlock', 'ConditionEncoder']