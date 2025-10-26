#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalable-Softmax + emotion-kernel (Green function inspired) TransformerEncoderLayer.

The layer is API-compatible with nn.TransformerEncoderLayer:
    forward(src, src_mask=None, src_key_padding_mask=None, is_causal=False, emotion_activation=None)

Key features
------------
- Two attention execution paths:
    (A) SDPA path (torch.nn.functional.scaled_dot_product_attention). Q is scaled by
        alpha = s * log(T) so that the internal softmax behaves like the Scalable-Softmax of the paper.
        Emotion bias is injected as an additive mask.
    (B) Dense (manual) path. Computes QK^T, adds emotion bias, applies scalable_softmax, multiplies V.
- Emotion kernel: bias[i, j] = gamma * norm( e[i] * exp(-|i-j| / lambda) )
  where e is an optional activation vector delivered as [B, T] or [B, 1, T].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _maybe_layer_norm(x: torch.Tensor, ln: nn.LayerNorm, pre_norm: bool) -> torch.Tensor:
    return ln(x) if pre_norm else x


def _residual(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return x + h


def scalable_softmax(logits: torch.Tensor, dim: int = -1, s: float = 0.5) -> torch.Tensor:
    """Softmax with dynamic scaling: softmax( (s * log(n)) * logits )."""
    n = logits.size(dim)
    scale = s * math.log(max(1, int(n)))
    return F.softmax(logits * scale, dim=dim)


# ---------------------------------------------------------------------------
# Emotion kernel (Green function inspired)
# ---------------------------------------------------------------------------

class EmotionKernel(nn.Module):
    """
    Generate an additive bias tensor for attention:
        bias[i, j] = gamma * norm( e[i] * exp(-|i - j| / lambda) )
    Inputs
    ------
    emotion_activation: optional tensor [B, T] or [B, 1, T]
    Output
    -------
    bias: tensor [B, H, T, T]
    """

    def __init__(self, lambda_decay: float = 8.0, gamma_strength: float = 0.5, learnable: bool = False) -> None:
        super().__init__()
        if learnable:
            self.lambda_decay = nn.Parameter(torch.tensor(float(lambda_decay)))
            self.gamma_strength = nn.Parameter(torch.tensor(float(gamma_strength)))
        else:
            self.register_buffer("lambda_decay", torch.tensor(float(lambda_decay)))
            self.register_buffer("gamma_strength", torch.tensor(float(gamma_strength)))
        self._cache = {}

    def _get_distance_kernel(self, T: int, device, dtype) -> torch.Tensor:
        key = (device, dtype, int(T), float(self.lambda_decay.item()))
        cached = self._cache.get(key)
        if cached is not None and cached.shape[0] == T:
            return cached
        idx = torch.arange(T, device=device, dtype=torch.int64)
        dist = (idx[:, None] - idx[None, :]).abs().to(dtype=torch.float32)
        lam = torch.clamp(self.lambda_decay, min=1e-6)
        kernel = torch.exp(-dist / lam).to(dtype=dtype, device=device)
        self._cache[key] = kernel
        return kernel

    def forward(
        self,
        B: int,
        H: int,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
        emotion_activation: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if emotion_activation is None:
            e = torch.zeros(B, T, device=device, dtype=dtype)
        else:
            if emotion_activation.dim() == 3 and emotion_activation.size(1) == 1:
                e = emotion_activation.squeeze(1).to(device=device, dtype=dtype)
            elif emotion_activation.dim() == 2:
                e = emotion_activation.to(device=device, dtype=dtype)
            else:
                raise ValueError("emotion_activation must be shaped [B, T] or [B, 1, T].")

        kernel = self._get_distance_kernel(T, device, dtype)
        bias_bt = e.unsqueeze(-1) * kernel.unsqueeze(0)

        mean = bias_bt.mean(dim=(-1, -2), keepdim=True)
        std = bias_bt.std(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        bias_norm = (bias_bt - mean) / std

        bias = bias_norm.unsqueeze(1).repeat(1, H, 1, 1)
        bias = bias * self.gamma_strength
        return bias


# ---------------------------------------------------------------------------
# Multi-head attention with scalable softmax + emotion bias
# ---------------------------------------------------------------------------

@dataclass
class SSMaxMHAConfig:
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    bias: bool = True
    ssmax_s_init: float = 0.6
    ssmax_s_learnable: bool = True
    use_sdpa: bool = True
    emotion_lambda: float = 8.0
    emotion_gamma: float = 0.5
    emotion_learnable: bool = False


class SSMaxEmotionSelfAttention(nn.Module):
    """Self-attention block supporting both SDPA and a manual dense path."""

    def __init__(self, cfg: SSMaxMHAConfig) -> None:
        super().__init__()
        assert cfg.embed_dim % cfg.num_heads == 0
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.embed_dim // cfg.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=cfg.bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=cfg.bias)

        if cfg.ssmax_s_learnable:
            self.s = nn.Parameter(torch.tensor(cfg.ssmax_s_init, dtype=torch.float32))
        else:
            self.register_buffer("s", torch.tensor(cfg.ssmax_s_init, dtype=torch.float32))

        self.dropout_attn = nn.Dropout(cfg.dropout)
        self.dropout_proj = nn.Dropout(cfg.dropout)

        self.emotion_kernel = EmotionKernel(cfg.emotion_lambda, cfg.emotion_gamma, cfg.emotion_learnable)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        emotion_activation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        Q = self._reshape_heads(self.q_proj(x))
        K = self._reshape_heads(self.k_proj(x))
        V = self._reshape_heads(self.v_proj(x))

        emo_bias = self.emotion_kernel(
            B=B, H=self.num_heads, T=T, device=device, dtype=dtype, emotion_activation=emotion_activation
        )

        add_mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.to(device=device, dtype=torch.bool).view(B, 1, 1, T)
            add_mask = torch.zeros(B, 1, 1, T, device=device, dtype=dtype)
            add_mask = add_mask.masked_fill(mask, float("-inf"))

        ext_mask = None
        if attn_mask is not None:
            if attn_mask.dim() == 2 and attn_mask.shape == (T, T):
                ext_mask = attn_mask.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 4:
                ext_mask = attn_mask.to(device=device, dtype=dtype)
            else:
                raise ValueError("attn_mask must have shape [T, T] or [B, 1, T, T] or [B, H, T, T].")

        def broadcast_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if mask is None:
                return None
            if mask.dim() == 4 and mask.shape == (B, 1, 1, T):
                return mask.expand(B, self.num_heads, T, T)
            if mask.dim() == 4 and mask.shape == (B, 1, T, T):
                return mask.expand(B, self.num_heads, T, T)
            if mask.dim() == 4 and mask.shape == (B, self.num_heads, T, T):
                return mask
            raise ValueError("Failed to broadcast mask to [B, H, T, T].")

        add_mask_bhtt = broadcast_mask(add_mask)
        ext_mask_bhtt = broadcast_mask(ext_mask)

        total_bias = emo_bias
        if add_mask_bhtt is not None:
            total_bias = total_bias + add_mask_bhtt
        if ext_mask_bhtt is not None:
            total_bias = total_bias + ext_mask_bhtt

        if self.cfg.use_sdpa:
            alpha = float(self.s) * math.log(max(1, int(T)))
            Q_scaled = Q * alpha
            q = Q_scaled.reshape(B * self.num_heads, T, self.head_dim)
            k = K.reshape(B * self.num_heads, T, self.head_dim)
            v = V.reshape(B * self.num_heads, T, self.head_dim)
            attn_bias = total_bias.reshape(B * self.num_heads, T, T)

            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=self.dropout_attn.p if self.training else 0.0,
                is_causal=is_causal,
            )
            attn_out = attn_out.reshape(B, self.num_heads, T, self.head_dim)
            out = self.dropout_proj(self.out_proj(self._merge_heads(attn_out)))

            with torch.no_grad():
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
                scores = scores * alpha + total_bias
                attn_prob = F.softmax(scores, dim=-1)

            return out, attn_prob

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if is_causal:
            causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal, float("-inf"))

        scores = scores + total_bias
        attn = scalable_softmax(scores, dim=-1, s=float(self.s))
        attn = self.dropout_attn(attn)

        attn_out = torch.matmul(attn, V)
        out = self.dropout_proj(self.out_proj(self._merge_heads(attn_out)))
        return out, attn


# ---------------------------------------------------------------------------
# Transformer encoder layer
# ---------------------------------------------------------------------------

@dataclass
class SSMaxEncoderConfig:
    d_model: int
    nhead: int
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    batch_first: bool = True
    norm_first: bool = True
    ssmax_s_init: float = 0.6
    ssmax_s_learnable: bool = True
    use_sdpa: bool = True
    emotion_lambda: float = 8.0
    emotion_gamma: float = 0.5
    emotion_learnable: bool = False


class SSMaxTransformerEncoderLayer(nn.Module):
    """API-compatible replacement for nn.TransformerEncoderLayer."""

    def __init__(self, cfg: SSMaxEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pre_norm = cfg.norm_first

        self.self_attn = SSMaxEmotionSelfAttention(
            SSMaxMHAConfig(
                embed_dim=cfg.d_model,
                num_heads=cfg.nhead,
                dropout=cfg.dropout,
                bias=True,
                ssmax_s_init=cfg.ssmax_s_init,
                ssmax_s_learnable=cfg.ssmax_s_learnable,
                use_sdpa=cfg.use_sdpa,
                emotion_lambda=cfg.emotion_lambda,
                emotion_gamma=cfg.emotion_gamma,
                emotion_learnable=cfg.emotion_learnable,
            )
        )

        self.linear1 = nn.Linear(cfg.d_model, cfg.dim_feedforward)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

        act = cfg.activation.lower()
        if act == "relu":
            self.activation = nn.ReLU()
        elif act == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("activation must be either 'relu' or 'gelu'.")

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        emotion_activation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = src
        if not self.cfg.batch_first:
            x = x.transpose(0, 1)

        h_in = _maybe_layer_norm(x, self.norm1, self.pre_norm)
        h_attn, attn_prob = self.self_attn(
            h_in,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
            emotion_activation=emotion_activation,
        )
        x = _residual(x, self.dropout1(h_attn))
        x = _maybe_layer_norm(x, self.norm1, not self.pre_norm)

        h = _maybe_layer_norm(x, self.norm2, self.pre_norm)
        h = self.linear2(self.dropout(self.activation(self.linear1(h))))
        x = _residual(x, self.dropout2(h))
        x = _maybe_layer_norm(x, self.norm2, not self.pre_norm)

        if not self.cfg.batch_first:
            x = x.transpose(0, 1)
        return x, attn_prob


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    torch.manual_seed(0)

    B, T, E = 2, 64, 128
    cfg = SSMaxEncoderConfig(
        d_model=E,
        nhead=8,
        dim_feedforward=4 * E,
        dropout=0.0,
        batch_first=True,
        norm_first=True,
        ssmax_s_init=0.6,
        ssmax_s_learnable=True,
        use_sdpa=True,
        emotion_lambda=6.0,
        emotion_gamma=0.8,
        emotion_learnable=False,
    )

    layer = SSMaxTransformerEncoderLayer(cfg)
    layer.eval()

    x = torch.randn(B, T, E)

    t = torch.arange(T, dtype=torch.float32)
    e1 = torch.exp(-0.5 * ((t - 18.0) / 3.0) ** 2)
    e2 = 0.7 * torch.exp(-0.5 * ((t - 45.0) / 4.5) ** 2)
    emotion = (e1 + e2).unsqueeze(0).repeat(B, 1)

    key_padding_mask = torch.zeros(B, T, dtype=torch.bool)
    key_padding_mask[:, -4:] = True

    y, attn = layer(
        x,
        src_mask=None,
        src_key_padding_mask=key_padding_mask,
        is_causal=False,
        emotion_activation=emotion,
    )

    print("Output shape:", y.shape)
    print("Attention shape:", attn.shape)
    with torch.no_grad():
        avg_max = attn.max(dim=-1).values.mean().item()
        entropy = (-(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=-1)).mean().item()
        print(f"Average max attention: {avg_max:.4f}")
        print(f"Average attention entropy: {entropy:.2f}")


if __name__ == "__main__":
    _demo()
