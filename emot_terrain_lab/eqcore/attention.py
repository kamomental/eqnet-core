from __future__ import annotations

from typing import Optional, Tuple

import torch

from emot_terrain_lab.eqcore.state import Params
from emot_terrain_lab.models.ssmax_encoder import (
    SSMaxEmotionSelfAttention,
    SSMaxMHAConfig,
)


def build_attention(
    embed_dim: int,
    num_heads: int = 8,
    params: Optional[Params] = None,
    dropout: float = 0.0,
) -> SSMaxEmotionSelfAttention:
    cfg = SSMaxMHAConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        bias=True,
        ssmax_s_init=params.s if params else 0.6,
        ssmax_s_learnable=False,
        use_sdpa=True,
        emotion_lambda=params.lam if params else 8.0,
        emotion_gamma=params.gamma if params else 0.6,
        emotion_learnable=False,
    )
    return SSMaxEmotionSelfAttention(cfg)


def ssmax_attention(
    h: torch.Tensor,
    params: Params,
    emotion_activation: Optional[torch.Tensor] = None,
    attn_module: Optional[SSMaxEmotionSelfAttention] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention weights using SSMax + emotion bias.

    Args:
        h: hidden states [B, T, E]
        params: control parameters (s, gamma, lam)
        emotion_activation: optional signal [B, T] or [B, 1, T]
        attn_module: reusable attention module (optional)

    Returns:
        attn_out: context vector [B, T, E]
        attn_prob: attention probabilities [B, H, T, T]
    """
    if attn_module is None:
        attn_module = build_attention(h.size(-1), params=params)
    # update module parameters (no grad)
    with torch.no_grad():
        attn_module.s.fill_(params.s)
        attn_module.emo.lambda_decay.copy_(torch.tensor(params.lam, device=h.device))
        attn_module.emo.gamma_strength.copy_(torch.tensor(params.gamma, device=h.device))

    out, attn = attn_module(
        h,
        attn_mask=None,
        key_padding_mask=None,
        is_causal=False,
        emotion_activation=emotion_activation,
    )
    return out, attn
