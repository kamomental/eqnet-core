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
from typing import Any, Callable, Dict, Optional, Tuple

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

_ASSOC_LOGGER: Optional[Callable[[Dict[str, Any]], None]] = None
_ASSOC_CONTEXT_FN: Optional[Callable[[], Dict[str, Any]]] = None
_ASSOC_HEALTH_HOOK: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None


def register_assoc_logger(
    writer: Optional[Callable[[Dict[str, Any]], None]],
    context_fn: Optional[Callable[[], Dict[str, Any]]] = None,
) -> None:
    """Configure a callback that receives assoc-kernel metrics per forward."""

    global _ASSOC_LOGGER, _ASSOC_CONTEXT_FN
    _ASSOC_LOGGER = writer
    _ASSOC_CONTEXT_FN = context_fn


def register_assoc_health_hook(
    hook: Optional[Callable[[Dict[str, Any]], Optional[str]]],
) -> None:
    """Register a guard hook that can request runtime adjustments."""

    global _ASSOC_HEALTH_HOOK
    _ASSOC_HEALTH_HOOK = hook


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


try:  # PyTorch >= 2.0
    RMSNorm = nn.RMSNorm  # type: ignore[attr-defined]
except AttributeError:

    class RMSNorm(nn.Module):
        """Fallback RMSNorm for older PyTorch versions."""

        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            norm = x.pow(2).mean(dim=-1, keepdim=True)
            norm = torch.rsqrt(norm + self.eps)
            return self.weight * x * norm


def _cosine_distance(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    similarity = torch.matmul(q, k.transpose(-2, -1)).clamp(-1.0, 1.0)
    # ||q - k||^2 = 2 - 2 cos(theta)
    return (2.0 - 2.0 * similarity).clamp_min(0.0)


def _make_local_mask(T: int, window: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(T, device=device)
    return (idx[None, :] - idx[:, None]).abs() <= window


def _make_stride_mask(T: int, stride: int, device: torch.device) -> torch.Tensor:
    if stride <= 1:
        return torch.ones(T, T, device=device, dtype=torch.bool)
    keep = (torch.arange(T, device=device) % stride) == 0
    return keep[None, :].repeat(T, 1)


class AssocKernelAdapter(nn.Module):
    """
    Gaussian kernel associative attention adapter.

    This module mirrors the interface of SSMaxEmotionSelfAttention so that we can
    toggle between the standard attention and the associative kernel without
    touching the rest of the stack.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        sigma_init: float = 1.2,
        beta_local_init: float = -3.0,
        beta_global_init: float = -3.0,
        window_w: int = 128,
        stride_s: int = 8,
        use_kv_asymmetry: bool = False,
        use_bandwidth_gating: bool = False,
        emotion_lambda: float = 8.0,
        emotion_gamma: float = 0.5,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.use_kv_asymmetry = use_kv_asymmetry
        self.use_bandwidth_gating = use_bandwidth_gating
        self.window_w = window_w
        self.stride_s = stride_s

        self.norm = RMSNorm(embed_dim)
        self.qk_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_proj = nn.Dropout(dropout)

        self.sigma = nn.Parameter(torch.tensor(float(sigma_init)))
        self.beta_local = nn.Parameter(torch.tensor(float(beta_local_init)))
        self.beta_global = nn.Parameter(torch.tensor(float(beta_global_init)))

        self.emotion_kernel = EmotionKernel(emotion_lambda, emotion_gamma, learnable=False)
        self._last_metrics: Optional[Dict[str, float]] = None

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def _bandwidth_gate(self, T: int, device: torch.device) -> torch.Tensor:
        local = _make_local_mask(T, self.window_w, device=device).float()
        global_mask = _make_stride_mask(T, self.stride_s, device=device).float()
        gate = 0.0
        if self.use_bandwidth_gating:
            gate = torch.sigmoid(self.beta_local) * local + torch.sigmoid(self.beta_global) * global_mask
        else:
            gate = local
        return gate.clamp_min(1e-4)

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

        h = self.norm(x)
        shared = self.qk_proj(h)
        values = self.v_proj(h)

        if self.use_kv_asymmetry:
            v_shift = torch.zeros_like(values)
            v_shift[:, :-1, :] = values[:, 1:, :]
            values = v_shift

        q = shared
        k = shared

        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(values)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        dist_sq = _cosine_distance(q, k)
        sigma_sq = torch.clamp(self.sigma, min=1e-4) ** 2
        scores = (-dist_sq / sigma_sq).clamp(-12.0, 12.0)

        gate_tensor = None
        if self.use_bandwidth_gating or self.window_w > 0:
            gate_tensor = self._bandwidth_gate(T, device).unsqueeze(0).unsqueeze(0)
            scores = scores + gate_tensor.log()

        if is_causal:
            causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal, float("-inf"))

        if key_padding_mask is not None:
            mask = key_padding_mask.to(device=device, dtype=torch.bool).view(B, 1, 1, T)
            scores = scores.masked_fill(mask, float("-inf"))

        if attn_mask is not None:
            if attn_mask.dim() == 2 and attn_mask.shape == (T, T):
                scores = scores + attn_mask.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 4:
                scores = scores + attn_mask.to(device=device, dtype=dtype)

        emo_bias = self.emotion_kernel(
            B=B, H=self.num_heads, T=T, device=device, dtype=dtype, emotion_activation=emotion_activation
        )
        scores = scores + emo_bias

        stats_scores = scores.detach()
        try:
            max_score_abs = float(stats_scores.abs().max().item()) if stats_scores.numel() > 0 else 0.0
        except Exception:
            max_score_abs = 0.0
        try:
            max_score = float(stats_scores.max().item()) if stats_scores.numel() > 0 else 0.0
        except Exception:
            max_score = 0.0
        try:
            min_score = float(stats_scores.min().item()) if stats_scores.numel() > 0 else 0.0
        except Exception:
            min_score = 0.0
        gate_mean = None
        if gate_tensor is not None:
            try:
                gate_mean = float(gate_tensor.mean().item())
            except Exception:
                gate_mean = None

        attn_raw = torch.softmax(scores, dim=-1)
        attn = self.dropout_attn(attn_raw)

        try:
            entropy = float(
                (-(attn_raw.clamp_min(1e-12) * attn_raw.clamp_min(1e-12).log()).sum(dim=-1).mean().item())
            )
        except Exception:
            entropy = None

        attn_out = torch.matmul(attn, v)
        out = self.dropout_proj(self.out_proj(self._merge_heads(attn_out)))
        sigma_val = float(self.sigma.detach().item())
        beta_local_val = float(self.beta_local.detach().item())
        beta_global_val = float(self.beta_global.detach().item())
        self._last_metrics = {
            "sigma": sigma_val,
            "beta_local": beta_local_val,
            "beta_global": beta_global_val,
            "max_score_abs": max_score_abs,
            "max_score": max_score,
            "min_score": min_score,
            "attn_entropy": entropy,
            "gate_mean": gate_mean,
        }
        return out, attn

    @property
    def last_metrics(self) -> Optional[Dict[str, float]]:
        return self._last_metrics

    def bump_sigma(self, delta: float) -> float:
        with torch.no_grad():
            self.sigma.add_(float(delta))
            if self.sigma.item() < 1e-4:
                self.sigma.fill_(1e-4)
        return float(self.sigma.detach().item())

    def adjust_beta_local(self, delta: float) -> float:
        with torch.no_grad():
            self.beta_local.add_(float(delta))
        return float(self.beta_local.detach().item())

    def adjust_beta_global(self, delta: float) -> float:
        with torch.no_grad():
            self.beta_global.add_(float(delta))
        return float(self.beta_global.detach().item())

    def adjust_window(self, delta: int) -> int:
        new_val = max(0, int(self.window_w + int(delta)))
        self.window_w = new_val
        return self.window_w


class AttentionOrAssoc(nn.Module):
    """
    Wrapper that toggles between standard Scalable-Softmax attention and the
    associative kernel adapter.
    """

    def __init__(
        self,
        base_attn: SSMaxEmotionSelfAttention,
        assoc_attn: Optional[AssocKernelAdapter],
        *,
        use_assoc_kernel: bool = False,
    ) -> None:
        super().__init__()
        self.base_attn = base_attn
        self.assoc_attn = assoc_attn
        self.use_assoc_kernel = use_assoc_kernel and assoc_attn is not None
        self._last_assoc_metrics: Optional[Dict[str, float]] = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        emotion_activation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_assoc_kernel and self.assoc_attn is not None:
            out, attn = self.assoc_attn(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                emotion_activation=emotion_activation,
            )
            metrics = getattr(self.assoc_attn, "last_metrics", None)
            context: Dict[str, Any] = {}
            if _ASSOC_CONTEXT_FN is not None:
                try:
                    ctx_candidate = _ASSOC_CONTEXT_FN() or {}
                except Exception:
                    ctx_candidate = {}
                if isinstance(ctx_candidate, dict):
                    context = dict(ctx_candidate)
            guard_action: Optional[str] = None
            if metrics and _ASSOC_HEALTH_HOOK is not None:
                payload = dict(metrics)
                payload.update({k: v for k, v in context.items() if k not in payload})
                try:
                    guard_action = _ASSOC_HEALTH_HOOK(payload)
                except Exception:
                    guard_action = None
            if guard_action == "fallback_attention":
                self.assoc_attn.bump_sigma(0.1)
                self._emit_assoc_record(metrics, context, guard_action)
                out, attn = self.base_attn(
                    x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    is_causal=is_causal,
                    emotion_activation=emotion_activation,
                )
                self._last_assoc_metrics = None
                return out, attn
            if guard_action == "broaden_bandwidth":
                self.assoc_attn.adjust_beta_global(-0.25)
                self.assoc_attn.adjust_window(+16)
            elif guard_action == "narrow_bandwidth":
                self.assoc_attn.adjust_beta_local(+0.25)
                self.assoc_attn.adjust_window(-16)
            self._emit_assoc_record(metrics, context, guard_action)
            self._last_assoc_metrics = metrics
            return out, attn
        self._last_assoc_metrics = None
        return self.base_attn(
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            emotion_activation=emotion_activation,
        )

    @property
    def last_assoc_metrics(self) -> Optional[Dict[str, float]]:
        return self._last_assoc_metrics

    def _emit_assoc_record(
        self,
        metrics: Optional[Dict[str, float]],
        context: Dict[str, Any],
        guard_action: Optional[str],
    ) -> None:
        if metrics is None or _ASSOC_LOGGER is None or self.assoc_attn is None:
            return
        record: Dict[str, Any] = dict(metrics)
        record.setdefault("layer", getattr(self.assoc_attn, "layer_id", None))
        record["use_assoc_kernel"] = bool(self.use_assoc_kernel)
        record["use_kv_asymmetry"] = bool(self.assoc_attn.use_kv_asymmetry)
        record["use_bandwidth_gating"] = bool(self.assoc_attn.use_bandwidth_gating)
        record["w"] = int(self.assoc_attn.window_w)
        record["stride"] = int(self.assoc_attn.stride_s)
        if guard_action:
            record["guard_action"] = guard_action
        for key, value in (context or {}).items():
            record.setdefault(key, value)
        try:
            _ASSOC_LOGGER(record)
        except Exception:
            pass


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
    use_assoc_kernel: bool = False
    assoc_sigma_init: float = 1.2
    use_kv_asymmetry: bool = False
    use_bandwidth_gating: bool = False
    assoc_beta_local_init: float = -3.0
    assoc_beta_global_init: float = -3.0
    assoc_window_w: int = 128
    assoc_stride_s: int = 8
    layer_id: Optional[int] = None


class SSMaxTransformerEncoderLayer(nn.Module):
    """API-compatible replacement for nn.TransformerEncoderLayer."""

    def __init__(self, cfg: SSMaxEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pre_norm = cfg.norm_first

        base_attn = SSMaxEmotionSelfAttention(
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
        assoc_attn: Optional[AssocKernelAdapter] = None
        if cfg.use_assoc_kernel or cfg.use_kv_asymmetry or cfg.use_bandwidth_gating:
            assoc_attn = AssocKernelAdapter(
                embed_dim=cfg.d_model,
                num_heads=cfg.nhead,
                dropout=cfg.dropout,
                sigma_init=cfg.assoc_sigma_init,
                beta_local_init=cfg.assoc_beta_local_init,
                beta_global_init=cfg.assoc_beta_global_init,
                window_w=cfg.assoc_window_w,
                stride_s=cfg.assoc_stride_s,
                use_kv_asymmetry=cfg.use_kv_asymmetry,
                use_bandwidth_gating=cfg.use_bandwidth_gating,
                emotion_lambda=cfg.emotion_lambda,
                emotion_gamma=cfg.emotion_gamma,
            )
            assoc_attn.layer_id = getattr(cfg, "layer_id", None)

        self.self_attn = AttentionOrAssoc(
            base_attn=base_attn,
            assoc_attn=assoc_attn,
            use_assoc_kernel=cfg.use_assoc_kernel,
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

    def last_assoc_metrics(self) -> Optional[Dict[str, float]]:
        return self.self_attn.last_assoc_metrics


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
