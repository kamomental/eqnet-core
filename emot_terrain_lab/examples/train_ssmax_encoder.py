#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSMax + emotion kernel Transformer encoder stack with training loop.
Synthetic needle-in-haystack classification dataset.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def scalable_softmax(logits: torch.Tensor, dim: int = -1, s: float = 0.5) -> torch.Tensor:
    n = logits.size(dim)
    scale = s * math.log(max(1, int(n)))
    return F.softmax(logits * scale, dim=dim)


# ---------------------------------------------------------------------------
# Emotion kernel
# ---------------------------------------------------------------------------

class EmotionKernel(nn.Module):
    def __init__(self, lambda_decay: float = 8.0, gamma_strength: float = 0.5, learnable: bool = False) -> None:
        super().__init__()
        if learnable:
            self.lambda_decay = nn.Parameter(torch.tensor(float(lambda_decay)))
            self.gamma_strength = nn.Parameter(torch.tensor(float(gamma_strength)))
        else:
            self.register_buffer("lambda_decay", torch.tensor(float(lambda_decay)))
            self.register_buffer("gamma_strength", torch.tensor(float(gamma_strength)))
        self._cache = {}

    def _get_kernel(self, T: int, device, dtype) -> torch.Tensor:
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
                raise ValueError("emotion_activation must be [B,T] or [B,1,T].")

        kernel = self._get_kernel(T, device, dtype)
        bias_bt = e.unsqueeze(-1) * kernel.unsqueeze(0)

        mean = bias_bt.mean(dim=(-1, -2), keepdim=True)
        std = bias_bt.std(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        bias_norm = (bias_bt - mean) / std

        bias = bias_norm.unsqueeze(1).repeat(1, H, 1, 1)
        bias = bias * self.gamma_strength
        return bias


# ---------------------------------------------------------------------------
# Multi-head attention with Scalable-Softmax and emotion bias
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
    def __init__(self, cfg: SSMaxMHAConfig) -> None:
        super().__init__()
        assert cfg.embed_dim % cfg.num_heads == 0
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.embed_dim // cfg.num_heads

        self.q_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.bias)

        if cfg.ssmax_s_learnable:
            self.s = nn.Parameter(torch.tensor(cfg.ssmax_s_init, dtype=torch.float32))
        else:
            self.register_buffer("s", torch.tensor(cfg.ssmax_s_init, dtype=torch.float32))

        self.dropout_attn = nn.Dropout(cfg.dropout)
        self.dropout_proj = nn.Dropout(cfg.dropout)
        self.emotion_kernel = EmotionKernel(cfg.emotion_lambda, cfg.emotion_gamma, cfg.emotion_learnable)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
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

        Q = self._reshape(self.q_proj(x))
        K = self._reshape(self.k_proj(x))
        V = self._reshape(self.v_proj(x))

        emo_bias = self.emotion_kernel(
            B=B, H=self.num_heads, T=T, device=device, dtype=dtype, emotion_activation=emotion_activation
        )

        add_mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.to(device=device, dtype=torch.bool).view(B, 1, 1, T)
            add_mask = torch.zeros(B, 1, 1, T, device=device, dtype=dtype)
            add_mask.masked_fill_(mask, float("-inf"))

        ext_mask = None
        if attn_mask is not None:
            if attn_mask.dim() == 2 and attn_mask.shape == (T, T):
                ext_mask = attn_mask.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 4:
                ext_mask = attn_mask.to(device=device, dtype=dtype)
            else:
                raise ValueError("attn_mask must be [T,T] or [B,1,T,T] or [B,H,T,T].")

        def broadcast_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if mask is None:
                return None
            if mask.dim() == 4 and mask.size(1) == 1:
                return mask.expand(B, self.num_heads, mask.size(-2), mask.size(-1))
            if mask.dim() == 4 and mask.size(1) == self.num_heads:
                return mask
            raise ValueError("Failed to broadcast mask to [B, H, T, T].")

        add_mask_b = broadcast_mask(add_mask)
        ext_mask_b = broadcast_mask(ext_mask)

        total_bias = emo_bias
        if add_mask_b is not None:
            total_bias = total_bias + add_mask_b
        if ext_mask_b is not None:
            total_bias = total_bias + ext_mask_b

        if self.cfg.use_sdpa:
            alpha = float(self.s) * math.log(max(1, int(T)))
            q = (Q * alpha).reshape(B * self.num_heads, T, self.head_dim)
            k = K.reshape(B * self.num_heads, T, self.head_dim)
            v = V.reshape(B * self.num_heads, T, self.head_dim)
            attn_bias = total_bias.reshape(B * self.num_heads, T, T)

            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=self.dropout_attn.p if self.training else 0.0,
                is_causal=is_causal,
            )
            y = y.reshape(B, self.num_heads, T, self.head_dim)
            out = self.dropout_proj(self.out_proj(self._merge(y)))

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

        y = torch.matmul(attn, V)
        out = self.dropout_proj(self.out_proj(self._merge(y)))
        return out, attn


# ---------------------------------------------------------------------------
# Encoder layer / stack
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
        self.activation = nn.GELU() if cfg.activation.lower() == "gelu" else nn.ReLU()

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

        h_in = self.norm1(x) if self.pre_norm else x
        h_attn, attn_prob = self.self_attn(
            h_in,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
            emotion_activation=emotion_activation,
        )
        x = x + self.dropout1(h_attn)
        x = self.norm1(x) if not self.pre_norm else x

        h = self.norm2(x) if self.pre_norm else x
        h = self.linear2(self.dropout(self.activation(self.linear1(h))))
        x = x + self.dropout2(h)
        x = self.norm2(x) if not self.pre_norm else x

        if not self.cfg.batch_first:
            x = x.transpose(0, 1)
        return x, attn_prob


class SSMaxTransformerEncoder(nn.Module):
    def __init__(self, layer: SSMaxTransformerEncoderLayer, num_layers: int) -> None:
        super().__init__()
        layers = [layer]
        for _ in range(1, num_layers):
            layers.append(SSMaxTransformerEncoderLayer(layer.cfg))
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        emotion_activation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = src
        last_attn = None
        for layer in self.layers:
            x, attn = layer(
                x,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
                emotion_activation=emotion_activation,
            )
            last_attn = attn
        return x, last_attn


# ---------------------------------------------------------------------------
# Synthetic dataset: Needle-in-haystack classification
# ---------------------------------------------------------------------------

class NeedleSequenceDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        num_classes: int,
        needle_margin: int = 16,
        seed: int = 7,
        use_emotion: bool = True,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.margin = needle_margin
        self.use_emotion = use_emotion
        random.seed(seed)
        self.needle_ids = list(range(num_classes))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        T, V, C, mg = self.seq_len, self.vocab_size, self.num_classes, self.margin
        label = random.randint(0, C - 1)
        needle_token = self.needle_ids[label]

        tokens = torch.randint(low=0, high=V, size=(T,), dtype=torch.long)
        pos = random.randint(mg, T - mg - 1)
        tokens[pos] = needle_token + 2
        key_padding_mask = torch.zeros(T, dtype=torch.bool)

        if self.use_emotion:
            t = torch.arange(T, dtype=torch.float32)
            emotion = torch.exp(-0.5 * ((t - float(pos)) / 3.0) ** 2)
        else:
            emotion = torch.zeros(T, dtype=torch.float32)

        return tokens, label, key_padding_mask, emotion


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 4096
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    num_classes: int = 8
    max_len: int = 8192
    ssmax_s_init: float = 0.6
    emotion_lambda: float = 8.0
    emotion_gamma: float = 0.8
    use_sdpa: bool = True
    use_emotion: bool = True


class SSMaxClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)

        enc_cfg = SSMaxEncoderConfig(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            ssmax_s_init=cfg.ssmax_s_init,
            ssmax_s_learnable=True,
            use_sdpa=cfg.use_sdpa,
            emotion_lambda=cfg.emotion_lambda,
            emotion_gamma=cfg.emotion_gamma,
            emotion_learnable=False,
        )
        layer = SSMaxTransformerEncoderLayer(enc_cfg)
        self.encoder = SSMaxTransformerEncoder(layer, cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_classes)

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        emotion_activation: Optional[torch.Tensor] = None,
    ):
        B, T = tokens.shape
        pos_ids = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        h = self.embed(tokens) + self.pos(pos_ids)
        h, attn = self.encoder(
            h,
            mask=None,
            src_key_padding_mask=key_padding_mask,
            is_causal=False,
            emotion_activation=emotion_activation,
        )
        pooled = self.norm(h.mean(dim=1))
        logits = self.classifier(pooled)
        return logits, attn


# ---------------------------------------------------------------------------
# Training & evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    attn_max_sum, attn_ent_sum, attn_batches = 0.0, 0.0, 0
    start = time.time()

    for tokens, labels, kpm, emotion in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        kpm = kpm.to(device)
        emotion = emotion.to(device) if emotion is not None else None

        optimizer.zero_grad()
        logits, attn = model(tokens, key_padding_mask=kpm, emotion_activation=emotion)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * tokens.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += tokens.size(0)

        with torch.no_grad():
            max_attn = attn.max(dim=-1).values.mean().item()
            entropy = (-(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=-1)).mean().item()
            attn_max_sum += max_attn
            attn_ent_sum += entropy
            attn_batches += 1

    elapsed = time.time() - start
    return {
        "loss": total_loss / max(1, total_samples),
        "acc": total_correct / max(1, total_samples),
        "avg_max_attn": attn_max_sum / max(1, attn_batches),
        "avg_entropy": attn_ent_sum / max(1, attn_batches),
        "sec": elapsed,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    attn_max_sum, attn_ent_sum, attn_batches = 0.0, 0.0, 0
    start = time.time()

    for tokens, labels, kpm, emotion in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        kpm = kpm.to(device)
        emotion = emotion.to(device) if emotion is not None else None

        logits, attn = model(tokens, key_padding_mask=kpm, emotion_activation=emotion)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * tokens.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += tokens.size(0)

        max_attn = attn.max(dim=-1).values.mean().item()
        entropy = (-(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=-1)).mean().item()
        attn_max_sum += max_attn
        attn_ent_sum += entropy
        attn_batches += 1

    elapsed = time.time() - start
    return {
        "loss": total_loss / max(1, total_samples),
        "acc": total_correct / max(1, total_samples),
        "avg_max_attn": attn_max_sum / max(1, attn_batches),
        "avg_entropy": attn_ent_sum / max(1, attn_batches),
        "sec": elapsed,
    }


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    seq_len = 512
    num_classes = 8
    vocab_size = 4096
    train_sz = 2000
    valid_sz = 400
    batch_size = 32
    epochs = 5

    train_ds = NeedleSequenceDataset(train_sz, seq_len, vocab_size, num_classes, use_emotion=True)
    valid_ds = NeedleSequenceDataset(valid_sz, seq_len, vocab_size, num_classes, use_emotion=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    cfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        num_classes=num_classes,
        max_len=8192,
        ssmax_s_init=0.6,
        emotion_lambda=6.0,
        emotion_gamma=0.8,
        use_sdpa=True,
        use_emotion=True,
    )
    model = SSMaxClassifier(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        valid_stats = evaluate(model, valid_loader, device)
        print(
            f"[Epoch {epoch:02d}] "
            f"train: loss={train_stats['loss']:.4f} acc={train_stats['acc']:.3f} "
            f"attn_max={train_stats['avg_max_attn']:.3f} ent={train_stats['avg_entropy']:.2f} "
            f"sec={train_stats['sec']:.2f} | "
            f"valid: loss={valid_stats['loss']:.4f} acc={valid_stats['acc']:.3f} "
            f"attn_max={valid_stats['avg_max_attn']:.3f} ent={valid_stats['avg_entropy']:.2f} "
            f"sec={valid_stats['sec']:.2f}"
        )

    tokens, label, kpm, emotion = valid_ds[0]
    tokens = tokens.unsqueeze(0).to(device)
    kpm = kpm.unsqueeze(0).to(device)
    emotion = emotion.unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    model(tokens, key_padding_mask=kpm, emotion_activation=emotion)
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.time() - start) * 1000
    print(f"Inference latency (B=1, T={seq_len}): {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()
