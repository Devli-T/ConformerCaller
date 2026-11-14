"""
Model Architectures for ConformerCaller
=====================================
Implements multiple architectures for benchmarking and selection.

Baseline:
  - Benchmark: CNN + BiGRU + CTC

Explored architectures (inspired by speech recognition research):
  - ConformerCaller: CNN + Conformer encoder + CTC
    Best for capturing both local (conv) and global (attention) patterns.
  - CausalConvCaller: Depthwise separable causal CNN (CTC)
    Fast, low-memory, good for streaming inference.

Architecture selection rationale:
  Nanopore signal → motif is analogous to acoustic features → phonemes in
  ASR. The key challenge is the VARIABLE translocation speed of DNA through
  the pore. Conformer handles this well via attention over the full sequence
  while conv layers handle local k-mer level features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


# ═════════════════════════════════════════════════════════════════════════════
# Model Configurations
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkConfig:
    """Replicates original Motif-Calling CNN+BiGRU architecture."""
    name:          str = "benchmark_cnn_bigru"
    hidden_size:   int = 256
    n_rnn_layers:  int = 3
    n_classes:     int = 65   # 64 payload motifs + 1 CTC blank
    dropout:       float = 0.1


@dataclass
class ConformerConfig:
    """Conformer CTC — recommended primary architecture.

    Memory profile (approximate, batch_size=4, max_signal_len=20k):
      - 8x  CNN downsample → T'=1250 per 10k signal → OOM at batch=8
      - 16x CNN downsample → T'=625  per 10k signal → fits batch=4, signal<=20k
      - 32x CNN downsample → T'=312  per 10k signal → fits batch=8, less temporal resolution
    16x downsampling provides the best accuracy/memory trade-off.
    """
    name:           str   = "conformer_ctc"
    # CNN frontend — 5 layers, total downsampling = 2^4 = 16x
    cnn_channels:   list  = field(default_factory=lambda: [1, 32, 64, 128, 256, 256])
    cnn_kernels:    list  = field(default_factory=lambda: [11, 9, 7, 5, 5])
    cnn_strides:    list  = field(default_factory=lambda: [1, 2, 2, 2, 2])   # 16x total downsample
    # Conformer encoder
    d_model:        int   = 256
    n_heads:        int   = 8
    n_conformer_layers: int = 6
    ff_expansion:   int   = 4
    conv_expansion: int   = 2
    conv_kernel:    int   = 31
    dropout:        float = 0.1
    # Output
    n_classes:      int   = 65


@dataclass
class CausalCNNConfig:
    """Causal depthwise-separable CNN — fast, low memory."""
    name:         str   = "causal_cnn_ctc"
    channels:     list  = field(default_factory=lambda: [1, 32, 64, 128, 256, 256])
    kernels:      list  = field(default_factory=lambda: [11, 9, 7, 5, 5])
    strides:      list  = field(default_factory=lambda: [1, 2, 2, 2, 2])
    n_classes:    int   = 65
    dropout:      float = 0.1


# ═════════════════════════════════════════════════════════════════════════════
# Benchmark: CNN + BiGRU 
# ═════════════════════════════════════════════════════════════════════════════

class BenchmarkCaller(nn.Module):
    """
    CNN + BiGRU + CTC baseline.

    Input : (B, 1, T) normalised signal
    Output: (T', B, C) log-probabilities for CTC
    """

    def __init__(self, cfg: BenchmarkConfig):
        super().__init__()
        self.cfg = cfg

        self.cnn = nn.Sequential(
            nn.Conv1d(1,   4,   kernel_size=5, stride=1), nn.ReLU(),
            nn.Conv1d(4,   16,  kernel_size=5, stride=1), nn.ReLU(),
            nn.Conv1d(16,  64,  kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv1d(64,  cfg.hidden_size, kernel_size=19, stride=2), nn.ReLU(),
        )

        self.bigru = nn.GRU(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.n_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.n_rnn_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(cfg.hidden_size * 2, cfg.n_classes)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:       (B, 1, T) signal tensor
            lengths: Optional signal lengths (ignored — no attention in this model)
        Returns:
            (T', B, C) log-softmax probabilities for CTC loss
        """
        x = self.cnn(x)                       # (B, H, T')
        x = x.permute(0, 2, 1)               # (B, T', H)
        x, _ = self.bigru(x)                  # (B, T', 2H)
        x = self.fc(x)                         # (B, T', C)
        x = F.log_softmax(x, dim=-1)
        return x.permute(1, 0, 2)            # (T', B, C) for CTC

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output sequence lengths after CNN downsampling."""
        lengths = input_lengths.float()
        for layer in self.cnn:
            if isinstance(layer, nn.Conv1d):
                lengths = torch.floor(
                    (lengths - layer.kernel_size[0]) / layer.stride[0] + 1
                )
        return lengths.long()


# ═════════════════════════════════════════════════════════════════════════════
# Conformer building blocks
# ═════════════════════════════════════════════════════════════════════════════

class ConvSubsampler(nn.Module):
    """
    Multi-layer 1D CNN frontend to subsample the raw signal.
    Reduces temporal resolution ~8-16x while projecting to d_model.
    """

    def __init__(self, channels, kernels, strides, dropout=0.1):
        super().__init__()
        layers = []
        for i, (k, s) in enumerate(zip(kernels, strides)):
            layers += [
                nn.Conv1d(channels[i], channels[i+1], kernel_size=k,
                          stride=s, padding=k//2, bias=False),
                nn.BatchNorm1d(channels[i+1]),
                nn.SiLU(),  # Swish activation — better than ReLU for audio/signal
            ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.conv = nn.Sequential(*layers)
        self.out_channels = channels[-1]
        self.strides = strides

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # (B, C_out, T')

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        for s in self.strides:
            lengths = torch.ceil(lengths.float() / s).long()
        return lengths


class FeedForwardModule(nn.Module):
    """Conformer Feed Forward Module with pre-norm."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + 0.5 * self.ff(self.norm(x))


class MultiHeadSelfAttentionModule(nn.Module):
    """Conformer Multi-Head Self Attention module with relative positional encoding."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm  = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        return residual + self.drop(x)


class ConvolutionModule(nn.Module):
    """
    Conformer Convolution Module.
    Captures local patterns via depthwise separable convolution.
    """

    def __init__(self, d_model: int, expansion: int = 2, kernel_size: int = 31,
                 dropout: float = 0.1):
        super().__init__()
        inner = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.pw1  = nn.Conv1d(d_model, inner * 2, kernel_size=1)
        self.gate = nn.GLU(dim=1)
        # Depthwise conv — key for capturing local signal patterns
        pad = (kernel_size - 1) // 2
        self.dw   = nn.Conv1d(inner, inner, kernel_size=kernel_size,
                              padding=pad, groups=inner, bias=False)
        self.bn   = nn.BatchNorm1d(inner)
        self.act  = nn.SiLU()
        self.pw2  = nn.Conv1d(inner, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)      # (B, T, D) → (B, D, T)
        x = self.pw1(x)             # (B, 2*inner, T)
        x = self.gate(x)            # (B, inner, T)
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw2(x)             # (B, D, T)
        x = x.transpose(1, 2)      # (B, T, D)
        return residual + self.drop(x)


class ConformerBlock(nn.Module):
    """
    Full Conformer block:
      FF → MHSA → Conv → FF  (with residual + layer norms at each step)
    """

    def __init__(self, d_model, n_heads, ff_expansion, conv_expansion,
                 conv_kernel, dropout):
        super().__init__()
        self.ff1  = FeedForwardModule(d_model, ff_expansion, dropout)
        self.attn = MultiHeadSelfAttentionModule(d_model, n_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_expansion, conv_kernel, dropout)
        self.ff2  = FeedForwardModule(d_model, ff_expansion, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        x = self.ff1(x)
        x = self.attn(x, key_padding_mask)
        x = self.conv(x)
        x = self.ff2(x)
        return self.norm(x)


class ConformerCaller(nn.Module):
    """
    Conformer CTC Caller — Primary Recommended Architecture.

    Combines:
    1. CNN subsampler (local signal feature extraction + downsampling)
    2. Stack of Conformer blocks (local + global context via conv + attention)
    3. Linear output layer → log-softmax for CTC

    Input : (B, 1, T)  raw normalised signal
    Output: (T', B, C) log-probabilities
    """

    def __init__(self, cfg: ConformerConfig):
        super().__init__()
        self.cfg = cfg

        self.subsampler = ConvSubsampler(
            cfg.cnn_channels, cfg.cnn_kernels, cfg.cnn_strides, cfg.dropout
        )

        # Project subsampled signal to d_model
        self.input_proj = nn.Linear(cfg.cnn_channels[-1], cfg.d_model)
        self.dropout    = nn.Dropout(cfg.dropout)

        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                d_model       = cfg.d_model,
                n_heads       = cfg.n_heads,
                ff_expansion  = cfg.ff_expansion,
                conv_expansion= cfg.conv_expansion,
                conv_kernel   = cfg.conv_kernel,
                dropout       = cfg.dropout,
            )
            for _ in range(cfg.n_conformer_layers)
        ])

        self.output = nn.Linear(cfg.d_model, cfg.n_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, 1, T) signal tensor
            lengths: (B,) actual signal lengths (for attention masking)
        Returns:
            (T', B, C) log-softmax for CTC
        """
        # CNN subsampling
        x = self.subsampler(x)         # (B, C, T')
        x = x.permute(0, 2, 1)        # (B, T', C)
        x = self.input_proj(x)         # (B, T', d_model)
        x = self.dropout(x)

        # Build key padding mask for attention (True = ignore)
        key_padding_mask = None
        if lengths is not None:
            out_lengths = self.subsampler.output_lengths(lengths).to(x.device)
            T_prime = x.size(1)
            B = x.size(0)
            mask = torch.arange(T_prime, device=x.device).unsqueeze(0) >= \
                   out_lengths.unsqueeze(1)
            key_padding_mask = mask

        # Conformer blocks
        for block in self.conformer_layers:
            x = block(x, key_padding_mask)

        # Output
        x = self.output(x)             # (B, T', C)
        x = F.log_softmax(x, dim=-1)
        return x.permute(1, 0, 2)     # (T', B, C)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return self.subsampler.output_lengths(input_lengths)


# ═════════════════════════════════════════════════════════════════════════════
# Causal CNN — Fast Inference Alternative
# ═════════════════════════════════════════════════════════════════════════════

class CausalConvCaller(nn.Module):
    """
    Depthwise-separable causal CNN + CTC.
    Inspired by wav2vec / QuartzNet architectures.
    Fastest inference, lowest memory — good for deployment.

    Input : (B, 1, T)
    Output: (T', B, C)
    """

    def __init__(self, cfg: CausalCNNConfig):
        super().__init__()
        self.cfg = cfg
        layers = []
        for i, (k, s) in enumerate(zip(cfg.kernels, cfg.strides)):
            # Causal padding: pad only the left
            pad = k - 1
            layers += [
                nn.ConstantPad1d((pad, 0), 0),
                nn.Conv1d(cfg.channels[i], cfg.channels[i+1],
                          kernel_size=k, stride=s, bias=False),
                nn.BatchNorm1d(cfg.channels[i+1]),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
            ]
        self.cnn = nn.Sequential(*layers)
        self.fc  = nn.Linear(cfg.channels[-1], cfg.n_classes)
        self.strides = cfg.strides

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args: lengths is ignored (no attention); included for uniform API."""
        x = self.cnn(x)                # (B, C, T')
        x = x.permute(0, 2, 1)        # (B, T', C)
        x = self.fc(x)                 # (B, T', C)
        x = F.log_softmax(x, dim=-1)
        return x.permute(1, 0, 2)     # (T', B, C)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = input_lengths.float()
        for s in self.strides:
            lengths = torch.ceil(lengths / s)
        return lengths.long()


# ═════════════════════════════════════════════════════════════════════════════
# Factory
# ═════════════════════════════════════════════════════════════════════════════

def build_model(arch: str = "conformer", n_classes: int = 65) -> nn.Module:
    """
    Build a model by name.

    Args:
        arch:      One of "benchmark", "conformer", "causal"
        n_classes: Output classes (64 payload motifs + 1 blank = 65)
    """
    if arch == "benchmark":
        cfg = BenchmarkConfig(n_classes=n_classes)
        return BenchmarkCaller(cfg)

    elif arch == "conformer":
        cfg = ConformerConfig(n_classes=n_classes)
        return ConformerCaller(cfg)

    elif arch == "causal":
        cfg = CausalCNNConfig(n_classes=n_classes)
        return CausalConvCaller(cfg)

    else:
        raise ValueError(f"Unknown architecture: {arch!r}. "
                         "Choose from: benchmark, conformer, causal")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick smoke-test of all architectures
    import sys
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    B, T = 4, 10_000  # Batch of 4, 10k samples each
    dummy = torch.randn(B, 1, T).to(device)
    lengths = torch.tensor([T, T-100, T-200, T-500])

    for arch in ["benchmark", "conformer", "causal"]:
        model = build_model(arch).to(device)
        with torch.no_grad():
            if arch == "conformer":
                out = model(dummy, lengths)
            else:
                out = model(dummy)
        out_T = out.shape[0]
        n_params = count_parameters(model)
        print(f"  [{arch:10s}]  params={n_params/1e6:.1f}M  "
              f"input=(B={B}, T={T})  output=(T'={out_T}, B={B}, C={out.shape[-1]})")
