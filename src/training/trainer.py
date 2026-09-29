"""
Trainer — CTC Training Loop for ConformerCaller
=============================================
Handles the full training lifecycle:
  - CTC loss with correct input/label length computation
  - Adam + cosine LR schedule with warmup
  - Gradient clipping (critical for CTC stability)
  - Validation loop with motif accuracy metrics
  - Checkpoint saving (best val loss)
  - Early stopping
"""


import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .ctc_decoder import GreedyCTCDecoder
from .metrics import motif_accuracy, edit_distance_ratio

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """All hyperparameters for one training run."""
    arch:             str   = "conformer"     # "benchmark" | "conformer" | "causal"
    n_classes:        int   = 65              # 64 payload motifs + 1 CTC blank
    hidden_size:      int   = 256
    n_epochs:         int   = 100
    batch_size:       int   = 8
    learning_rate:    float = 3e-4
    weight_decay:     float = 1e-5
    grad_clip:        float = 5.0
    warmup_epochs:    int   = 5
    patience:         int   = 15             # Early stopping
    checkpoint_dir:   Path  = Path("checkpoints")
    log_interval:     int   = 50             # Log every N batches
    device:           str   = "auto"         # "auto" | "mps" | "cuda" | "cpu"

    def get_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self.device)


class CosineWarmupScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_epochs
        self.total_steps  = total_epochs
        self.base_lrs     = [g["lr"] for g in optimizer.param_groups]
        self.step_num     = 0

    def step(self):
        self.step_num += 1
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = self._get_lr(base_lr)

    def _get_lr(self, base_lr: float) -> float:
        if self.step_num <= self.warmup_steps:
            return base_lr * self.step_num / max(1, self.warmup_steps)
        progress = (self.step_num - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


class Trainer:
    """Trains and evaluates a ConformerCaller model."""

    def __init__(self, cfg: TrainConfig, label_names: List[str]):
        """
        Args:
            cfg:          Training configuration
            label_names:  List of label names (index = label int, 0 = blank)
        """
        self.cfg          = cfg
        self.device       = cfg.get_device()
        self.label_names  = label_names
        self.decoder      = GreedyCTCDecoder(labels=label_names, blank=0)

        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Trainer on device: {self.device}")

    def build_model(self) -> nn.Module:
        """Instantiate model from architecture name."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.model.motifcaller import build_model
        model = build_model(self.cfg.arch, self.cfg.n_classes)
        model = model.to(self.device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: {self.cfg.arch} | Parameters: {n_params/1e6:.2f}M")
        return model

    def _compute_input_lengths(
        self, model: nn.Module, signal_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute output lengths after CNN downsampling."""
        return model.output_lengths(signal_lengths.to(self.device))

    def _run_batch(
        self,
        model:     nn.Module,
        ctc_loss:  nn.CTCLoss,
        batch,
    ) -> Tuple[torch.Tensor, List[str], List[str]]:
        """
        Forward pass + CTC loss for one batch.

        Returns:
            loss, predicted_transcripts, target_transcripts
        """
        if batch is None:
            return None, [], []

        signals, flat_labels, signal_lengths, label_lengths = batch

        signals      = signals.to(self.device)
        flat_labels  = flat_labels.to(self.device)
        label_lengths = label_lengths.to(self.device)

        # Forward — pass signal_lengths for conformer attention mask
        # (lengths=None is fine for benchmark/causal which don't use attention)
        log_probs = model(signals, signal_lengths.to(self.device))    # (T', B, C)
        input_lengths = self._compute_input_lengths(model, signal_lengths)

        # CTC loss requires: input_lengths > label_lengths
        # Guard against degenerate cases
        valid = input_lengths > label_lengths
        if not valid.all():
            # Skip batches with insufficient output length
            n_invalid = (~valid).sum().item()
            logger.debug(f"Skipping {n_invalid} samples with output < label length")
            return None, [], []

        # CTC loss is computed on CPU for compatibility across backends
        loss = ctc_loss(
            log_probs.cpu(),
            flat_labels.cpu(),
            input_lengths.cpu(),
            label_lengths.cpu()
        )

        # Decode predictions for metrics
        pred_transcripts = []
        tgt_transcripts  = []
        offset = 0
        for i, ll in enumerate(label_lengths.cpu()):
            ll = ll.item()
            # Greedy decode for batch item i
            emission = log_probs[:, i, :].unsqueeze(1)  # (T', 1, C)
            pred = self.decoder(emission)
            pred_transcripts.append(pred)

            # Target
            tgt = [self.label_names[l.item()]
                   for l in flat_labels[offset:offset+ll]]
            tgt_transcripts.append(tgt)
            offset += ll

        return loss, pred_transcripts, tgt_transcripts

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        resume_from:  Optional[Path] = None,
    ) -> nn.Module:
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader
            val_loader:   Validation DataLoader
            resume_from:  Optional checkpoint path to resume from

        Returns:
            Best model (by validation loss)
        """
        model    = self.build_model()
        ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = CosineWarmupScheduler(
            optimizer, self.cfg.warmup_epochs, self.cfg.n_epochs
        )

        start_epoch   = 0
        best_val_loss = float("inf")
        patience_cnt  = 0

        if resume_from and resume_from.exists():
            ckpt = torch.load(resume_from, map_location=self.device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            logger.info(f"Resumed from epoch {start_epoch}, best_val={best_val_loss:.4f}")

        for epoch in range(start_epoch, self.cfg.n_epochs):
            t0 = time.time()

            # ── Training epoch ─────────────────────────────────────────
            model.train()
            train_losses, train_accs = [], []

            for batch_idx, batch in enumerate(train_loader):
                if batch is None:
                    continue
                optimizer.zero_grad()
                loss, preds, targets = self._run_batch(model, ctc_loss, batch)
                if loss is None:
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                optimizer.step()

                train_losses.append(loss.item())
                if targets:
                    acc = motif_accuracy(preds, targets)
                    train_accs.append(acc)

                if batch_idx % self.cfg.log_interval == 0:
                    logger.info(
                        f"  Epoch {epoch} batch {batch_idx}/{len(train_loader)} "
                        f"loss={loss.item():.4f}"
                    )

            scheduler.step()

            # ── Validation epoch ────────────────────────────────────────
            model.eval()
            val_losses, val_accs, val_edrs = [], [], []

            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    loss, preds, targets = self._run_batch(model, ctc_loss, batch)
                    if loss is None:
                        continue
                    val_losses.append(loss.item())
                    if targets:
                        val_accs.append(motif_accuracy(preds, targets))
                        val_edrs.append(edit_distance_ratio(preds, targets))

            # ── Epoch summary ───────────────────────────────────────────
            avg_train = sum(train_losses) / max(1, len(train_losses))
            avg_val   = sum(val_losses)   / max(1, len(val_losses))
            avg_acc   = sum(val_accs)     / max(1, len(val_accs))
            avg_edr   = sum(val_edrs)     / max(1, len(val_edrs))
            elapsed   = time.time() - t0
            lr_now    = optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch:3d} | "
                f"train_loss={avg_train:.4f} val_loss={avg_val:.4f} | "
                f"val_acc={avg_acc:.3f} val_edr={avg_edr:.3f} | "
                f"lr={lr_now:.2e} | {elapsed:.0f}s"
            )

            # ── Checkpoint ─────────────────────────────────────────────
            is_best = avg_val < best_val_loss
            if is_best:
                best_val_loss = avg_val
                patience_cnt  = 0
                torch.save(
                    {
                        "epoch":         epoch,
                        "model":         model.state_dict(),
                        "optimizer":     optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "config":        self.cfg,
                    },
                    self.cfg.checkpoint_dir / "best_model.pt",
                )
                logger.info(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
            else:
                patience_cnt += 1
                if patience_cnt >= self.cfg.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Save latest checkpoint (for resuming)
            torch.save(
                {
                    "epoch":         epoch,
                    "model":         model.state_dict(),
                    "optimizer":     optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config":        self.cfg,
                },
                self.cfg.checkpoint_dir / "latest.pt",
            )

        # Load best model before returning
        best_ckpt = torch.load(
            self.cfg.checkpoint_dir / "best_model.pt",
            map_location=self.device
        )
        model.load_state_dict(best_ckpt["model"])
        logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
        return model
