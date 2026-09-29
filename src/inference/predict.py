"""
Inference — Run ConformerCaller on FAST5 Files
===============================================
Given a directory of FAST5 files, predict the payload motif sequence
for each read and output a results CSV.

Usage:
    python src/inference/predict.py \
        --checkpoint checkpoints/conformer/best_model.pt \
        --fast5-dir /path/to/fast5 \
        --output results/predictions.csv

Output CSV columns:
    read_id, predicted_motifs, n_motifs, confidence
"""

import os
import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: torch.device):
    """Load a trained model from checkpoint."""
    import vbz_h5py_plugin  # noqa
    from src.model.motifcaller import build_model

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    arch = cfg.arch if hasattr(cfg, "arch") else "conformer"
    n_classes = cfg.n_classes if hasattr(cfg, "n_classes") else 65

    model = build_model(arch, n_classes=n_classes)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded {arch} model from epoch {ckpt.get('epoch', '?')} "
                f"(val_loss={ckpt.get('best_val_loss', '?'):.4f})")
    return model, arch


def predict_batch(
    model,
    signals: List[np.ndarray],
    read_ids: List[str],
    label_names: List[str],
    device: torch.device,
    arch: str,
    max_signal_len: int = 80_000,
) -> List[dict]:
    """Run inference on a batch of signals."""
    from src.training.ctc_decoder import GreedyCTCDecoder

    decoder = GreedyCTCDecoder(labels=label_names, blank=0)

    # Pad and stack signals
    signals_trimmed = [s[:max_signal_len] for s in signals]
    max_len = max(len(s) for s in signals_trimmed)
    signal_lengths = torch.tensor([len(s) for s in signals_trimmed])

    padded = torch.zeros(len(signals_trimmed), 1, max_len)
    for i, s in enumerate(signals_trimmed):
        padded[i, 0, :len(s)] = torch.from_numpy(s.astype(np.float32))
    padded = padded.to(device)

    with torch.no_grad():
        if arch == "conformer":
            log_probs = model(padded, signal_lengths.to(device))
        else:
            log_probs = model(padded)

        # Per-sample confidence: mean of max probabilities
        probs = torch.exp(log_probs)  # (T', B, C)
        max_probs = probs.max(dim=-1).values  # (T', B)
        confidences = max_probs.mean(dim=0).cpu().tolist()  # (B,)

    results = []
    for i, read_id in enumerate(read_ids):
        emission = log_probs[:, i, :].unsqueeze(1)
        decoded = decoder(emission)
        if not isinstance(decoded, list):
            decoded = [decoded] if decoded else []

        results.append({
            "read_id":          read_id,
            "predicted_motifs": "|".join(decoded),
            "n_motifs":         len(decoded),
            "confidence":       round(confidences[i], 4),
        })
    return results


def predict_fast5_dir(
    fast5_dir:       Path,
    checkpoint_path: Path,
    output_path:     Path,
    batch_size:      int  = 16,
    max_signal_len:  int  = 80_000,
):
    """Main inference function — processes all FAST5 files in a directory."""
    import vbz_h5py_plugin  # noqa: must be before h5py
    import h5py
    from tqdm import tqdm
    from config import COMPLETED_LIBRARY_PATH, DEVICE
    from src.data.motif_library import MotifLibrary
    from src.data.fast5_reader import mad_normalize

    # Setup
    device = DEVICE
    logger.info(f"Device: {device}")

    lib = MotifLibrary(COMPLETED_LIBRARY_PATH)
    model, arch = load_model(checkpoint_path, device)

    fast5_files = sorted(fast5_dir.glob("*.fast5"))
    logger.info(f"Found {len(fast5_files)} FAST5 files")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_reads = 0
    with open(output_path, "w", newline="") as csvfile:
        fieldnames = ["read_id", "predicted_motifs", "n_motifs", "confidence"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for f5_path in tqdm(fast5_files, desc="Processing FAST5 files"):
            batch_ids    = []
            batch_sigs   = []

            try:
                with h5py.File(f5_path, "r") as f:
                    for rk in f.keys():
                        read_id = rk.replace("read_", "")
                        raw = f[rk]["Raw"]["Signal"][:]
                        norm = mad_normalize(raw)

                        batch_ids.append(read_id)
                        batch_sigs.append(norm)

                        if len(batch_ids) >= batch_size:
                            rows = predict_batch(
                                model, batch_sigs, batch_ids,
                                lib.label_names, device, arch, max_signal_len
                            )
                            writer.writerows(rows)
                            total_reads += len(rows)
                            batch_ids, batch_sigs = [], []

            except Exception as e:
                logger.warning(f"Error in {f5_path.name}: {e}")
                continue

            # Flush remaining batch
            if batch_ids:
                rows = predict_batch(
                    model, batch_sigs, batch_ids,
                    lib.label_names, device, arch, max_signal_len
                )
                writer.writerows(rows)
                total_reads += len(rows)

    logger.info(f"✓ Predicted {total_reads:,} reads → {output_path}")


def main():
    p = argparse.ArgumentParser(description="Run ConformerCaller inference")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to trained model checkpoint (.pt)")
    p.add_argument("--fast5-dir",  type=Path, required=True,
                   help="Directory containing .fast5 files")
    p.add_argument("--output",     type=Path,
                   default=Path("results/predictions.csv"),
                   help="Output CSV file path")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-signal-len", type=int, default=80_000)
    args = p.parse_args()

    predict_fast5_dir(
        fast5_dir       = args.fast5_dir,
        checkpoint_path = args.checkpoint,
        output_path     = args.output,
        batch_size      = args.batch_size,
        max_signal_len  = args.max_signal_len,
    )


if __name__ == "__main__":
    main()
