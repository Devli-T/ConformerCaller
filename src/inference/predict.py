"""
Inference — Run ConformerCaller on FAST5 Files
===============================================
Given a directory of FAST5 files, predict the payload motif sequence
for each read and output a results CSV.

Usage:
    # Greedy decoding (standard)
    python src/inference/predict.py \
        --checkpoint checkpoints/conformer/best_model.pt \
        --fast5-dir /path/to/data_root/FAST5 \
        --library /path/to/data_root/Reference/completed_library.csv \
        --output results/predictions.csv

    # Confidence-filtered decoding (e.g. tau >= 0.70)
    python src/inference/predict.py \
        --checkpoint checkpoints/conformer/best_model.pt \
        --fast5-dir /path/to/data_root/FAST5 \
        --library /path/to/data_root/Reference/completed_library.csv \
        --threshold 0.70 \
        --output results/predictions_filtered.csv

Output CSV columns:
    read_id, predicted_motifs, n_motifs, confidence
"""

import os
# Enable CPU fallback for operations without MPS kernels on Apple Silicon
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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
    """Load a trained model from checkpoint, supporting various checkpoint formats."""
    import vbz_h5py_plugin  # noqa: register VBZ filter
    from src.model.motifcaller import build_model

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Resolve architecture and number of classes from checkpoint
    if "config" in ckpt:
        cfg = ckpt["config"]
        arch = getattr(cfg, "arch", "conformer")
        n_classes = getattr(cfg, "n_classes", 65)
    else:
        arch = ckpt.get("arch", "conformer")
        n_classes = ckpt.get("n_classes", 65)

    model = build_model(arch, n_classes=n_classes)

    # Extract state dict
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", None))
    val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)
    logger.info(f"Loaded {arch} model from epoch {epoch} (val_loss={val_loss_str})")

    return model, arch


def predict_batch(
    model,
    signals: List[np.ndarray],
    read_ids: List[str],
    label_names: List[str],
    device: torch.device,
    arch: str,
    max_signal_len: int = 20_000,
    threshold: Optional[float] = None,
) -> List[dict]:
    """Run inference on a batch of pre-normalised signals.

    Args:
        threshold: Optional confidence threshold (e.g. 0.70). If specified,
                   uses decoder.decode_with_confidence to filter motif emissions.
    """
    from src.training.ctc_decoder import GreedyCTCDecoder

    decoder = GreedyCTCDecoder(labels=label_names, blank=0)

    # Trim to max length and determine batch maximum length
    signals_trimmed = [s[:max_signal_len] for s in signals]
    max_len = max(len(s) for s in signals_trimmed)
    signal_lengths = torch.tensor([len(s) for s in signals_trimmed], dtype=torch.long)

    padded = torch.zeros(len(signals_trimmed), 1, max_len, dtype=torch.float32)
    for i, s in enumerate(signals_trimmed):
        padded[i, 0, :len(s)] = torch.from_numpy(s.astype(np.float32))
    padded = padded.to(device)

    with torch.no_grad():
        if arch == "conformer":
            log_probs = model(padded, signal_lengths.to(device))
        else:
            log_probs = model(padded)

        # Confidence: average of maximum frame probabilities
        probs = torch.exp(log_probs)  # (T', B, C)
        max_probs = probs.max(dim=-1).values  # (T', B)
        confidences = max_probs.mean(dim=0).cpu().tolist()  # (B,)

    results = []
    for i, read_id in enumerate(read_ids):
        emission = log_probs[:, i, :].unsqueeze(1)  # (T', 1, C)

        if threshold is not None:
            decoded = decoder.decode_with_confidence(emission, threshold=threshold)
        else:
            decoded = decoder(emission)

        # Unpack if wrapped in outer batch list
        if isinstance(decoded, list) and len(decoded) == 1 and isinstance(decoded[0], list):
            decoded = decoded[0]
        elif not isinstance(decoded, list):
            decoded = [decoded] if decoded else []

        results.append({
            "read_id":          read_id,
            "predicted_motifs": "|".join(decoded),
            "n_motifs":         len(decoded),
            "confidence":       round(confidences[i], 4),
        })

    return results


def resolve_library_path(explicit_path: Optional[Path]) -> Path:
    """Locate completed_library.csv (--library, else <DATA_ROOT>/Reference/)."""
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Specified library file not found: {explicit_path}")
        return explicit_path

    from config import COMPLETED_LIBRARY_PATH
    if COMPLETED_LIBRARY_PATH.exists():
        return COMPLETED_LIBRARY_PATH

    raise FileNotFoundError(
        "Could not locate completed_library.csv.\n"
        "Pass --library /path/to/completed_library.csv, or set "
        "CONFORMERCALLER_DATA_ROOT (see scripts/prepare_data.py)."
    )


def list_fast5_files(fast5_dir: Path) -> List[Path]:
    """List .fast5 files, ignoring hidden files (e.g. macOS ._ sidecars on exFAT)."""
    if not fast5_dir.exists():
        raise FileNotFoundError(f"FAST5 directory not found: {fast5_dir}")
    return sorted(p for p in fast5_dir.glob("*.fast5") if not p.name.startswith("."))


def predict_fast5_dir(
    fast5_dir:        Path,
    checkpoint_path:  Path,
    output_path:      Path,
    library_path:     Optional[Path] = None,
    batch_size:       int  = 16,
    max_signal_len:   int  = 20_000,
    threshold:        Optional[float] = None,
    max_reads:        Optional[int] = None,
):
    """Main inference function — processes FAST5 files and writes predictions to CSV."""
    import vbz_h5py_plugin  # noqa
    import h5py
    from tqdm import tqdm
    from config import DEVICE
    from src.data.motif_library import MotifLibrary
    from src.data.fast5_reader import mad_normalise

    # Resolve paths
    library_path = resolve_library_path(library_path)

    device = DEVICE
    logger.info(f"Device: {device}")
    logger.info(f"Using motif library: {library_path}")
    if threshold is not None:
        logger.info(f"Confidence threshold: {threshold:.2f}")

    lib = MotifLibrary(library_path)
    model, arch = load_model(checkpoint_path, device)

    fast5_files = sorted(p for p in fast5_dir.glob("*.fast5") if not p.name.startswith("."))  # skip macOS ._ sidecar files
    if not fast5_files:
        raise FileNotFoundError(f"No .fast5 files found in {fast5_dir}")
    logger.info(f"Found {len(fast5_files)} FAST5 file(s)")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_reads = 0
    done = False

    with open(output_path, "w", newline="") as csvfile:
        fieldnames = ["read_id", "predicted_motifs", "n_motifs", "confidence"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for f5_path in tqdm(fast5_files, desc="Processing FAST5 files"):
            if done:
                break

            batch_ids  = []
            batch_sigs = []

            try:
                with h5py.File(f5_path, "r") as f:
                    for rk in f.keys():
                        read_id = rk.replace("read_", "")
                        try:
                            raw = f[rk]["Raw"]["Signal"][:]
                        except KeyError:
                            continue

                        norm = mad_normalise(raw)
                        batch_ids.append(read_id)
                        batch_sigs.append(norm)

                        if len(batch_ids) >= batch_size:
                            rows = predict_batch(
                                model, batch_sigs, batch_ids,
                                lib.label_names, device, arch,
                                max_signal_len, threshold,
                            )
                            writer.writerows(rows)
                            csvfile.flush()
                            total_reads += len(rows)
                            batch_ids, batch_sigs = [], []

                            if max_reads is not None and total_reads >= max_reads:
                                done = True
                                break

            except Exception as e:
                logger.warning(f"Error reading {f5_path.name}: {e}")
                continue

            # Flush residual batch
            if batch_ids and not done:
                rows = predict_batch(
                    model, batch_sigs, batch_ids,
                    lib.label_names, device, arch,
                    max_signal_len, threshold,
                )
                writer.writerows(rows)
                csvfile.flush()
                total_reads += len(rows)
                if max_reads is not None and total_reads >= max_reads:
                    done = True

    logger.info(f"Completed: Predicted {total_reads:,} reads -> {output_path}")


def main():
    p = argparse.ArgumentParser(description="Run ConformerCaller inference on FAST5 files")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to trained model checkpoint (.pt)")
    p.add_argument("--fast5-dir",  type=Path, required=True,
                   help="Directory containing .fast5 files")
    p.add_argument("--library",    type=Path, default=None,
                   help="Path to completed_library.csv (motif catalog). "
                        "If omitted, resolved via CONFORMERCALLER_DATA_ROOT or local default.")
    p.add_argument("--output",     type=Path,
                   default=Path("results/predictions.csv"),
                   help="Output CSV file path (default: results/predictions.csv)")
    p.add_argument("--threshold",  type=float, default=None,
                   help="Confidence threshold for filtering motifs (e.g. 0.70). "
                        "Leave unset for greedy decoding on all frames.")
    p.add_argument("--batch-size",     type=int, default=16,
                   help="Inference batch size (default: 16)")
    p.add_argument("--max-signal-len", type=int, default=20_000,
                   help="Maximum signal samples per read (default: 20000)")
    p.add_argument("--max-reads",      type=int, default=None,
                   help="Stop after processing N reads (useful for quick testing)")

    args = p.parse_args()

    predict_fast5_dir(
        fast5_dir       = args.fast5_dir,
        checkpoint_path = args.checkpoint,
        output_path     = args.output,
        library_path    = args.library,
        batch_size      = args.batch_size,
        max_signal_len  = args.max_signal_len,
        threshold       = args.threshold,
        max_reads       = args.max_reads,
    )


if __name__ == "__main__":
    main()
