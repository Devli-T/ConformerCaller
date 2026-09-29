"""
Build Training Set — One-Time Preprocessing Pipeline
=====================================================
Orchestrates the full preprocessing pipeline:
  1. Load label records from master_db.csv (filtered, sorted)
  2. Build a FAST5 file index (read_id → file path)
  3. Intersect to produce the final dataset index
  4. Save partitioned dataset indices and label records

Run once before training. Takes ~1-2 hours for the full 3.16M-row dataset.

Usage:
    python3 src/data/build_training_set.py [--max-reads N] [--min-cycles N]
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

# Allow running as a script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    MASTER_DB_PATH, FAST5_DIR, COMPLETED_LIBRARY_PATH,
    PROCESSED_DATA_DIR, TRAINING_DATA_DIR,
    DEFAULT_TRAIN_CONFIG,
)
from src.data.motif_library import MotifLibrary
from src.data.label_builder import LabelBuilder, summarise_labels
from src.data.fast5_reader import build_fast5_index
from src.data.dataset import MotifDatasetIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Build ConformerCaller training dataset")
    p.add_argument("--max-reads",   type=int, default=None,
                   help="Cap on reads (for quick tests, e.g. 10000)")
    p.add_argument("--min-cycles",  type=int,
                   default=DEFAULT_TRAIN_CONFIG["min_payload_motifs"],
                   help="Minimum payload cycles per read (default: 4)")
    # Data exploration showed 65.7%% of reads are minus-strand.
    # Default is both strands. Pass --forward-only to restrict.
    p.add_argument("--forward-only", action="store_true", default=False,
                   help="Use forward-strand reads only (default: False — use both strands)")
    p.add_argument("--val-frac",    type=float, default=0.15)
    p.add_argument("--test-frac",   type=float, default=0.10)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--out-dir",     type=Path,
                   default=PROCESSED_DATA_DIR / "dataset_index",
                   help="Output directory for index files")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Motif library ────────────────────────────────────────────────────
    logger.info("Loading motif library…")
    lib = MotifLibrary(COMPLETED_LIBRARY_PATH)
    logger.info(f"  {lib}")

    # ── 2. Label records from master_db ────────────────────────────────────
    labels_path = out_dir / "label_records.pkl"
    if labels_path.exists():
        logger.info(f"Loading cached label records from {labels_path}")
        with open(labels_path, "rb") as f:
            label_records = pickle.load(f)
    else:
        logger.info("Building label records from master_db.csv…")
        builder = LabelBuilder(
            motif_library       = lib,
            min_payload_cycles  = args.min_cycles,
            forward_strand_only = args.forward_only,
        )
        label_records = builder.build(MASTER_DB_PATH, max_reads=args.max_reads)
        builder.save(label_records, labels_path)

    logger.info(f"Label records: {len(label_records):,}")
    stats = summarise_labels(label_records)

    # Save stats
    with open(out_dir / "label_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── 3. FAST5 index ──────────────────────────────────────────────────────
    fast5_index_path = out_dir / "fast5_index.pkl"
    if fast5_index_path.exists():
        logger.info(f"Loading cached FAST5 index from {fast5_index_path}")
        with open(fast5_index_path, "rb") as f:
            fast5_index = pickle.load(f)
    else:
        logger.info("Building FAST5 index (this takes ~5-10 min)…")
        fast5_index = build_fast5_index(FAST5_DIR)
        with open(fast5_index_path, "wb") as f:
            pickle.dump(fast5_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved FAST5 index: {len(fast5_index):,} reads")

    # ── 4. Dataset index (intersection) ────────────────────────────────────
    logger.info("Building dataset index (label ∩ FAST5)…")
    full_index = MotifDatasetIndex(label_records, fast5_index)
    logger.info(f"Dataset size: {len(full_index):,} reads")

    # ── 5. Train/val/test split ─────────────────────────────────────────────
    train_idx, val_idx, test_idx = full_index.split(
        val_fraction  = args.val_frac,
        test_fraction = args.test_frac,
        seed          = args.seed,
    )
    logger.info(f"  Train: {len(train_idx):,}")
    logger.info(f"  Val:   {len(val_idx):,}")
    logger.info(f"  Test:  {len(test_idx):,}")

    # ── 6. Save ─────────────────────────────────────────────────────────────
    full_index.save(out_dir / "dataset_full.pkl")
    train_idx.save(out_dir / "dataset_train.pkl")
    val_idx.save(out_dir / "dataset_val.pkl")
    test_idx.save(out_dir / "dataset_test.pkl")

    # Save summary
    summary = {
        "total": len(full_index),
        "train": len(train_idx),
        "val":   len(val_idx),
        "test":  len(test_idx),
        "min_payload_cycles": args.min_cycles,
        "forward_strand_only": args.forward_only,
        "label_stats": stats,
    }
    with open(out_dir / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("✓ Preprocessing complete.")
    logger.info(f"  Output written to: {out_dir}")
    logger.info(f"  Total usable reads: {len(full_index):,}")


if __name__ == "__main__":
    main()
