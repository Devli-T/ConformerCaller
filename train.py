"""
train.py — Main ConformerCaller Training Entry Point
=================================================
Usage:
    python3.12 train.py --arch conformer --epochs 100
    python3.12 train.py --arch benchmark --epochs 100  # baseline
    python3.12 train.py --smoke-test                   # quick 5-batch test

Checkpoints and results are saved locally.
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

import torch
from config import (
    COMPLETED_LIBRARY_PATH, PROCESSED_DATA_DIR,
    MODEL_SAVE_DIR, RESULTS_DIR, DEVICE,
    DEFAULT_TRAIN_CONFIG,
)
from src.data.motif_library import MotifLibrary
from src.data.dataset import MotifDatasetIndex, build_dataloaders
from src.training.trainer import Trainer, TrainConfig


def parse_args():
    p = argparse.ArgumentParser(description="Train a ConformerCaller model")
    p.add_argument("--arch",         default="conformer",
                   choices=["benchmark", "conformer", "causal"],
                   help="Model architecture (default: conformer)")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--min-cycles",   type=int,   default=4,
                   help="Minimum payload cycles required per training sample")
    p.add_argument("--dataset-dir",  type=Path,
                   default=PROCESSED_DATA_DIR / "dataset_index",
                   help="Directory with pre-built dataset index files")
    p.add_argument("--signals-cache", type=Path, default=None,
                   help="Path to signals.npz (pre-extracted). Auto-detected in dataset-dir if not given.")
    p.add_argument("--checkpoint-dir", type=Path,
                   default=MODEL_SAVE_DIR,
                   help="Where to save model checkpoints")
    p.add_argument("--resume",       type=Path,  default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--smoke-test",   action="store_true",
                   help="Run 5 batches only to verify the pipeline works")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load motif library ───────────────────────────────────────────────────
    logger.info("Loading motif library…")
    lib = MotifLibrary(COMPLETED_LIBRARY_PATH)
    logger.info(f"  {lib}")

    # ── Load dataset index ───────────────────────────────────────────────────
    dataset_dir = args.dataset_dir
    if not args.smoke_test:
        logger.info("Loading dataset index from disk…")
        if not (dataset_dir / "dataset_train.pkl").exists():
            logger.error(
                f"Dataset index not found at {dataset_dir}.\n"
                "Run preprocessing first:\n"
                "  python3 src/data/build_training_set.py"
            )
            sys.exit(1)

        train_idx = MotifDatasetIndex.load(dataset_dir / "dataset_train.pkl")
        val_idx   = MotifDatasetIndex.load(dataset_dir / "dataset_val.pkl")
        test_idx  = MotifDatasetIndex.load(dataset_dir / "dataset_test.pkl")
        logger.info(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")

        # ── Load signals cache (fast path) ───────────────────────────────────
        import numpy as np
        signals_cache = None
        cache_path = args.signals_cache or (dataset_dir / "signals.npz")
        if cache_path.exists():
            logger.info(f"Loading signals cache from {cache_path} …")
            npz = np.load(str(cache_path), allow_pickle=False)
            signals_cache = {k: npz[k] for k in npz.files}
            logger.info(f"  Loaded {len(signals_cache):,} pre-extracted signals")
        else:
            logger.warning(
                f"No signals cache found at {cache_path}.\n"
                "Training will use slow on-the-fly FAST5 reading.\n"
                "Run:  python3 src/data/extract_signals.py --index-dir <dataset-dir>"
            )
    else:
        # Smoke test: build a tiny index from first FAST5 file
        logger.info("SMOKE TEST MODE — using tiny subset of data")
        from config import FAST5_DIR, MASTER_DB_PATH
        from src.data.fast5_reader import Fast5Reader
        from src.data.label_builder import LabelBuilder

        f5_files = sorted(FAST5_DIR.glob("*.fast5"))[:1]
        logger.info(f"  Using FAST5 file: {f5_files[0].name}")

        # Get read IDs from this file
        import vbz_h5py_plugin, h5py
        with h5py.File(f5_files[0], "r") as f:
            read_ids_in_file = {k.replace("read_", "") for k in list(f.keys())[:50]}
        logger.info(f"  Checking {len(read_ids_in_file)} reads from first 50 in file")

        # Build minimal label records
        builder = LabelBuilder(lib, min_payload_cycles=1, forward_strand_only=False)
        label_records = builder.build(MASTER_DB_PATH, max_reads=500)

        # Filter to those in file
        label_records = {rid: rec for rid, rec in label_records.items()
                         if rid in read_ids_in_file}
        logger.info(f"  Found {len(label_records)} matching reads with labels")

        if len(label_records) < 5:
            # Relax filter — take any reads we can match
            builder2 = LabelBuilder(lib, min_payload_cycles=1, forward_strand_only=False)
            label_records_all = builder2.build(MASTER_DB_PATH, max_reads=5000)
            label_records = {rid: rec for rid, rec in label_records_all.items()
                             if rid in read_ids_in_file}
            logger.info(f"  After relaxed filter: {len(label_records)} reads")

        fast5_index = {rid: f5_files[0] for rid in label_records}
        full_idx = MotifDatasetIndex(label_records, fast5_index)
        n = len(full_idx)

        if n == 0:
            logger.error("No reads matched between FAST5 and label records. "
                         "Run build_training_set.py first for a proper run.")
            sys.exit(1)

        # Use all for train in smoke test
        train_idx = full_idx
        val_idx   = full_idx
        test_idx  = full_idx

    # ── Build DataLoaders ────────────────────────────────────────────────────
    _signals = signals_cache if not args.smoke_test else None
    train_loader, val_loader, test_loader = build_dataloaders(
        train_idx, val_idx, test_idx,
        batch_size    = args.batch_size,
        signals_cache = _signals,
    )

    if args.smoke_test:
        # Limit to 5 batches
        from itertools import islice
        original_loader = train_loader
        class LimitedLoader:
            def __init__(self, loader, n): self.loader = loader; self.n = n
            def __iter__(self): return islice(iter(self.loader), self.n)
            def __len__(self): return min(self.n, len(self.loader))
        train_loader = LimitedLoader(train_loader, 5)
        val_loader   = LimitedLoader(val_loader,   2)

    # ── Training config ──────────────────────────────────────────────────────
    arch_dir = args.checkpoint_dir / args.arch
    arch_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = TrainConfig(
        arch            = args.arch,
        n_classes       = lib.n_classes,
        n_epochs        = 2 if args.smoke_test else args.epochs,
        batch_size      = args.batch_size,
        learning_rate   = args.lr,
        checkpoint_dir  = arch_dir,
        patience        = 5 if args.smoke_test else DEFAULT_TRAIN_CONFIG["patience"],
    )

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = Trainer(train_cfg, label_names=lib.label_names)
    best_model = trainer.train(
        train_loader = train_loader,
        val_loader   = val_loader,
        resume_from  = args.resume,
    )

    logger.info(f"✓ Training complete. Best model at: {arch_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
