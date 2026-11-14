"""
Build Training Set — One-Time Preprocessing Pipeline
=====================================================
Orchestrates the preprocessing pipeline on the public HELIX01-04 data:
  1. Build per-read payload-motif labels from the master database
     (public HELIX01-04_master_db.txt, or the legacy wide master_db.csv)
  2. Build a FAST5 index (read_id → file path)
  3. Intersect labels with available FAST5 reads
  4. Assign reads to train / val / test and save the dataset indices

By default the train/val/test assignment is the exact split used in the
paper (read-ID lists in ``splits/``), so results are comparable. Reads that
are not in the manifest are dropped; if only part of the FAST5 data has been
extracted you therefore get the corresponding subset of each paper split.
Use ``--random-split`` to draw a fresh seeded random split instead.

Approximate run time on a laptop with the data on an external drive:
  labels ≈ 2–10 min · FAST5 index ≈ 15 min (all 1,131 files) · split < 1 min.

Usage:
    export CONFORMERCALLER_DATA_ROOT=/path/to/data_root
    python src/data/build_training_set.py                 # paper split
    python src/data/build_training_set.py --random-split  # new random split
"""

import argparse
import gzip
import json
import logging
import pickle
import sys
from pathlib import Path

# Allow running as a script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    MASTER_DB_PATH, FAST5_DIR, COMPLETED_LIBRARY_PATH,
    PROCESSED_DATA_DIR, PROJECT_ROOT, DEFAULT_TRAIN_CONFIG,
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


def load_manifest(split_dir: Path):
    """Load the read-ID lists (train/val/test_read_ids.txt.gz) as sets."""
    out = {}
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}_read_ids.txt.gz"
        with gzip.open(path, "rt") as f:
            out[name] = {line.strip() for line in f if line.strip()}
    return out["train"], out["val"], out["test"]


def parse_args():
    p = argparse.ArgumentParser(description="Build ConformerCaller training dataset")
    p.add_argument("--master-db",   type=Path, default=MASTER_DB_PATH,
                   help="HELIX01-04_master_db.txt (public) or master_db.csv")
    p.add_argument("--library",     type=Path, default=COMPLETED_LIBRARY_PATH,
                   help="completed_library.csv")
    p.add_argument("--fast5-dir",   type=Path, default=FAST5_DIR,
                   help="Directory with extracted .fast5 files")
    p.add_argument("--max-reads",   type=int, default=None,
                   help="Cap on labelled reads (for quick tests, e.g. 10000)")
    p.add_argument("--min-cycles",  type=int,
                   default=DEFAULT_TRAIN_CONFIG["min_payload_motifs"],
                   help="Minimum payload cycles per read (default: 4)")
    # 65.7% of reads are minus-strand; both strands are used by default.
    p.add_argument("--forward-only", action="store_true", default=False,
                   help="Use forward-strand reads only (default: both strands)")
    p.add_argument("--split-dir",   type=Path, default=PROJECT_ROOT / "splits",
                   help="Folder with {train,val,test}_read_ids.txt.gz (paper split)")
    p.add_argument("--random-split", action="store_true",
                   help="Ignore the paper split and shuffle with --seed instead")
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

    for label, path in (("master database", args.master_db),
                        ("motif library", args.library),
                        ("FAST5 directory", args.fast5_dir)):
        if not path.exists():
            sys.exit(f"ERROR: {label} not found: {path}\n"
                     "Run scripts/prepare_data.py and set CONFORMERCALLER_DATA_ROOT.")

    # ── 1. Motif library ────────────────────────────────────────────────────
    logger.info("Loading motif library…")
    lib = MotifLibrary(args.library)
    logger.info(f"  {lib}")

    # ── 2. Label records ────────────────────────────────────────────────────
    labels_path = out_dir / "label_records.pkl"
    if labels_path.exists():
        logger.info(f"Loading cached label records from {labels_path}")
        with open(labels_path, "rb") as f:
            label_records = pickle.load(f)
    else:
        logger.info(f"Building label records from {args.master_db.name}…")
        builder = LabelBuilder(
            motif_library       = lib,
            min_payload_cycles  = args.min_cycles,
            forward_strand_only = args.forward_only,
        )
        label_records = builder.build(args.master_db, max_reads=args.max_reads)
        builder.save(label_records, labels_path)

    logger.info(f"Label records: {len(label_records):,}")
    stats = summarise_labels(label_records)
    with open(out_dir / "label_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── 3. FAST5 index (cache is invalidated if the file count changes) ─────
    fast5_files = [p for p in args.fast5_dir.glob("*.fast5") if not p.name.startswith(".")]
    fast5_index_path = out_dir / "fast5_index.pkl"
    meta_path        = out_dir / "fast5_index.meta.json"
    cached_ok = (fast5_index_path.exists() and meta_path.exists()
                 and json.load(open(meta_path)).get("n_files") == len(fast5_files))
    if cached_ok:
        logger.info(f"Loading cached FAST5 index from {fast5_index_path}")
        with open(fast5_index_path, "rb") as f:
            fast5_index = pickle.load(f)
    else:
        logger.info(f"Building FAST5 index over {len(fast5_files)} files…")
        fast5_index = build_fast5_index(args.fast5_dir)
        with open(fast5_index_path, "wb") as f:
            pickle.dump(fast5_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        json.dump({"n_files": len(fast5_files)}, open(meta_path, "w"))
        logger.info(f"Saved FAST5 index: {len(fast5_index):,} reads")

    # ── 4. Dataset index (intersection) ─────────────────────────────────────
    logger.info("Building dataset index (label ∩ FAST5)…")
    full_index = MotifDatasetIndex(label_records, fast5_index)
    logger.info(f"Dataset size: {len(full_index):,} reads")

    # ── 5. Train/val/test split ─────────────────────────────────────────────
    use_manifest = (not args.random_split) and (args.split_dir / "test_read_ids.txt.gz").exists()
    if use_manifest:
        logger.info(f"Using the paper split from {args.split_dir}")
        tr, va, te = load_manifest(args.split_dir)
        train_idx, val_idx, test_idx = full_index.split_by_manifest(tr, va, te)
        n_manifest = len(tr) + len(va) + len(te)
        n_assigned = len(train_idx) + len(val_idx) + len(test_idx)
        logger.info(f"  {n_assigned:,} of {len(full_index):,} reads are in the paper split "
                    f"(paper total: {n_manifest:,})")
        if n_assigned < n_manifest:
            logger.warning(f"  {n_manifest - n_assigned:,} paper-split reads are missing "
                           "(FAST5 files not extracted yet?)")
    else:
        logger.info(f"Random split (seed={args.seed})")
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

    summary = {
        "total": len(full_index),
        "train": len(train_idx),
        "val":   len(val_idx),
        "test":  len(test_idx),
        "split": "paper manifest" if use_manifest else f"random (seed {args.seed})",
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
