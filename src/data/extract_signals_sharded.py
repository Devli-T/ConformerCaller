"""
Extract Signals in Shards — Memory-Safe Signal Extraction
==========================================================
Extracts raw nanopore signals into manageable shards of N reads each to ensure memory-efficient preprocessing.

Each shard is saved as an independent .npz file on the external volume.
This allows:
  1. Predictable, low-memory extraction
  2. Streamed / distributed data loading for large-scale training
  3. Scalable chunking across cloud or high-performance compute clusters

Usage:
    # Test reads only (≈ 7 shards, enough to reproduce the paper evaluation)
    python src/data/extract_signals_sharded.py \
        --index-dir $CONFORMERCALLER_DATA_ROOT/processed/dataset_index \
        --splits test

    # Everything (needed for training; ~27 GB of shards for the full dataset)
    python src/data/extract_signals_sharded.py \
        --index-dir $CONFORMERCALLER_DATA_ROOT/processed/dataset_index

    # Resume an interrupted run
    python src/data/extract_signals_sharded.py --index-dir ... --start-shard 5

Shards are written to <index-dir>/signal_shards/ together with
shard_split_ids.json (which split each read in each shard belongs to — used
by the Colab training notebook).
"""

import argparse
import gc
import json
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import vbz_h5py_plugin  # noqa: must be before h5py
except ImportError:
    pass
import h5py

from src.data.fast5_reader import mad_normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MAX_SIGNAL_LEN = 20_000
MIN_SIGNAL_LEN = 1_000


def extract_shards(
    index_dir:   Path,
    out_dir:     Path,
    shard_size:  int  = 10_000,
    start_shard: int  = 0,
    splits:      list = None,
):
    """
    Extract signals into shards stored on the external volume.

    Memory peak = (shard_size × avg_signal_len × 4 bytes)
               ≈ 10,000 × 10,000 × 4 = 400MB per shard, keeping preprocessing memory predictable.
    """
    if splits is None:
        splits = ["train", "val", "test"]

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Collect all (read_id, fast5_path) pairs ───────────────────────
    logger.info("Loading dataset index entries (streaming, no full index in RAM)…")
    all_entries = []  # list of (read_id, fast5_path)
    split_of = {}     # read_id -> split name

    for split in splits:
        pkl = index_dir / f"dataset_{split}.pkl"
        if not pkl.exists():
            logger.warning(f"  Skipping missing {pkl.name}")
            continue
        with open(pkl, "rb") as f:
            idx = pickle.load(f)
        for rid, f5path, _ in idx.entries:
            all_entries.append((rid, f5path))
            split_of[rid] = split
        logger.info(f"  {split}: {len(idx.entries):,} entries")
        del idx
        gc.collect()

    logger.info(f"Total unique reads: {len(all_entries):,}")

    # Deduplicate (reads appear in only one split, but just in case)
    seen = set()
    unique_entries = []
    for e in all_entries:
        if e[0] not in seen:
            seen.add(e[0])
            unique_entries.append(e)
    del all_entries, seen
    gc.collect()

    n_total = len(unique_entries)
    n_shards = (n_total + shard_size - 1) // shard_size
    logger.info(f"Creating {n_shards} shards of {shard_size:,} reads each")

    # ── Step 2: Process shard by shard ────────────────────────────────────────
    for shard_idx in range(start_shard, n_shards):
        shard_path = out_dir / f"signals_shard_{shard_idx:04d}.npz"
        if shard_path.exists():
            logger.info(f"Shard {shard_idx}/{n_shards-1} already exists, skipping")
            continue

        start = shard_idx * shard_size
        end   = min(start + shard_size, n_total)
        shard_entries = unique_entries[start:end]

        logger.info(f"Shard {shard_idx}/{n_shards-1}: reads {start:,}-{end:,}")

        # Group reads by FAST5 file to open each file only once
        f5_to_reads: dict[Path, list[str]] = defaultdict(list)
        for rid, f5path in shard_entries:
            f5_to_reads[f5path].append(rid)

        shard_signals: dict[str, np.ndarray] = {}
        n_skipped = 0

        for f5path, read_ids in tqdm(
            f5_to_reads.items(),
            desc=f"  Shard {shard_idx}",
            leave=False,
        ):
            try:
                with h5py.File(f5path, "r") as f:
                    for rid in read_ids:
                        key = f"read_{rid}"
                        if key not in f:
                            n_skipped += 1
                            continue
                        raw = f[key]["Raw"]["Signal"][:]
                        if len(raw) < MIN_SIGNAL_LEN:
                            n_skipped += 1
                            continue
                        if len(raw) > MAX_SIGNAL_LEN:
                            raw = raw[:MAX_SIGNAL_LEN]
                        shard_signals[rid] = mad_normalize(raw).astype(np.float32)
            except Exception as e:
                logger.warning(f"  Error reading {f5path.name}: {e}")

        logger.info(
            f"  Shard {shard_idx}: {len(shard_signals):,} signals "
            f"({n_skipped} skipped) → {shard_path.name}"
        )

        np.savez(shard_path, **shard_signals)
        shard_mb = shard_path.stat().st_size / 1e6
        logger.info(f"  Saved {shard_path.name} ({shard_mb:.0f} MB)")

        # Critical: free memory before next shard
        del shard_signals, f5_to_reads
        gc.collect()

    # ── Step 3: Write manifest ─────────────────────────────────────────────────
    shard_files = sorted(out_dir.glob("signals_shard_*.npz"))
    manifest_path = out_dir / "manifest.txt"
    with open(manifest_path, "w") as f:
        f.write(f"total_reads={n_total}\n")
        f.write(f"n_shards={len(shard_files)}\n")
        f.write(f"shard_size={shard_size}\n")
        for sf in shard_files:
            f.write(f"{sf.name}\n")

    # ── Step 4: shard_split_ids.json (split membership of every shard) ────────
    shard_split_ids = {}
    for sf in shard_files:
        with np.load(sf) as z:
            keys = list(z.files)
        per_split = {sp: [k for k in keys if split_of.get(k) == sp] for sp in ("train", "val", "test")}
        n_assigned = sum(len(v) for v in per_split.values())
        shard_split_ids[sf.name] = {
            "total": len(keys),
            "train": len(per_split["train"]),
            "val":   len(per_split["val"]),
            "test":  len(per_split["test"]),
            "unassigned": len(keys) - n_assigned,
            "train_ids": per_split["train"],
            "val_ids":   per_split["val"],
            "test_ids":  per_split["test"],
        }
    with open(out_dir / "shard_split_ids.json", "w") as f:
        json.dump(shard_split_ids, f)
    logger.info(f"  Split membership: {out_dir / 'shard_split_ids.json'}")

    total_mb = sum(sf.stat().st_size for sf in shard_files) / 1e6
    logger.info(f"\n✓ Extraction complete: {len(shard_files)} shards, {total_mb:.0f} MB total")
    logger.info(f"  Manifest: {manifest_path}")
    return out_dir


def main():
    p = argparse.ArgumentParser(description="Extract signals to shards (memory-safe)")
    p.add_argument("--index-dir",  type=Path, required=True,
                   help="Directory with dataset_train.pkl, dataset_val.pkl, dataset_test.pkl")
    p.add_argument("--out-dir",    type=Path,
                   default=None,
                   help="Output shard directory (default: <index-dir>/signal_shards)")
    p.add_argument("--shard-size", type=int, default=10_000,
                   help="Reads per shard (default: 10000 ≈ 400MB/shard)")
    p.add_argument("--start-shard", type=int, default=0,
                   help="Resume from this shard number")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    out_dir = args.out_dir or (args.index_dir / "signal_shards")
    extract_shards(args.index_dir, out_dir, args.shard_size, args.start_shard, args.splits)


if __name__ == "__main__":
    main()
