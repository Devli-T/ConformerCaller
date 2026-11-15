"""
Dataset — PyTorch Dataset for ConformerCaller Training
====================================================
Pairs raw nanopore signals (from FAST5) with ordered payload motif
label sequences (from master_db via LabelBuilder).

Design principles:
  - Signals can be streamed from disk or loaded via pre-extracted cache
  - A lightweight index (read_id → fast5 path + labels) is kept in memory
  - Supports on-the-fly loading or pre-extracted signal caching
"""

import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .fast5_reader import Fast5Reader
from .label_builder import LabelRecord
from .motif_library import MotifLibrary

logger = logging.getLogger(__name__)


class MotifDatasetIndex:
    """
    Lightweight index mapping read_id → (fast5_path, LabelRecord).
    Created once from label records + FAST5 file index, then reused.
    """

    def __init__(
        self,
        label_records:   Dict[str, LabelRecord],
        fast5_index:     Dict[str, Path],
    ):
        """
        Args:
            label_records: {read_id: LabelRecord} from LabelBuilder
            fast5_index:   {read_id: fast5_path} from build_fast5_index()
        """
        # Keep only reads that exist in both
        common = set(label_records.keys()) & set(fast5_index.keys())
        logger.info(f"Label records: {len(label_records):,}, "
                    f"FAST5 index:   {len(fast5_index):,}, "
                    f"Intersection:  {len(common):,}")

        self.entries: List[Tuple[str, Path, LabelRecord]] = [
            (rid, fast5_index[rid], label_records[rid])
            for rid in sorted(common)
        ]

    def __len__(self):
        return len(self.entries)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "MotifDatasetIndex":
        with open(path, "rb") as f:
            return pickle.load(f)

    def split(
        self,
        val_fraction:  float = 0.15,
        test_fraction: float = 0.10,
        seed:          int   = 42,
    ) -> Tuple["MotifDatasetIndex", "MotifDatasetIndex", "MotifDatasetIndex"]:
        """
        Split into train/val/test by shuffling entries.

        Splits are done at the read level (not file level) for simplicity.
        For a production system, split by FAST5 file to avoid data leakage
        (reads from the same flow cell run may be correlated).
        """
        indices = list(range(len(self.entries)))
        rng = random.Random(seed)
        rng.shuffle(indices)

        n = len(indices)
        n_test = int(n * test_fraction)
        n_val  = int(n * val_fraction)

        test_idx  = indices[:n_test]
        val_idx   = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

        def subset(idxs):
            ds = MotifDatasetIndex.__new__(MotifDatasetIndex)
            ds.entries = [self.entries[i] for i in idxs]
            return ds

        return subset(train_idx), subset(val_idx), subset(test_idx)


    def split_by_manifest(
        self,
        train_ids: set,
        val_ids:   set,
        test_ids:  set,
    ) -> Tuple["MotifDatasetIndex", "MotifDatasetIndex", "MotifDatasetIndex"]:
        """
        Split using fixed read-ID lists (e.g. the split published with the
        paper in ``splits/``). Reads that are in none of the lists are dropped,
        so this also works when only part of the FAST5 data is available.
        """
        buckets = {"train": [], "val": [], "test": []}
        for e in self.entries:
            rid = e[0]
            if rid in test_ids:
                buckets["test"].append(e)
            elif rid in val_ids:
                buckets["val"].append(e)
            elif rid in train_ids:
                buckets["train"].append(e)

        def subset(entries):
            ds = MotifDatasetIndex.__new__(MotifDatasetIndex)
            ds.entries = entries
            return ds

        return subset(buckets["train"]), subset(buckets["val"]), subset(buckets["test"])


class MotifSignalDataset(Dataset):
    """
    PyTorch Dataset for (signal, label_sequence) pairs.

    Each item is:
      signal:  Float32 tensor of shape (1, signal_length) — raw normalised signal
      labels:  LongTensor of shape (n_labels,) — ordered payload motif indices
    """

    def __init__(
        self,
        index:           MotifDatasetIndex,
        reader:          Fast5Reader,
        max_signal_len:  int = 80_000,
        signals_cache:   Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Args:
            index:          Dataset index (read_id → fast5_path + labels)
            reader:         Fast5Reader for on-the-fly fallback
            max_signal_len: Truncate signals to this length
            signals_cache:  Pre-extracted signals dict {read_id: signal_array}.
                            When provided, skips FAST5 I/O entirely for a ~100x speedup.
                            Load with: np.load("signals.npz", allow_pickle=False)
        """
        self.index          = index
        self.reader         = reader
        self.max_signal_len = max_signal_len
        self.signals_cache  = signals_cache  # None → fall back to FAST5
        if signals_cache is not None:
            logger.info(f"Using pre-extracted signals cache ({len(signals_cache):,} signals)")

    def __len__(self):
        return len(self.index.entries)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        read_id, fast5_path, record = self.index.entries[idx]

        # --- Fast path: use pre-extracted cache ---
        if self.signals_cache is not None:
            signal = self.signals_cache.get(read_id)
            if signal is None:
                return None
        else:
            # --- Slow path: on-the-fly FAST5 read (for small datasets / inference) ---
            signal_dict = self.reader.read_all(fast5_path)
            signal = signal_dict.get(read_id)
            if signal is None:
                for k, v in signal_dict.items():
                    if read_id in k or k in read_id:
                        signal = v
                        break
            if signal is None:
                logger.debug(f"Signal not found for read_id={read_id}")
                return None

        # Truncate if necessary
        if len(signal) > self.max_signal_len:
            signal = signal[:self.max_signal_len]

        signal_tensor = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0)  # (1, T)
        label_tensor  = torch.tensor(record.payload_labels, dtype=torch.long)

        return signal_tensor, label_tensor


def motif_collate_fn(
    batch: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Collate function for variable-length signals and label sequences.

    Pads signals to the longest in the batch; pads labels to the longest.

    Returns:
        signals:       (B, 1, T_max) padded signals
        labels:        (B * L_max,) flat label tensor for CTC
        input_lengths: (B,) actual signal lengths (after CNN downsampling — computed later)
        label_lengths: (B,) actual label sequence lengths
    """
    # Filter None items (failed signal loads)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    signals, labels = zip(*batch)

    # Pad signals (right-pad with 0)
    signal_lengths = torch.tensor([s.shape[-1] for s in signals], dtype=torch.long)
    max_len = signal_lengths.max().item()
    padded_signals = torch.zeros(len(signals), 1, max_len)
    for i, s in enumerate(signals):
        padded_signals[i, 0, :s.shape[-1]] = s[0]

    # Labels: CTC expects flat concatenation
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    flat_labels   = torch.cat(labels)

    return padded_signals, flat_labels, signal_lengths, label_lengths


def build_dataloaders(
    train_index:   MotifDatasetIndex,
    val_index:     MotifDatasetIndex,
    test_index:    MotifDatasetIndex,
    batch_size:    int = 8,
    num_workers:   int = 0,   # Number of workers for data loading
    max_signal_len: int = 80_000,
    signals_cache: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders.

    Args:
        signals_cache: Pre-extracted {read_id: signal} dict for fast I/O.
                       If None, falls back to on-the-fly FAST5 reading.
    """
    reader = Fast5Reader(normalise=True)

    train_ds = MotifSignalDataset(train_index, reader, max_signal_len, signals_cache)
    val_ds   = MotifSignalDataset(val_index,   reader, max_signal_len, signals_cache)
    test_ds  = MotifSignalDataset(test_index,  reader, max_signal_len, signals_cache)

    kwargs = dict(
        batch_size  = batch_size,
        collate_fn  = motif_collate_fn,
        num_workers = num_workers,
        pin_memory  = False,
    )

    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )
