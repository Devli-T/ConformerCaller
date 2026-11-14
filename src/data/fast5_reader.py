"""
FAST5 Reader — Raw Nanopore Signal Extraction
==============================================
Reads raw electrical signal data from FAST5 (HDF5) files produced
by Oxford Nanopore sequencers. Supports the multi-read FAST5 format
used by the Helixworks dataset.

Signal normalization uses Median Absolute Deviation (MAD), the standard
approach for nanopore data as used in Guppy and Bonito basecallers.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import logging

# Register ONT VBZ compression filter — MUST be imported before any h5py reads
try:
    import vbz_h5py_plugin  # noqa: F401
except ImportError:
    import warnings
    warnings.warn(
        "vbz_h5py_plugin not found. Install with: pip install vbz-h5py-plugin\n"
        "FAST5 files compressed with VBZ will fail to read without this plugin.",
        RuntimeWarning, stacklevel=2
    )


logger = logging.getLogger(__name__)

# MAD normalization scale factor (converts MAD to approximate std dev)
_MAD_SCALE = 1.4826


def mad_normalize(signal: np.ndarray,
                  clip_min: float = -5.0,
                  clip_max: float = 5.0) -> np.ndarray:
    """
    Normalize a raw nanopore signal using Median Absolute Deviation.

    This is the standard normalization for nanopore data, used by Guppy,
    Bonito.

    Args:
        signal:   Raw integer signal array from FAST5
        clip_min: Lower clip value after normalization
        clip_max: Upper clip value after normalization

    Returns:
        Normalized float32 signal
    """
    signal = signal.astype(np.float32)
    median = np.median(signal)
    mad    = np.median(np.abs(signal - median))
    if mad < 1e-6:
        mad = 1.0  # Avoid divide-by-zero for flat signals
    normalized = (signal - median) / (_MAD_SCALE * mad)
    return np.clip(normalized, clip_min, clip_max).astype(np.float32)


def _find_raw_signal(group: h5py.Group) -> Optional[np.ndarray]:
    """
    Navigate the HDF5 group hierarchy to find the raw signal dataset.

    Handles both single-read and multi-read FAST5 structures.
    Common paths:
      - Raw/Signal  (multi-read FAST5)
      - Raw/Reads/Read_N/Signal  (single-read legacy)
    """
    # Try direct path first (multi-read format)
    if "Raw" in group and "Signal" in group["Raw"]:
        return group["Raw"]["Signal"][:]

    # Try nested path (legacy single-read format)
    if "Raw" in group:
        raw_grp = group["Raw"]
        if "Reads" in raw_grp:
            reads_grp = raw_grp["Reads"]
            for read_key in reads_grp.keys():
                if "Signal" in reads_grp[read_key]:
                    return reads_grp[read_key]["Signal"][:]

    # Recursive search as fallback
    def _search(g, depth=0):
        if depth > 4:
            return None
        for k in g.keys():
            obj = g[k]
            if isinstance(obj, h5py.Dataset) and k == "Signal":
                return obj[:]
            if isinstance(obj, h5py.Group):
                result = _search(obj, depth + 1)
                if result is not None:
                    return result
        return None

    return _search(group)


class Fast5Reader:
    """
    Reads raw signals from multi-read FAST5 files.

    Usage:
        reader = Fast5Reader(normalize=True)
        for read_id, signal in reader.iter_reads("/path/to/file.fast5"):
            process(read_id, signal)
    """

    def __init__(
        self,
        normalize:    bool  = True,
        clip_min:     float = -5.0,
        clip_max:     float = 5.0,
        min_length:   int   = 1000,
        max_length:   int   = 100_000,
        read_ids_filter: Optional[set] = None,
    ):
        """
        Args:
            normalize:        Apply MAD normalization to signals
            clip_min/max:     Clipping bounds after normalization
            min_length:       Skip reads shorter than this (likely fragments)
            max_length:       Skip reads longer than this (memory guard)
            read_ids_filter:  If provided, only yield reads in this set
        """
        self.normalize       = normalize
        self.clip_min        = clip_min
        self.clip_max        = clip_max
        self.min_length      = min_length
        self.max_length      = max_length
        self.read_ids_filter = read_ids_filter

    def iter_reads(
        self,
        fast5_path: Path,
    ) -> Iterator[Tuple[str, np.ndarray]]:
        """
        Yield (read_id, signal) pairs from a FAST5 file.

        Args:
            fast5_path: Path to the .fast5 file

        Yields:
            (read_id, signal_array) tuples
        """
        try:
            with h5py.File(fast5_path, "r") as f:
                for top_key in f.keys():
                    grp = f[top_key]

                    # Extract read_id from key or attributes
                    read_id = top_key.replace("read_", "").replace("Read_", "")

                    # Check filter
                    if (self.read_ids_filter is not None
                            and read_id not in self.read_ids_filter):
                        continue

                    # Find raw signal
                    signal = _find_raw_signal(grp)
                    if signal is None:
                        logger.debug(f"No signal found for {read_id} in {fast5_path.name}")
                        continue

                    # Length filtering
                    if len(signal) < self.min_length:
                        continue
                    if len(signal) > self.max_length:
                        signal = signal[:self.max_length]

                    # Normalize
                    if self.normalize:
                        signal = mad_normalize(signal, self.clip_min, self.clip_max)
                    else:
                        signal = signal.astype(np.float32)

                    yield read_id, signal

        except OSError as e:
            logger.warning(f"Could not open {fast5_path}: {e}")

    def read_all(
        self,
        fast5_path: Path,
    ) -> Dict[str, np.ndarray]:
        """
        Return all reads from a FAST5 file as a dict.

        Returns:
            {read_id: signal_array}
        """
        return dict(self.iter_reads(fast5_path))

    def get_file_read_ids(self, fast5_path: Path) -> List[str]:
        """Return all read IDs in a FAST5 file without loading signals."""
        try:
            with h5py.File(fast5_path, "r") as f:
                return [k.replace("read_", "") for k in f.keys()]
        except OSError:
            return []

    def get_read_count(self, fast5_path: Path) -> int:
        """Return the number of reads in a FAST5 file."""
        return len(self.get_file_read_ids(fast5_path))


def build_fast5_index(fast5_dir: Path) -> Dict[str, Path]:
    """
    Build a mapping from read_id → fast5 file path for an entire directory.

    This is computed once and cached since it requires opening every file.
    For 1,131 files this takes ~5-10 minutes.

    Args:
        fast5_dir: Directory containing .fast5 files

    Returns:
        {read_id: fast5_filepath} mapping
    """
    from tqdm import tqdm

    fast5_files = sorted(p for p in fast5_dir.glob("*.fast5") if not p.name.startswith("."))  # skip macOS ._ sidecar files
    logger.info(f"Building index for {len(fast5_files)} FAST5 files…")

    index: Dict[str, Path] = {}
    for f5 in tqdm(fast5_files, desc="Indexing FAST5"):
        try:
            with h5py.File(f5, "r") as f:
                for key in f.keys():
                    rid = key.replace("read_", "")
                    index[rid] = f5
        except OSError as e:
            logger.warning(f"Skipping {f5.name}: {e}")

    logger.info(f"Indexed {len(index):,} reads across {len(fast5_files)} files")
    return index
