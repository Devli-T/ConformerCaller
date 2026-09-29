"""
Label Builder — Ground Truth Construction from master_db.csv
=============================================================
Parses the master_db.csv file to extract ordered payload motif
label sequences for each read, ready for CTC training.

Key decisions implemented:
  - Labels are payload-motif-only (Cycles 3-10, 64 classes + blank)
  - Motifs are sorted by their start position in the read
  - Strand orientation filtering (configurable, default: forward only)
  - Minimum payload motif count filtering
"""

import re
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .motif_library import (
    MotifLibrary,
    parse_pipe_delimited_motifs,
    parse_pipe_delimited_start_end,
    parse_pipe_delimited_strand,
)

logger = logging.getLogger(__name__)

MOTIF_RE = re.compile(r"ltm8_(\d+)x(\d+)")


class LabelRecord:
    """Ground truth label info for a single read."""
    __slots__ = ["read_id", "filename", "hw_address",
                 "payload_labels", "n_payload_cycles",
                 "all_motifs", "strand_orientation"]

    def __init__(self, read_id, filename, hw_address,
                 payload_labels, n_payload_cycles,
                 all_motifs, strand_orientation):
        self.read_id           = read_id
        self.filename          = filename
        self.hw_address        = hw_address
        self.payload_labels    = payload_labels        # List[int] for CTC
        self.n_payload_cycles  = n_payload_cycles      # int: distinct cycles present
        self.all_motifs        = all_motifs             # List[str] raw motif names
        self.strand_orientation = strand_orientation   # "plus", "minus", or "mixed"


class LabelBuilder:
    """
    Builds ground truth label sequences from master_db.csv.

    Scans master_db.csv in chunks (it's ~1GB) and returns a dict
    mapping read_id → LabelRecord.
    """

    def __init__(
        self,
        motif_library:          MotifLibrary,
        min_payload_cycles:     int   = 4,
        forward_strand_only:    bool  = True,
        chunksize:              int   = 100_000,
    ):
        """
        Args:
            motif_library:        Initialised MotifLibrary
            min_payload_cycles:   Minimum distinct payload cycles required
                                  (reads below this are excluded)
            forward_strand_only:  If True, skip reverse-complement reads
            chunksize:            Rows per pandas chunk when reading master_db
        """
        self.lib                 = motif_library
        self.min_payload_cycles  = min_payload_cycles
        self.forward_strand_only = forward_strand_only
        self.chunksize           = chunksize

    def _classify_orientation(self, strands: List[str]) -> str:
        unique = set(strands)
        if not unique:          return "unknown"
        if unique == {"+"}:     return "plus"
        if unique == {"-"}:     return "minus"
        return "mixed"

    def _process_row(self, row) -> Optional[LabelRecord]:
        """Parse one master_db row into a LabelRecord (or None if filtered)."""

        motifs  = parse_pipe_delimited_motifs(row.get("library_motif", ""))
        ses     = parse_pipe_delimited_start_end(row.get("start_end", ""))
        strands = parse_pipe_delimited_strand(row.get("strand", ""))

        orientation = self._classify_orientation(strands)

        # Strand filter
        if self.forward_strand_only and orientation != "plus":
            return None

        # Build (start, label) pairs for payload motifs only
        payload_entries: List[Tuple[int, int]] = []
        payload_cycles_seen = set()

        n_entries = min(len(motifs), len(ses)) if ses else len(motifs)
        for i in range(n_entries):
            name = motifs[i]
            label = self.lib.motif_to_label(name)
            if label is None:
                continue  # Not a payload motif

            # Determine payload cycle
            m = MOTIF_RE.match(name)
            if m:
                payload_cycles_seen.add(int(m.group(1)))

            # Use position if available, else append
            if ses and i < len(ses):
                start = ses[i][0]
            else:
                start = i * 50  # fallback estimate

            payload_entries.append((start, label))

        n_cycles = len(payload_cycles_seen)
        if n_cycles < self.min_payload_cycles:
            return None

        # Sort by position
        payload_entries.sort(key=lambda x: x[0])
        payload_labels = [label for _, label in payload_entries]

        return LabelRecord(
            read_id           = str(row.get("read_id", "")),
            filename          = str(row.get("filename", "")),
            hw_address        = str(row.get("HW_Address", "")),
            payload_labels    = payload_labels,
            n_payload_cycles  = n_cycles,
            all_motifs        = motifs,
            strand_orientation= orientation,
        )

    def build(
        self,
        master_db_path: Path,
        max_reads: Optional[int] = None,
    ) -> Dict[str, LabelRecord]:
        """
        Parse master_db.csv and build label records.

        Args:
            master_db_path: Path to master_db.csv
            max_reads:      Optional cap (for quick tests)

        Returns:
            {read_id: LabelRecord} dict
        """
        records: Dict[str, LabelRecord] = {}
        total_seen = 0
        total_skipped_strand = 0
        total_skipped_few    = 0

        for chunk in pd.read_csv(
            master_db_path,
            chunksize  = self.chunksize,
            dtype      = str,
            low_memory = False,
        ):
            for _, row in chunk.iterrows():
                total_seen += 1

                rec = self._process_row(row)
                if rec is None:
                    # Determine skip reason for stats
                    motifs = parse_pipe_delimited_motifs(row.get("library_motif", ""))
                    strands = parse_pipe_delimited_strand(row.get("strand", ""))
                    orient = self._classify_orientation(strands)
                    if self.forward_strand_only and orient != "plus":
                        total_skipped_strand += 1
                    else:
                        total_skipped_few += 1
                    continue

                records[rec.read_id] = rec

                if max_reads and len(records) >= max_reads:
                    break

            if total_seen % 500_000 == 0:
                logger.info(f"  Processed {total_seen:,} rows, "
                            f"kept {len(records):,}…")

            if max_reads and len(records) >= max_reads:
                break

        logger.info(
            f"Label building complete: {total_seen:,} rows scanned, "
            f"{len(records):,} records kept, "
            f"{total_skipped_strand:,} skipped (strand), "
            f"{total_skipped_few:,} skipped (<{self.min_payload_cycles} payload cycles)"
        )
        return records

    def save(self, records: Dict[str, LabelRecord], out_path: Path):
        """Save label records to a pickle file."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved {len(records):,} label records to {out_path}")

    @staticmethod
    def load(path: Path) -> Dict[str, LabelRecord]:
        """Load label records from a pickle file."""
        with open(path, "rb") as f:
            records = pickle.load(f)
        logger.info(f"Loaded {len(records):,} label records from {path}")
        return records


def summarise_labels(records: Dict[str, LabelRecord]) -> dict:
    """Print and return summary statistics for a label dict."""
    from collections import Counter
    n_cycle_dist   = Counter(r.n_payload_cycles for r in records.values())
    hw_addr_dist   = Counter(r.hw_address       for r in records.values())
    orient_dist    = Counter(r.strand_orientation for r in records.values())
    label_len_dist = Counter(len(r.payload_labels) for r in records.values())

    print(f"Total labelled reads : {len(records):,}")
    print(f"\nPayload cycles per read:")
    for k in sorted(n_cycle_dist):
        print(f"  {k} cycles: {n_cycle_dist[k]:,}")
    print(f"\nStrand orientation:")
    for k, v in orient_dist.most_common():
        print(f"  {k}: {v:,}")
    print(f"\nTop 10 HW_Address values:")
    for k, v in hw_addr_dist.most_common(10):
        print(f"  {k}: {v:,}")
    print(f"\nLabel sequence length distribution:")
    for k in sorted(label_len_dist)[:15]:
        print(f"  {k} labels: {label_len_dist[k]:,}")

    return {
        "total":           len(records),
        "n_cycle_dist":    dict(n_cycle_dist),
        "orientation":     dict(orient_dist),
        "hw_addr_top10":   dict(hw_addr_dist.most_common(10)),
        "label_len_dist":  dict(label_len_dist),
    }
