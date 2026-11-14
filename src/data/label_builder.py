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

    # ── Input formats ────────────────────────────────────────────────────────
    # 1. "wide" : one row per read, pipe-delimited motif/strand/position fields
    #             (columns: read_id, filename, fastq_id, HW_Address, strand,
    #              start_end, library_motif) — the format of the internal
    #              master_db.csv the model was originally trained from.
    # 2. "long" : one row per motif hit, space-delimited, no header
    #             (read_id filename barcode strand start end motif cycle index)
    #             — the format of the public Zenodo file
    #             HELIX01-04_master_db.txt.
    # The format is detected automatically; both give identical payload labels.
    LONG_COLUMNS = ["read_id", "filename", "barcode", "strand",
                    "start", "end", "motif", "cycle", "index"]

    @staticmethod
    def detect_format(path: Path) -> str:
        """Return "wide" or "long" by inspecting the first line of the file."""
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
        return "wide" if "library_motif" in first else "long"

    def _iter_rows(self, path: Path):
        """Yield one wide-format row (dict-like) per read, for either format."""
        if self.detect_format(path) == "wide":
            for chunk in pd.read_csv(path, chunksize=self.chunksize,
                                     dtype=str, low_memory=False):
                for _, row in chunk.iterrows():
                    yield row
            return
        yield from self._iter_long_rows(path)

    def _iter_long_rows(self, path: Path):
        """Collapse the long (one-row-per-hit) format into one row per read.

        Requires all rows of a read to be contiguous (true for the public
        file, which is sorted by read_id).
        """
        seen: set = set()
        cur, filename, barcode = None, "", ""
        strands, spans, motifs = [], [], []

        def collapse():
            return {
                "read_id":       cur,
                "filename":      filename,
                "HW_Address":    barcode,
                "strand":        "|".join(strands),
                "start_end":     "|".join(spans),
                "library_motif": "|".join(motifs),
            }

        for chunk in pd.read_csv(
            path, sep=" ", header=None, names=self.LONG_COLUMNS,
            dtype=str, chunksize=self.chunksize * 5,
        ):
            for rid, fn, bc, st, s, e, mo in zip(
                chunk["read_id"], chunk["filename"], chunk["barcode"],
                chunk["strand"], chunk["start"], chunk["end"], chunk["motif"],
            ):
                if rid != cur:
                    if cur is not None:
                        yield collapse()
                    if rid in seen:
                        raise ValueError(
                            f"{path} is not grouped by read_id (read {rid} "
                            "reappears). Sort the file by read_id first."
                        )
                    seen.add(rid)
                    cur, filename, barcode = rid, fn, bc
                    strands, spans, motifs = [], [], []
                strands.append(st)
                spans.append(f"{s}-{e}")
                motifs.append(mo)
        if cur is not None:
            yield collapse()

    def build(
        self,
        master_db_path: Path,
        max_reads: Optional[int] = None,
        only_read_ids: Optional[set] = None,
    ) -> Dict[str, LabelRecord]:
        """
        Parse a master database and build label records.

        Args:
            master_db_path: Path to master_db.csv (wide format) or the public
                            HELIX01-04_master_db.txt (long format) — detected
                            automatically.
            max_reads:      Optional cap (for quick tests)
            only_read_ids:  Optional set of read IDs; all other reads are skipped

        Returns:
            {read_id: LabelRecord} dict
        """
        records: Dict[str, LabelRecord] = {}
        total_seen = 0
        total_skipped_strand = 0
        total_skipped_few    = 0

        for row in self._iter_rows(Path(master_db_path)):
            total_seen += 1
            if only_read_ids is not None and row.get("read_id") not in only_read_ids:
                continue

            rec = self._process_row(row)
            if rec is None:
                # Determine skip reason for stats
                strands = parse_pipe_delimited_strand(row.get("strand", ""))
                orient = self._classify_orientation(strands)
                if self.forward_strand_only and orient != "plus":
                    total_skipped_strand += 1
                else:
                    total_skipped_few += 1
            else:
                records[rec.read_id] = rec

            if total_seen % 500_000 == 0:
                logger.info(f"  Processed {total_seen:,} reads, "
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
