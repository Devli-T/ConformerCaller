"""
Motif Library — Definitions, Parsing, and Label Mapping
========================================================
Handles the motif naming convention, library parsing, and conversion
between motif names and integer labels for CTC training.

Motif naming convention: ltm8_CxV
  - C = Cycle position (0-11 for library motifs, 89-99 for splints)
  - V = Variant number (1-8 for standard, 1-11 for splints)

Cycle types:
  - 0: 5' primer flank
  - 1-2: Address regions (16 motifs total)
  - 3-10: Payload regions (64 motifs total) ← OUR PREDICTION TARGET
  - 11: 3' primer flank
  - 89-99: Splint sequences (expected and unexpected combinations)
"""

import re
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class MotifInfo:
    """Information about a single motif from the library."""
    name: str           # e.g., "ltm8_3x5"
    cycle: int          # e.g., 3
    variant: int        # e.g., 5
    sequence: str       # 50bp DNA sequence
    cycle_type: str     # "primer", "address", "payload", or "splint"
    length: int = 0     # Sequence length

    def __post_init__(self):
        self.length = len(self.sequence)


class MotifLibrary:
    """
    Manages the complete motif library including label mappings for ML.

    Label encoding for CTC:
      - Index 0: CTC blank token
      - Index 1-64: Payload motifs (ltm8_3x1 → 1, ltm8_3x2 → 2, ..., ltm8_10x8 → 64)

    The label mapping encodes BOTH cycle position and variant, so the model
    predicts which specific payload motif (cycle + variant) is present.
    """

    # Regex to parse motif names like "ltm8_3x5" or "ltm8_99x11"
    MOTIF_PATTERN = re.compile(r"ltm8_(\d+)x(\d+)")

    # Special primer motif names (in completed_library.csv these may be named differently)
    PRIMER_5P = "ltm8_5pf"  # or ltm8_0x1
    PRIMER_3P = "ltm8_3pf"  # or ltm8_11x1

    def __init__(self, completed_library_path: Optional[Path] = None):
        """
        Initialise the motif library.

        Args:
            completed_library_path: Path to completed_library.csv.
                If None, builds from hardcoded motif naming convention.
        """
        self.motifs: Dict[str, MotifInfo] = {}
        self._name_to_label: Dict[str, int] = {}
        self._label_to_name: Dict[int, str] = {}
        self._payload_motifs: Dict[str, MotifInfo] = {}
        self._address_motifs: Dict[str, MotifInfo] = {}
        self._splint_motifs: Dict[str, MotifInfo] = {}
        self._primer_motifs: Dict[str, MotifInfo] = {}

        if completed_library_path:
            self._load_from_csv(completed_library_path)

        self._build_label_mapping()

    def _load_from_csv(self, path: Path):
        """Load motif definitions from completed_library.csv."""
        df = pd.read_csv(path)
        # Strip whitespace from column names (CSV has trailing spaces in headers)
        df.columns = df.columns.str.strip()

        for _, row in df.iterrows():
            name = row["Name"]
            sequence = row["Sequence"]
            cycle = int(row["Cycle"])
            cycle_type_raw = str(row["Cycle Type"]).strip().lower()

            # Map cycle type
            if "primer" in cycle_type_raw or "flank" in cycle_type_raw:
                cycle_type = "primer"
            elif "address" in cycle_type_raw:
                cycle_type = "address"
            elif "payload" in cycle_type_raw:
                cycle_type = "payload"
            else:
                cycle_type = "unknown"

            # Parse variant from name
            variant = self._parse_variant(name)

            motif = MotifInfo(
                name=name,
                cycle=cycle,
                variant=variant,
                sequence=sequence,
                cycle_type=cycle_type
            )
            self.motifs[name] = motif

            # Categorise
            if cycle_type == "payload":
                self._payload_motifs[name] = motif
            elif cycle_type == "address":
                self._address_motifs[name] = motif
            elif cycle_type == "primer":
                self._primer_motifs[name] = motif

    def _parse_variant(self, name: str) -> int:
        """Extract variant number from motif name."""
        match = self.MOTIF_PATTERN.match(name)
        if match:
            return int(match.group(2))
        return 0

    @staticmethod
    def parse_motif_name(name: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Parse a motif name into (cycle, variant).

        Args:
            name: Motif name like "ltm8_3x5" or "ltm8_99x2"

        Returns:
            (cycle, variant) tuple, or (None, None) if unparseable
        """
        match = re.match(r"ltm8_(\d+)x(\d+)", name)
        if match:
            return int(match.group(1)), int(match.group(2))
        # Handle primer names
        if name in ("ltm8_5pf", "ltm8_0x1"):
            return 0, 1
        if name in ("ltm8_3pf", "ltm8_11x1"):
            return 11, 1
        return None, None

    def _build_label_mapping(self):
        """
        Build the integer label mapping for CTC training.

        Label 0 = CTC blank
        Labels 1-64 = payload motifs, ordered by (cycle, variant):
            ltm8_3x1 → 1, ltm8_3x2 → 2, ..., ltm8_3x8 → 8,
            ltm8_4x1 → 9, ltm8_4x2 → 10, ..., ltm8_4x8 → 16,
            ...
            ltm8_10x1 → 57, ..., ltm8_10x8 → 64
        """
        self._name_to_label = {}
        self._label_to_name = {0: "<blank>"}

        label_idx = 1
        for cycle in range(3, 11):  # Payload cycles 3-10
            for variant in range(1, 9):  # Variants 1-8
                name = f"ltm8_{cycle}x{variant}"
                self._name_to_label[name] = label_idx
                self._label_to_name[label_idx] = name
                label_idx += 1

        assert label_idx == 65, f"Expected 65 labels (64 + blank), got {label_idx}"

    def motif_to_label(self, motif_name: str) -> Optional[int]:
        """
        Convert a motif name to its integer label.

        Returns None if the motif is not a payload motif (i.e., not in our label set).
        """
        return self._name_to_label.get(motif_name)

    def label_to_motif(self, label: int) -> str:
        """Convert an integer label back to a motif name."""
        return self._label_to_name.get(label, "<unknown>")

    def is_payload_motif(self, motif_name: str) -> bool:
        """Check if a motif name is a payload motif (Cycles 3-10)."""
        cycle, variant = self.parse_motif_name(motif_name)
        if cycle is None:
            return False
        return 3 <= cycle <= 10 and 1 <= variant <= 8

    def is_address_motif(self, motif_name: str) -> bool:
        """Check if a motif name is an address motif (Cycles 1-2)."""
        cycle, _ = self.parse_motif_name(motif_name)
        if cycle is None:
            return False
        return cycle in (1, 2)

    def is_splint_motif(self, motif_name: str) -> bool:
        """Check if a motif name is a splint motif (Cycles 89-99)."""
        cycle, _ = self.parse_motif_name(motif_name)
        if cycle is None:
            return False
        return 89 <= cycle <= 99

    def is_primer_motif(self, motif_name: str) -> bool:
        """Check if a motif name is a primer motif."""
        cycle, _ = self.parse_motif_name(motif_name)
        if cycle is None:
            return False
        return cycle in (0, 11)

    def classify_motif(self, motif_name: str) -> str:
        """Classify a motif as 'payload', 'address', 'splint', 'primer', or 'unknown'."""
        if self.is_payload_motif(motif_name):
            return "payload"
        if self.is_address_motif(motif_name):
            return "address"
        if self.is_splint_motif(motif_name):
            return "splint"
        if self.is_primer_motif(motif_name):
            return "primer"
        return "unknown"

    def get_payload_labels_from_motif_list(
        self, motif_names: List[str]
    ) -> List[int]:
        """
        Extract payload motif labels from a list of motif names.

        Filters to only payload motifs and returns their integer labels
        in the order they appear.
        """
        labels = []
        for name in motif_names:
            label = self.motif_to_label(name)
            if label is not None:
                labels.append(label)
        return labels

    def get_ordered_payload_labels(
        self,
        motif_names: List[str],
        start_ends: List[Tuple[int, int]],
    ) -> List[int]:
        """
        Extract payload motif labels ordered by their position in the read.

        Args:
            motif_names: List of motif names (pipe-delimited from master_db)
            start_ends: Corresponding (start, end) positions for each motif

        Returns:
            List of integer labels for payload motifs, sorted by start position
        """
        payload_entries = []
        for name, (start, end) in zip(motif_names, start_ends):
            label = self.motif_to_label(name)
            if label is not None:
                payload_entries.append((start, label))

        # Sort by position in the read
        payload_entries.sort(key=lambda x: x[0])
        return [label for _, label in payload_entries]

    @property
    def n_classes(self) -> int:
        """Total number of output classes (including CTC blank)."""
        return len(self._label_to_name)

    @property
    def n_payload_motifs(self) -> int:
        """Number of unique payload motifs."""
        return self.n_classes - 1

    @property
    def label_names(self) -> List[str]:
        """List of all label names in order."""
        return [self._label_to_name[i] for i in range(self.n_classes)]

    def get_motif_sequence(self, motif_name: str) -> Optional[str]:
        """Get the DNA sequence for a motif, if loaded from CSV."""
        if motif_name in self.motifs:
            return self.motifs[motif_name].sequence
        return None

    def __repr__(self):
        return (
            f"MotifLibrary("
            f"n_classes={self.n_classes}, "
            f"payload_motifs={self.n_payload_motifs}, "
            f"library_motifs={len(self.motifs)})"
        )


# =============================================================================
# Utility functions for parsing master_db entries
# =============================================================================

def parse_pipe_delimited_motifs(motif_str: str) -> List[str]:
    """Parse pipe-delimited motif string from master_db.csv."""
    if pd.isna(motif_str) or not motif_str:
        return []
    return [m.strip() for m in motif_str.split("|") if m.strip()]


def parse_pipe_delimited_start_end(se_str: str) -> List[Tuple[int, int]]:
    """Parse pipe-delimited start_end string from master_db.csv."""
    if pd.isna(se_str) or not se_str:
        return []
    pairs = []
    for entry in se_str.split("|"):
        entry = entry.strip()
        if not entry or "-" not in entry:
            continue
        parts = entry.split("-")
        try:
            pairs.append((int(parts[0]), int(parts[1])))
        except (ValueError, IndexError):
            continue
    return pairs


def parse_pipe_delimited_strand(strand_str: str) -> List[str]:
    """Parse pipe-delimited strand string from master_db.csv."""
    if pd.isna(strand_str) or not strand_str:
        return []
    return [s.strip() for s in strand_str.split("|") if s.strip()]
