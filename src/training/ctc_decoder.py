"""
CTC Decoder — Greedy and Beam Search Decoding
==============================================
Converts model output log-probabilities into motif label sequences.

Decoders:
  - GreedyCTCDecoder:    Fast best-path decoding (argmax per timestep)
  - BeamSearchCTCDecoder: Beam search with length normalisation
  - decode_with_confidence: Greedy with confidence threshold filtering
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import heapq
from dataclasses import dataclass, field


class GreedyCTCDecoder(nn.Module):
    """
    Greedy (best-path) CTC decoder.

    At each timestep picks the highest-probability label, then:
    1. Merges consecutive identical labels
    2. Removes blank tokens (index 0)

    This is fast and ~equivalent to beam search for well-trained models.
    """

    def __init__(self, labels: List[str], blank: int = 0):
        """
        Args:
            labels: List of label strings, index 0 = blank
            blank:  Blank token index (default: 0)
        """
        super().__init__()
        self.labels = labels
        self.blank  = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """
        Args:
            emission: (T, B, C) or (T, C) log-probabilities

        Returns:
            List of decoded label strings for each item in batch,
            or a single list if no batch dimension.
        """
        if emission.dim() == 2:
            emission = emission.unsqueeze(1)  # add batch dim

        T, B, C = emission.shape
        results = []

        for b in range(B):
            # Greedy best path
            indices = torch.argmax(emission[:, b, :], dim=-1)  # (T,)

            # Collapse consecutive duplicates
            collapsed = []
            prev = -1
            for idx in indices.tolist():
                if idx != prev:
                    collapsed.append(idx)
                prev = idx

            # Remove blanks
            decoded = [self.labels[i] for i in collapsed if i != self.blank]
            results.append(decoded)

        return results if B > 1 else results[0]

    def decode_with_confidence(
        self, emission: torch.Tensor, threshold: float = 0.5
    ) -> List[str]:
        """
        Greedy decode with confidence threshold — only include labels
        where the max probability exceeds threshold.
        Useful for filtering low-confidence predictions.
        """
        if emission.dim() == 2:
            emission = emission.unsqueeze(1)

        probs = torch.exp(emission)  # (T, B, C)
        max_probs, indices = probs.max(dim=-1)  # (T, B)

        T, B = indices.shape
        results = []

        for b in range(B):
            decoded = []
            prev = -1
            for t in range(T):
                idx = indices[t, b].item()
                prob = max_probs[t, b].item()
                if idx != prev:
                    if idx != self.blank and prob >= threshold:
                        decoded.append(self.labels[idx])
                prev = idx
            results.append(decoded)

        return results if B > 1 else results[0]


@dataclass(order=True)
class BeamEntry:
    """A single beam hypothesis during CTC beam search."""
    neg_score: float  # negative log prob (for min-heap ordering)
    sequence: tuple = field(compare=False)  # tuple of label indices (no blanks)
    last_label: int = field(compare=False)  # last emitted label (for CTC merge rule)
    raw_log_prob: float = field(compare=False)  # unnormalised log probability


class BeamSearchCTCDecoder(nn.Module):
    """
    CTC Beam Search decoder with Word Insertion Penalty (WIP).

    Explores multiple CTC paths simultaneously, keeping the top-K beams.
    A positive word_insertion_penalty subtracts from the score for every
    non-blank token emitted, discouraging spurious insertions — the main
    over-prediction failure mode in undertrained CTC models.

    Args:
        labels:                  List of label strings, index 0 = blank
        blank:                   Blank token index (default: 0)
        beam_width:              Number of beams to keep (default: 10)
        word_insertion_penalty:  Subtracted per emitted token (default: 0.0).
                                 Increase (e.g. 0.3–1.0) to suppress insertions.
                                 Set to 0 for standard CTC beam search.
        top_k_per_step:          Only consider top-K labels per timestep (default: 10).
    """

    def __init__(
        self,
        labels: List[str],
        blank: int = 0,
        beam_width: int = 10,
        word_insertion_penalty: float = 0.0,
        top_k_per_step: int = 10,
    ):
        super().__init__()
        self.labels = labels
        self.blank = blank
        self.beam_width = beam_width
        self.word_insertion_penalty = word_insertion_penalty
        self.top_k_per_step = min(top_k_per_step, len(labels))

    def _beam_score(self, log_prob: float, seq_len: int) -> float:
        """Ranking score: raw log_prob minus WIP per emitted token."""
        return log_prob - self.word_insertion_penalty * seq_len

    def _beam_search_single(self, log_probs: torch.Tensor) -> List[int]:
        """
        Run beam search on a single sequence's log-probabilities.

        Args:
            log_probs: (T, C) log-probabilities

        Returns:
            List of label indices (blanks removed, duplicates merged)
        """
        T, C = log_probs.shape

        # Initialise: each beam is (sequence_tuple, last_label, raw_log_prob)
        beams = [((), -1, 0.0)]  # (seq, last_label, log_prob)

        for t in range(T):
            step_log_probs = log_probs[t]  # (C,)

            # Top-K pruning per timestep for efficiency
            if self.top_k_per_step < C:
                topk_vals, topk_ids = torch.topk(step_log_probs, self.top_k_per_step)
                candidates = list(zip(topk_ids.tolist(), topk_vals.tolist()))
            else:
                candidates = [(i, step_log_probs[i].item()) for i in range(C)]

            # Always include blank
            blank_lp = step_log_probs[self.blank].item()
            if self.blank not in [c[0] for c in candidates]:
                candidates.append((self.blank, blank_lp))

            new_beams: dict = {}  # (sequence, last_label) -> best raw log_prob

            for seq, last_label, beam_lp in beams:
                for label_idx, label_lp in candidates:
                    new_lp = beam_lp + label_lp

                    if label_idx == self.blank:
                        # Blank: sequence unchanged, reset last_label
                        key = (seq, -1)
                        if key not in new_beams or new_beams[key] < new_lp:
                            new_beams[key] = new_lp

                    elif label_idx == last_label:
                        # CTC merge rule: same as last, don't extend
                        key = (seq, label_idx)
                        if key not in new_beams or new_beams[key] < new_lp:
                            new_beams[key] = new_lp

                    else:
                        # New label: extend sequence
                        new_seq = seq + (label_idx,)
                        key = (new_seq, label_idx)
                        if key not in new_beams or new_beams[key] < new_lp:
                            new_beams[key] = new_lp

            # Prune to beam_width using WIP score
            scored = [
                (self._beam_score(lp, len(seq)), seq, ll, lp)
                for (seq, ll), lp in new_beams.items()
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            beams = [(seq, ll, lp) for _, seq, ll, lp in scored[:self.beam_width]]

        if not beams:
            return []

        # Return highest WIP-scored beam
        best_seq = max(beams, key=lambda b: self._beam_score(b[2], len(b[0])))[0]
        return list(best_seq)

    def forward(self, emission: torch.Tensor) -> List[str]:
        """
        Args:
            emission: (T, B, C) or (T, C) log-probabilities
        Returns:
            Decoded motif list per batch item.
        """
        if emission.dim() == 2:
            emission = emission.unsqueeze(1)

        T, B, C = emission.shape
        results = []
        for b in range(B):
            indices = self._beam_search_single(emission[:, b, :])
            decoded = [self.labels[i] for i in indices]
            results.append(decoded)

        return results if B > 1 else results[0]


def build_decoder(
    labels: List[str],
    method: str = "greedy",
    blank: int = 0,
    beam_width: int = 10,
    word_insertion_penalty: float = 0.0,
    confidence_threshold: float = 0.70,
    **kwargs,
):
    """
    Factory to create a CTC decoder.

    Args:
        labels:                  Label strings (index 0 = blank)
        method:                  'greedy' | 'beam' | 'confidence'
        blank:                   Blank index
        beam_width:              Beam width for beam search
        word_insertion_penalty:  WIP for beam search (0 = standard beam search)
        confidence_threshold:    Threshold for confidence-filtered greedy (0.70 default)

    Returns:
        Decoder instance
    """
    if method == "greedy":
        return GreedyCTCDecoder(labels=labels, blank=blank)
    elif method == "beam":
        return BeamSearchCTCDecoder(
            labels=labels, blank=blank,
            beam_width=beam_width,
            word_insertion_penalty=word_insertion_penalty,
            **kwargs,
        )
    elif method == "confidence":
        return GreedyCTCDecoder(labels=labels, blank=blank)
    else:
        raise ValueError(f"Unknown decoder method: {method}")

