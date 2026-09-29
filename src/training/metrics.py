"""
Metrics — Evaluation Metrics for ConformerCaller
=============================================
"""

from typing import List
from Levenshtein import ratio as levenshtein_ratio


def motif_accuracy(
    predictions: List[List[str]],
    targets:     List[List[str]],
) -> float:
    """
    Fraction of correctly predicted motifs, averaged over reads.

    For each read, computes the number of matching motifs
    (at the sequence level using longest common subsequence logic)
    divided by max(pred_len, target_len).

    Note: Uses Levenshtein ratio as a proxy (handles insertions/deletions).
    """
    if not predictions:
        return 0.0
    scores = []
    for pred, tgt in zip(predictions, targets):
        if not tgt:
            scores.append(1.0 if not pred else 0.0)
            continue
        pred_str = " ".join(pred)
        tgt_str  = " ".join(tgt)
        scores.append(levenshtein_ratio(pred_str, tgt_str))
    return sum(scores) / len(scores)


def edit_distance_ratio(
    predictions: List[List[str]],
    targets:     List[List[str]],
) -> float:
    """
    Mean Levenshtein ratio between predicted and target motif sequences.
    1.0 = perfect match, 0.0 = completely different.
    """
    return motif_accuracy(predictions, targets)


def exact_match_rate(
    predictions: List[List[str]],
    targets:     List[List[str]],
) -> float:
    """Fraction of reads with a perfectly predicted motif sequence."""
    if not predictions:
        return 0.0
    return sum(p == t for p, t in zip(predictions, targets)) / len(predictions)


def per_cycle_accuracy(
    predictions: List[List[str]],
    targets:     List[List[str]],
) -> dict:
    """
    Accuracy broken down by payload cycle position (cycles 3-10).
    Only evaluated on reads where both pred and target have that cycle.

    Returns:
        {cycle_num: accuracy}
    """
    from collections import defaultdict
    correct = defaultdict(int)
    total   = defaultdict(int)

    for pred, tgt in zip(predictions, targets):
        # Build {cycle: motif_name} for both
        def cycle_map(motifs):
            m = {}
            import re
            for name in motifs:
                match = re.match(r"ltm8_(\d+)x(\d+)", name)
                if match:
                    cycle = int(match.group(1))
                    if 3 <= cycle <= 10:
                        m[cycle] = name
            return m

        pred_map = cycle_map(pred)
        tgt_map  = cycle_map(tgt)

        for cycle in range(3, 11):
            if cycle in tgt_map:
                total[cycle] += 1
                if pred_map.get(cycle) == tgt_map[cycle]:
                    correct[cycle] += 1

    return {
        c: correct[c] / total[c] if total[c] > 0 else 0.0
        for c in range(3, 11)
    }
