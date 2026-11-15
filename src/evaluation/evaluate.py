"""
Evaluate — Test Set Evaluation & Architecture Comparison
=========================================================
Runs evaluation on the held-out test set for one or more trained models,
computing all metrics for the research paper.

Outputs:
  - Per-read predictions CSV
  - Summary metrics JSON
  - Per-cycle accuracy breakdown

Usage:
    python src/evaluation/evaluate.py \
        --checkpoint checkpoints/conformer/best_model.pt \
        --arch conformer \
        --split test

    # Compare all architectures
    python src/evaluation/evaluate.py --compare-all
"""

import os
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_model(
    arch:            str,
    checkpoint_path: Path,
    test_index_path: Path,
    results_dir:     Path,
    library_path:    Optional[Path] = None,
    batch_size:      int = 4,
    max_signal_len:  int = 80_000,
    n_samples:       Optional[int] = None,
) -> dict:
    """
    Run full evaluation on the test set.

    Returns a dict of all computed metrics.
    """
    import vbz_h5py_plugin  # noqa
    from config import COMPLETED_LIBRARY_PATH, DEVICE, PROCESSED_DATA_DIR
    from src.data.motif_library import MotifLibrary
    from src.data.dataset import MotifDatasetIndex, MotifSignalDataset, motif_collate_fn
    from src.data.fast5_reader import Fast5Reader
    from src.model.motifcaller import build_model
    from src.training.ctc_decoder import GreedyCTCDecoder
    from src.training.metrics import (
        motif_accuracy, exact_match_rate, per_cycle_accuracy
    )
    from torch.utils.data import DataLoader

    device = DEVICE
    logger.info(f"Evaluating [{arch}] on {device}")

    # Load components
    if library_path is None:
        library_path = COMPLETED_LIBRARY_PATH
    lib = MotifLibrary(library_path)
    decoder = GreedyCTCDecoder(labels=lib.label_names, blank=0)

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model = build_model(arch, n_classes=lib.n_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load test index
    test_index = MotifDatasetIndex.load(test_index_path)
    if n_samples:
        test_index.entries = test_index.entries[:n_samples]
    logger.info(f"  Test set: {len(test_index):,} reads")

    reader = Fast5Reader(normalise=True)
    test_ds = MotifSignalDataset(test_index, reader, max_signal_len)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        collate_fn=motif_collate_fn, shuffle=False
    )

    all_preds   = []
    all_targets = []
    read_results = []

    for batch in tqdm(test_loader, desc=f"  Evaluating {arch}"):
        if batch is None:
            continue

        signals, flat_labels, signal_lengths, label_lengths = batch
        signals = signals.to(device)

        with torch.no_grad():
            if arch == "conformer":
                log_probs = model(signals, signal_lengths.to(device))
            else:
                log_probs = model(signals)

        offset = 0
        for i, ll in enumerate(label_lengths):
            ll = ll.item()
            emission = log_probs[:, i, :].unsqueeze(1)
            pred = decoder(emission)

            tgt = [lib.label_to_motif(l.item())
                   for l in flat_labels[offset:offset+ll]]
            all_preds.append(pred)
            all_targets.append(tgt)
            read_results.append({
                "pred": "|".join(pred),
                "target": "|".join(tgt),
                "n_pred": len(pred),
                "n_target": len(tgt),
                "exact_match": pred == tgt,
            })
            offset += ll

    # Compute metrics
    acc        = motif_accuracy(all_preds, all_targets)
    exact      = exact_match_rate(all_preds, all_targets)
    per_cycle  = per_cycle_accuracy(all_preds, all_targets)

    metrics = {
        "architecture":    arch,
        "n_test_reads":    len(all_preds),
        "motif_accuracy":  round(acc, 4),
        "exact_match_rate": round(exact, 4),
        "per_cycle_accuracy": {
            f"cycle_{c}": round(v, 4)
            for c, v in per_cycle.items()
        },
        "mean_n_pred":   round(np.mean([r["n_pred"]   for r in read_results]), 2),
        "mean_n_target": round(np.mean([r["n_target"] for r in read_results]), 2),
    }

    logger.info(f"\n  Results for [{arch}]:")
    logger.info(f"    Motif accuracy:   {acc:.4f}")
    logger.info(f"    Exact match rate: {exact:.4f}")
    logger.info(f"    Per-cycle accuracy:")
    for cycle, v in per_cycle.items():
        logger.info(f"      Cycle {cycle}: {v:.4f}")

    # Save
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{arch}_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Metrics saved to {out_json}")

    return metrics


def compare_all_architectures(
    checkpoint_base: Path,
    test_index_path: Path,
    results_dir:     Path,
    library_path:    Optional[Path] = None,
):
    """Evaluate all three architectures and produce a comparison table."""
    archs = ["benchmark", "conformer", "causal"]
    all_metrics = {}

    for arch in archs:
        ckpt_path = checkpoint_base / arch / "best_model.pt"
        if not ckpt_path.exists():
            logger.warning(f"No checkpoint for {arch} at {ckpt_path}, skipping")
            continue
        metrics = evaluate_model(
            arch=arch,
            checkpoint_path=ckpt_path,
            test_index_path=test_index_path,
            results_dir=results_dir,
            library_path=library_path,
        )
        all_metrics[arch] = metrics

    # Print comparison table
    print("\n" + "=" * 60)
    print("ARCHITECTURE COMPARISON")
    print("=" * 60)
    print(f"{'Architecture':<15} {'Motif Acc':>10} {'Exact Match':>12} {'Cycle 3-5 Avg':>14}")
    print("-" * 60)
    for arch, m in all_metrics.items():
        c35_avg = np.mean([m["per_cycle_accuracy"].get(f"cycle_{c}", 0) for c in [3,4,5]])
        print(f"{arch:<15} {m['motif_accuracy']:>10.4f} {m['exact_match_rate']:>12.4f} {c35_avg:>14.4f}")

    # Save combined
    combined_path = results_dir / "comparison.json"
    with open(combined_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nComparison saved to {combined_path}")


def main():
    from config import COMPLETED_LIBRARY_PATH, PROCESSED_DATA_DIR

    default_test_idx = PROCESSED_DATA_DIR / "dataset_index" / "dataset_test.pkl"
    if not default_test_idx.exists():
        default_test_idx = Path("data/dataset_index/dataset_test.pkl")

    p = argparse.ArgumentParser(description="Evaluate ConformerCaller models")
    p.add_argument("--checkpoint",  type=Path, default=None)
    p.add_argument("--arch",        default="conformer",
                   choices=["benchmark", "conformer", "causal"])
    p.add_argument("--library",     type=Path, default=COMPLETED_LIBRARY_PATH,
                   help="Path to completed_library.csv (default: config.COMPLETED_LIBRARY_PATH)")
    p.add_argument("--test-index",  type=Path,
                   default=default_test_idx,
                   help=f"Path to dataset_test.pkl (default: {default_test_idx})")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--compare-all", action="store_true",
                   help="Evaluate all architectures")
    p.add_argument("--n-samples",   type=int, default=None,
                   help="Limit to N test samples (for quick evaluation)")
    p.add_argument("--checkpoint-base", type=Path, default=Path("checkpoints"),
                   help="Base dir for --compare-all")
    args = p.parse_args()

    if args.compare_all:
        compare_all_architectures(
            args.checkpoint_base,
            args.test_index,
            args.results_dir,
            library_path=args.library,
        )
    else:
        if not args.checkpoint:
            p.error("--checkpoint required unless using --compare-all")
        evaluate_model(
            arch            = args.arch,
            checkpoint_path = args.checkpoint,
            test_index_path = args.test_index,
            results_dir     = args.results_dir,
            library_path    = args.library,
            n_samples       = args.n_samples,
        )


if __name__ == "__main__":
    main()
