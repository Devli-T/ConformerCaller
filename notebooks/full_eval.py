"""
Full test-set evaluation from pre-extracted signal shards.

Evaluates the trained ConformerCaller on every test read found in the signal
shards (66,843 reads when the complete public dataset has been processed) and
reports detection rate, exact match, Levenshtein similarity, per-cycle
accuracy and the most frequent confusions for both decoders (greedy and
confidence ≥ 0.70).

Prerequisites (see README, "Reproducing the paper results"):
    python src/data/build_training_set.py
    python src/data/extract_signals_sharded.py --index-dir <processed>/dataset_index --splits test

Usage:
    python notebooks/full_eval.py
    python notebooks/full_eval.py --max-shards 1        # quick sanity check

Results are written to results/full_test_eval_reproduced.json — the
reference numbers from the paper are kept separately in
results/paper/full_test_eval.json and are never overwritten.
"""

import argparse
import gc
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from Levenshtein import ratio as lev_ratio
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COMPLETED_LIBRARY_PATH, DEVICE, PROCESSED_DATA_DIR, PROJECT_ROOT
from src.data.dataset import MotifDatasetIndex, MotifSignalDataset, motif_collate_fn
from src.data.fast5_reader import Fast5Reader
from src.data.motif_library import MotifLibrary
from src.model.motifcaller import build_model, count_parameters
from src.training.ctc_decoder import GreedyCTCDecoder

MAX_SIG_LEN = 20_000


def get_cycle(motif_name):
    m = re.match(r"ltm8_(\d+)x", motif_name)
    return int(m.group(1)) if m else None


def parse_args():
    default_idx = PROCESSED_DATA_DIR / "dataset_index"
    p = argparse.ArgumentParser(description="Evaluate ConformerCaller on the test shards")
    p.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints/conformer/best_model.pt")
    p.add_argument("--library",    type=Path, default=COMPLETED_LIBRARY_PATH)
    p.add_argument("--index-dir",  type=Path, default=default_idx,
                   help="Folder with dataset_test.pkl (default: <processed>/dataset_index)")
    p.add_argument("--shard-dir",  type=Path, default=None,
                   help="Folder with signals_shard_*.npz (default: <index-dir>/signal_shards)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-shards", type=int, default=None, help="Evaluate only the first N shards")
    p.add_argument("--max-reads",  type=int, default=None, help="Evaluate at most N test reads in total")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results/full_test_eval_reproduced.json")
    return p.parse_args()


def main():
    args = parse_args()
    shard_dir = args.shard_dir or (args.index_dir / "signal_shards")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    device = DEVICE
    lib = MotifLibrary(args.library)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model("conformer", n_classes=lib.n_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model: conformer | {count_parameters(model)/1e6:.2f}M params | device: {device}")
    print(f"Checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['best_val_loss']:.4f}")

    # ── Test index and shards ────────────────────────────────────────────────
    test_idx = MotifDatasetIndex.load(args.index_dir / "dataset_test.pkl")
    test_lookup = {e[0]: e for e in test_idx.entries}
    print(f"Test index: {len(test_lookup):,} reads")

    shard_files = sorted(shard_dir.glob("signals_shard_*.npz"))
    if args.max_shards:
        shard_files = shard_files[: args.max_shards]
    if not shard_files:
        sys.exit(f"No signal shards found in {shard_dir}. Run extract_signals_sharded.py --splits test first.")
    print(f"Signal shards: {len(shard_files)}\n")

    reader  = Fast5Reader(normalise=True)
    decoder = GreedyCTCDecoder(labels=lib.label_names, blank=0)

    g_lev, g_det, g_exact, g_pred_lens, g_tgt_lens = [], [], 0, [], []   # greedy
    c_lev, c_det, c_exact, c_pred_lens = [], [], 0, []                    # conf >= 0.70
    cycle_correct = defaultdict(int)
    cycle_total   = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
    n_reads_total = 0

    for shard_path in shard_files:
        if args.max_reads and n_reads_total >= args.max_reads:
            break
        with np.load(str(shard_path)) as npz:
            entries = [test_lookup[rid] for rid in npz.files if rid in test_lookup]
            if args.max_reads:
                entries = entries[: args.max_reads - n_reads_total]
            if not entries:
                print(f"  {shard_path.name}: no test reads, skipping")
                continue
            cache = {e[0]: npz[e[0]] for e in entries}

        shard_idx = MotifDatasetIndex.__new__(MotifDatasetIndex)
        shard_idx.entries = entries
        ds = MotifSignalDataset(shard_idx, reader, MAX_SIG_LEN, cache)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=motif_collate_fn, num_workers=0)

        shard_reads = 0
        with torch.no_grad():
            for b_idx, batch in enumerate(tqdm(loader, desc=f"{shard_path.name} ({len(entries)} reads)", leave=False)):
                if batch is None:
                    continue
                sigs, flat_lbl, sig_lens, lbl_lens = batch
                log_probs = model(sigs.to(device), sig_lens.to(device))

                offset = 0
                for i, ll in enumerate(lbl_lens):
                    ll = ll.item()
                    em  = log_probs[:, i, :]
                    tgt = [lib.label_to_motif(l.item()) for l in flat_lbl[offset:offset + ll]]

                    pred_g = decoder(em.unsqueeze(1))
                    pred_c = decoder.decode_with_confidence(em.unsqueeze(1), 0.70)

                    g_pred_lens.append(len(pred_g)); g_tgt_lens.append(len(tgt))
                    if tgt:
                        g_lev.append(lev_ratio(" ".join(pred_g), " ".join(tgt)))
                        g_det.append(sum(1 for m in tgt if m in pred_g) / len(tgt))
                    if pred_g == tgt:
                        g_exact += 1

                    c_pred_lens.append(len(pred_c))
                    if tgt:
                        c_lev.append(lev_ratio(" ".join(pred_c), " ".join(tgt)))
                        c_det.append(sum(1 for m in tgt if m in pred_c) / len(tgt))
                    if pred_c == tgt:
                        c_exact += 1

                    pred_map = {get_cycle(m): m for m in pred_g if get_cycle(m)}
                    tgt_map  = {get_cycle(m): m for m in tgt  if get_cycle(m)}
                    for c, motif in tgt_map.items():
                        cycle_total[c] += 1
                        if pred_map.get(c) == motif:
                            cycle_correct[c] += 1
                        elif c in pred_map:
                            confusion[motif][pred_map[c]] += 1

                    n_reads_total += 1
                    shard_reads += 1
                    offset += ll

                if (b_idx + 1) % 10 == 0 and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        if hasattr(cache, "close"):
            cache.close()
        del cache, ds, loader
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"  {shard_path.name}: {shard_reads} reads processed | Running total: {n_reads_total:,}")

    # ── Report ───────────────────────────────────────────────────────────────
    N = n_reads_total
    if N == 0:
        sys.exit("No test reads were evaluated.")
    print("\n" + "=" * 72)
    print(f"TEST SET EVALUATION — {N:,} reads")
    print("=" * 72)
    print(f"\n{'Metric':<35} {'Greedy':>10} {'Conf≥0.70':>12}")
    print("-" * 60)
    print(f"{'Detection %/read':<35} {np.mean(g_det)*100:>10.2f} {np.mean(c_det)*100:>12.2f}")
    print(f"{'Levenshtein similarity':<35} {np.mean(g_lev):>10.4f} {np.mean(c_lev):>12.4f}")
    print(f"{'Exact match rate %':<35} {g_exact/N*100:>10.2f} {c_exact/N*100:>12.2f}")
    print(f"{'Mean pred length':<35} {np.mean(g_pred_lens):>10.2f} {np.mean(c_pred_lens):>12.2f}")
    print(f"{'Mean target length':<35} {np.mean(g_tgt_lens):>10.2f} {np.mean(g_tgt_lens):>12.2f}")

    print("\nPer-Cycle Accuracy (greedy decoder):")
    print(f"  {'Cycle':<8} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 38)
    for c in sorted(cycle_total):
        acc = cycle_correct[c] / cycle_total[c] if cycle_total[c] else 0
        print(f"  Cycle {c:<3} {cycle_correct[c]:>8} {cycle_total[c]:>8} {acc*100:>10.1f}%")
    overall = sum(cycle_correct.values()) / max(1, sum(cycle_total.values()))
    print(f"  {'Overall':<8} {sum(cycle_correct.values()):>8} {sum(cycle_total.values()):>8} {overall*100:>10.1f}%")

    pairs = sorted(((c, t, p) for t, pc in confusion.items() for p, c in pc.items()), reverse=True)
    print("\nTop 15 confused motif pairs:")
    print(f"  {'Count':>6}  {'True':>18}  {'Predicted':>18}")
    print("-" * 50)
    for count, true_m, pred_m in pairs[:15]:
        print(f"  {count:>6}  {true_m:>18}  {pred_m:>18}")

    results = {
        "n_reads": N,
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_loss": ckpt["best_val_loss"],
        "greedy": {
            "detection_pct": float(np.mean(g_det) * 100),
            "levenshtein": float(np.mean(g_lev)),
            "exact_match_pct": g_exact / N * 100,
            "mean_pred_len": float(np.mean(g_pred_lens)),
            "mean_tgt_len": float(np.mean(g_tgt_lens)),
        },
        "confidence_0.70": {
            "detection_pct": float(np.mean(c_det) * 100),
            "levenshtein": float(np.mean(c_lev)),
            "exact_match_pct": c_exact / N * 100,
            "mean_pred_len": float(np.mean(c_pred_lens)),
        },
        "per_cycle_accuracy": {
            str(c): {"correct": cycle_correct[c], "total": cycle_total[c],
                     "accuracy": cycle_correct[c] / cycle_total[c] if cycle_total[c] else 0}
            for c in sorted(cycle_total)
        },
        "top_confusions": [{"count": c, "true": t, "predicted": p} for c, t, p in pairs[:30]],
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {args.out}")


if __name__ == "__main__":
    main()
