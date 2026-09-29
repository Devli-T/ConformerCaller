"""
Full test set evaluation across 7 test shards (66,843 reads).

Usage:
    PYTHONPATH=. python3 notebooks/full_eval.py

Reports all metrics and saves results to results/paper/.
"""

import sys, json, copy, gc
from pathlib import Path
from collections import defaultdict
import re

import torch
import numpy as np
from tqdm import tqdm
from Levenshtein import ratio as lev_ratio

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COMPLETED_LIBRARY_PATH, DEVICE, PROCESSED_DATA_DIR
from src.data.motif_library import MotifLibrary
from src.data.dataset import MotifDatasetIndex, MotifSignalDataset, motif_collate_fn
from src.data.fast5_reader import Fast5Reader
from src.model.motifcaller import build_model, count_parameters
from src.training.ctc_decoder import GreedyCTCDecoder
from torch.utils.data import DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
SHARD_DIR   = PROCESSED_DATA_DIR / "signal_shards"
IDX_DIR     = PROCESSED_DATA_DIR / "dataset_index_full"
CHECKPOINT  = Path("checkpoints/conformer/best_model.pt")
RESULTS_DIR = Path("results/paper")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_SIG_LEN = 20_000
BATCH_SIZE  = 4  # increase for multi-GPU or larger GPU memory

def get_cycle(motif_name):
    m = re.match(r"ltm8_(\d+)x", motif_name)
    return int(m.group(1)) if m else None

# ── Load model ────────────────────────────────────────────────────────────────
device = DEVICE
lib  = MotifLibrary(COMPLETED_LIBRARY_PATH)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = build_model("conformer", n_classes=lib.n_classes).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Model: conformer | {count_parameters(model)/1e6:.2f}M params")
print(f"Checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['best_val_loss']:.4f}")

# ── Load shard manifest & test index ─────────────────────────────────────────
with open(SHARD_DIR / "shard_split_ids.json") as f:
    shard_split_ids = json.load(f)

# Use full index for test lookup
test_idx = MotifDatasetIndex.load(IDX_DIR / "dataset_test.pkl")
test_lookup = {e[0]: e for e in test_idx.entries}
print(f"Full test index: {len(test_lookup):,} reads")

TEST_SHARDS = sorted([s for s in shard_split_ids if shard_split_ids[s]["test"] > 0])
total_test_reads = sum(shard_split_ids[s]["test"] for s in TEST_SHARDS)
print(f"Test shards: {len(TEST_SHARDS)} | Total reads: {total_test_reads:,}")
print()

reader  = Fast5Reader(normalize=True)
decoder = GreedyCTCDecoder(labels=lib.label_names, blank=0)

# ── Aggregate accumulators ────────────────────────────────────────────────────
# Greedy
g_lev, g_det, g_exact, g_pred_lens, g_tgt_lens = [], [], 0, [], []
# Confidence 0.70
c_lev, c_det, c_exact, c_pred_lens = [], [], 0, []
# Per-cycle (greedy)
cycle_correct = defaultdict(int)
cycle_total   = defaultdict(int)
# Confusion
confusion = defaultdict(lambda: defaultdict(int))
n_reads_total = 0

# ── Shard loop ────────────────────────────────────────────────────────────────
for shard_name in TEST_SHARDS:
    n_test = shard_split_ids[shard_name]["test"]
    if n_test == 0:
        continue

    # Build filtered index for this shard
    ids = shard_split_ids[shard_name].get("test_ids", [])
    entries = [test_lookup[rid] for rid in ids if rid in test_lookup]
    if not entries:
        print(f"  Shard {shard_name}: 0 entries found (index mismatch?), skipping")
        continue

    shard_idx = copy.copy(test_idx)
    shard_idx.entries = entries

    # Stream arrays directly on demand via memory mapping
    cache = np.load(str(SHARD_DIR / shard_name), mmap_mode="r")

    ds = MotifSignalDataset(shard_idx, reader, MAX_SIG_LEN, cache)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=motif_collate_fn, num_workers=0)

    shard_reads = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{shard_name} ({len(entries)} reads)", leave=False):
            if batch is None:
                continue
            sigs, flat_lbl, sig_lens, lbl_lens = batch
            log_probs = model(sigs.to(device), sig_lens.to(device))

            offset = 0
            for i, ll in enumerate(lbl_lens):
                ll = ll.item()
                em  = log_probs[:, i, :]
                tgt = [lib.label_to_motif(l.item()) for l in flat_lbl[offset:offset+ll]]

                pred_g = decoder(em.unsqueeze(1))
                pred_c = decoder.decode_with_confidence(em.unsqueeze(1), 0.70)

                # Greedy metrics
                g_pred_lens.append(len(pred_g))
                g_tgt_lens.append(len(tgt))
                if tgt:
                    g_lev.append(lev_ratio(" ".join(pred_g), " ".join(tgt)))
                    g_det.append(sum(1 for m in tgt if m in pred_g) / len(tgt))
                if pred_g == tgt:
                    g_exact += 1

                # Confidence metrics
                c_pred_lens.append(len(pred_c))
                if tgt:
                    c_lev.append(lev_ratio(" ".join(pred_c), " ".join(tgt)))
                    c_det.append(sum(1 for m in tgt if m in pred_c) / len(tgt))
                if pred_c == tgt:
                    c_exact += 1

                # Per-cycle accuracy (greedy)
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

    # Explicitly close file handles and release GPU/device memory
    if hasattr(cache, "close"):
        cache.close()
    del cache, ds, loader
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print(f"  {shard_name}: {shard_reads} reads processed | Running total: {n_reads_total:,}")

# ── Print results ─────────────────────────────────────────────────────────────
N = n_reads_total
print()
print("=" * 72)
print(f"FULL TEST SET EVALUATION — {N:,} reads")
print("=" * 72)

print(f"\n{'Metric':<35} {'Greedy':>10} {'Conf≥0.70':>12}")
print("-" * 60)
print(f"{'Detection %/read':<35} {np.mean(g_det)*100:>10.2f} {np.mean(c_det)*100:>12.2f}")
print(f"{'Levenshtein similarity':<35} {np.mean(g_lev):>10.4f} {np.mean(c_lev):>12.4f}")
print(f"{'Exact match rate %':<35} {g_exact/N*100:>10.2f} {c_exact/N*100:>12.2f}")
print(f"{'Mean pred length':<35} {np.mean(g_pred_lens):>10.2f} {np.mean(c_pred_lens):>12.2f}")
print(f"{'Mean target length':<35} {np.mean(g_tgt_lens):>10.2f} {np.mean(g_tgt_lens):>12.2f}")

print(f"\nPer-Cycle Accuracy (greedy decoder):")
print(f"  {'Cycle':<8} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
print("-" * 38)
for c in sorted(cycle_total.keys()):
    acc = cycle_correct[c] / cycle_total[c] if cycle_total[c] else 0
    print(f"  Cycle {c:<3} {cycle_correct[c]:>8} {cycle_total[c]:>8} {acc*100:>10.1f}%")
overall_acc = sum(cycle_correct.values()) / max(1, sum(cycle_total.values()))
print(f"  {'Overall':<8} {sum(cycle_correct.values()):>8} {sum(cycle_total.values()):>8} {overall_acc*100:>10.1f}%")

print(f"\nTop 15 confused motif pairs:")
pairs = [(c, t, p) for t, pc in confusion.items() for p, c in pc.items()]
pairs.sort(reverse=True)
print(f"  {'Count':>6}  {'True':>18}  {'Predicted':>18}")
print("-" * 50)
for count, true_m, pred_m in pairs[:15]:
    print(f"  {count:>6}  {true_m:>18}  {pred_m:>18}")

# ── Save results ──────────────────────────────────────────────────────────────
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
        str(c): {
            "correct": cycle_correct[c],
            "total": cycle_total[c],
            "accuracy": cycle_correct[c] / cycle_total[c] if cycle_total[c] else 0,
        }
        for c in sorted(cycle_total.keys())
    },
    "top_confusions": [
        {"count": c, "true": t, "predicted": p}
        for c, t, p in pairs[:30]
    ],
}

out_path = RESULTS_DIR / "full_test_eval.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {out_path}")
print(f"\nShare the contents of {out_path} to continue paper writing.")