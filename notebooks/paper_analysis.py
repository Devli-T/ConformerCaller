"""
Paper Analysis Tools
====================
Produces confusion matrix, per-cycle accuracy breakdown, decoder comparison,
and information recovery curves.

Usage:
    python3.12 notebooks/paper_analysis.py
"""

import sys, json, pickle
from pathlib import Path
from collections import defaultdict
import re

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COMPLETED_LIBRARY_PATH, DEVICE, PROCESSED_DATA_DIR
from src.data.motif_library import MotifLibrary
from src.data.dataset import MotifDatasetIndex, MotifSignalDataset, motif_collate_fn
from src.data.fast5_reader import Fast5Reader
from src.model.motifcaller import build_model, count_parameters
from src.training.ctc_decoder import GreedyCTCDecoder
from torch.utils.data import DataLoader

RESULTS_DIR = Path("results/paper")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT   = Path("checkpoints/conformer/best_model.pt")
IDX_DIR      = PROCESSED_DATA_DIR / "dataset_index"
MAX_SIG_LEN  = 20_000
BATCH_SIZE   = 4  # increase for multi-GPU or larger GPU memory

ARCH_NAME = "Conformer (ours)"
PAPER_COLOR  = "#2196F3"
GREEDY_COLOR = "#4CAF50"
CONF_COLOR   = "#FF9800"


def run_inference(model, test_loader, decoder, lib, device, threshold=None):
    """Run inference and return (preds, targets)."""
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            if batch is None:
                continue
            sigs, flat_lbl, sig_lens, lbl_lens = batch
            log_probs = model(sigs.to(device), sig_lens.to(device))
            offset = 0
            for i, ll in enumerate(lbl_lens):
                ll = ll.item()
                em = log_probs[:, i, :]
                if threshold is not None:
                    pred = decoder.decode_with_confidence(em.unsqueeze(1), threshold)
                else:
                    pred = decoder(em.unsqueeze(1))
                tgt = [lib.label_to_motif(l.item()) for l in flat_lbl[offset:offset+ll]]
                all_preds.append(pred)
                all_targets.append(tgt)
                offset += ll
    return all_preds, all_targets


def get_cycle(motif_name):
    m = re.match(r"ltm8_(\d+)x(\d+)", motif_name)
    return int(m.group(1)) if m else None


# ── 1. Load model & data ──────────────────────────────────────────────────────
device = DEVICE
lib  = MotifLibrary(COMPLETED_LIBRARY_PATH)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = build_model("conformer", n_classes=lib.n_classes).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

print(f"Model: conformer | {count_parameters(model)/1e6:.2f}M params")
print(f"Checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['best_val_loss']:.4f}")

test_idx = MotifDatasetIndex.load(IDX_DIR / "dataset_test.pkl")
npz = np.load(str(IDX_DIR / "signals.npz"), allow_pickle=False)
signals_cache = {k: npz[k] for k in npz.files}
reader = Fast5Reader(normalize=True)
test_ds = MotifSignalDataset(test_idx, reader, MAX_SIG_LEN, signals_cache)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=motif_collate_fn, shuffle=False)

decoder = GreedyCTCDecoder(labels=lib.label_names, blank=0)

print(f"\nRunning inference on {len(test_idx)} test reads...")
preds_greedy, targets = run_inference(model, test_loader, decoder, lib, device)
preds_conf70, _       = run_inference(model, test_loader, decoder, lib, device, threshold=0.70)

n = len(targets)
print(f"Done — {n} reads evaluated")


# ── 2. Decoder comparison table (for paper) ───────────────────────────────────
from Levenshtein import ratio as lev_ratio

def compute_metrics(preds, targets):
    lev, det, exact = [], [], 0
    for pred, tgt in zip(preds, targets):
        if tgt:
            lev.append(lev_ratio(" ".join(pred), " ".join(tgt)))
            det.append(sum(1 for m in tgt if m in pred) / len(tgt))
        if pred == tgt:
            exact += 1
    return {
        "levenshtein": np.mean(lev),
        "detection_pct": np.mean(det) * 100,
        "exact_match_pct": exact / max(1, len(preds)) * 100,
        "mean_pred_len": np.mean([len(p) for p in preds]),
        "mean_tgt_len":  np.mean([len(t) for t in targets]),
    }

m_greedy = compute_metrics(preds_greedy, targets)
m_conf   = compute_metrics(preds_conf70, targets)

table = {
    "methods": [
        {"name": "Motif Search (paper)",     "detection_pct": 8.83,  "n_reads_needed": 66},
        {"name": "Motif Caller CNN+BiGRU (paper)", "detection_pct": 15.90, "n_reads_needed": 37},
        {"name": "Conformer — greedy (ours)", **m_greedy},
        {"name": "Conformer — conf≥0.70 (ours)", **m_conf},
    ]
}

print("\n" + "="*72)
print("DECODER COMPARISON TABLE (for paper)")
print("="*72)
print(f"{'Method':<35} {'Det%':>7} {'Lev':>7} {'EM%':>6} {'pred':>6} {'tgt':>5}")
print("-"*72)
for m in table["methods"]:
    det = f"{m['detection_pct']:.2f}"
    lev = f"{m.get('levenshtein', '—'):>7}" if isinstance(m.get('levenshtein'), float) else f"{'—':>7}"
    em  = f"{m.get('exact_match_pct', '—'):>6}" if isinstance(m.get('exact_match_pct'), float) else f"{'—':>6}"
    pl  = f"{m.get('mean_pred_len', '—'):>6}" if isinstance(m.get('mean_pred_len'), float) else f"{'—':>6}"
    tl  = f"{m.get('mean_tgt_len', '—'):>5}" if isinstance(m.get('mean_tgt_len'), float) else f"{'—':>5}"
    print(f"{m['name']:<35} {det:>7} {lev} {em} {pl} {tl}")

with open(RESULTS_DIR / "decoder_comparison.json", "w") as f:
    json.dump(table, f, indent=2, default=str)


# ── 3. Per-cycle accuracy bar chart ───────────────────────────────────────────
def per_cycle_acc(preds, targets, cycles=range(3, 11)):
    correct = defaultdict(int)
    total   = defaultdict(int)
    for pred, tgt in zip(preds, targets):
        pred_map = {get_cycle(m): m for m in pred if get_cycle(m) in cycles}
        tgt_map  = {get_cycle(m): m for m in tgt  if get_cycle(m) in cycles}
        for c in cycles:
            if c in tgt_map:
                total[c] += 1
                if pred_map.get(c) == tgt_map[c]:
                    correct[c] += 1
    return {c: correct[c]/total[c] if total[c] else 0 for c in cycles}

cycles = list(range(3, 11))
acc_greedy = per_cycle_acc(preds_greedy, targets)
acc_conf   = per_cycle_acc(preds_conf70, targets)

x = np.arange(len(cycles))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 4))
bars_g = ax.bar(x - w/2, [acc_greedy[c]*100 for c in cycles], w,
                label="Greedy decoder", color=GREEDY_COLOR, alpha=0.85)
bars_c = ax.bar(x + w/2, [acc_conf[c]*100 for c in cycles], w,
                label="Confidence ≥0.70", color=CONF_COLOR, alpha=0.85)

ax.set_xlabel("Payload Cycle", fontsize=12)
ax.set_ylabel("Per-Cycle Accuracy (%)", fontsize=12)
ax.set_title("ConformerCaller — Per-Cycle Accuracy (5k Test Set)", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([f"C{c}" for c in cycles])
ax.set_ylim(0, 110)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Annotate bars
for bar in list(bars_g) + list(bars_c):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "per_cycle_accuracy.png", dpi=200, bbox_inches="tight")
print(f"\n✓ Per-cycle accuracy chart saved")

# Print table
print(f"\n{'Cycle':<8} {'Greedy %':>10} {'Conf0.70 %':>12} {'N samples':>10}")
print("-"*44)
for c in cycles:
    n_samp = sum(1 for tgt in targets if any(get_cycle(m) == c for m in tgt))
    print(f"Cycle {c:<3} {acc_greedy[c]*100:>10.1f} {acc_conf[c]*100:>12.1f} {n_samp:>10}")


# ── 4. Confusion matrix (top-N most confused motifs) ─────────────────────────
from collections import Counter

# For each cycle, track (predicted, true) pairs where prediction is wrong
confusion = defaultdict(Counter)  # {true_motif: Counter(predicted_motif)}

for pred, tgt in zip(preds_conf70, targets):
    pred_map = {}
    for m in pred:
        c = get_cycle(m)
        if c: pred_map[c] = m

    for m in tgt:
        c = get_cycle(m)
        if not c: continue
        if c in pred_map and pred_map[c] != m:
            confusion[m][pred_map[c]] += 1

# Find top-20 most confused pairs
confused_pairs = []
for true_m, pred_counter in confusion.items():
    for pred_m, count in pred_counter.items():
        confused_pairs.append((count, true_m, pred_m))
confused_pairs.sort(reverse=True)

print("\n" + "="*60)
print("TOP 20 CONFUSED MOTIF PAIRS (true→predicted)")
print("="*60)
print(f"{'Count':>6}  {'True':>18}  {'Predicted':>18}")
print("-"*50)
for count, true_m, pred_m in confused_pairs[:20]:
    print(f"{count:>6}  {true_m:>18}  {pred_m:>18}")

# Save confusion data
confusion_export = [
    {"count": c, "true": t, "predicted": p}
    for c, t, p in confused_pairs[:50]
]
with open(RESULTS_DIR / "top_confusions.json", "w") as f:
    json.dump(confusion_export, f, indent=2)


# ── 5. Information recovery curve (paper Fig-6 equivalent) ────────────────────
# Simulates: given N reads from the same block, how many unique payload cycles
# are recovered (detected in at least one read)?

def simulate_recovery(preds, targets, n_trials=500, max_reads=50):
    """
    For each number of reads R, sample R reads randomly and count
    how many unique payload cycles (3-10) have at least one correct prediction.
    Returns: {R: mean_fraction_cycles_recovered}
    """
    paired = [(p, t) for p, t in zip(preds, targets) if t]
    results = defaultdict(list)

    for _ in range(n_trials):
        sample = np.random.choice(len(paired), size=min(max_reads, len(paired)), replace=True)
        recovered_so_far = set()
        for r, idx in enumerate(sample, 1):
            p, t = paired[idx]
            pred_map = {get_cycle(m): m for m in p if get_cycle(m)}
            tgt_map  = {get_cycle(m): m for m in t if get_cycle(m)}
            for c, motif in tgt_map.items():
                if pred_map.get(c) == motif:
                    recovered_so_far.add(c)
            results[r].append(len(recovered_so_far & set(range(3, 11))) / 8.0)

    return {r: np.mean(v) for r, v in sorted(results.items())}

MAX_READS = 50
print("\nComputing information recovery curves...")
recovery_greedy = simulate_recovery(preds_greedy, targets, max_reads=MAX_READS)
recovery_conf   = simulate_recovery(preds_conf70, targets, max_reads=MAX_READS)

# Paper baseline: Motif Search ~8.83% per read
# Approximate their recovery curve: P(cycle detected in R reads) = 1-(1-0.0883)^R
reads = sorted(recovery_greedy.keys())
motif_search_baseline = [1 - (1 - 0.0883)**r for r in reads]
motif_caller_paper    = [1 - (1 - 0.1590)**r for r in reads]

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(reads, motif_search_baseline,    "k--",  lw=1.5, label="Motif Search pipeline (paper ~8.8%/read)", alpha=0.7)
ax2.plot(reads, motif_caller_paper,       "b--",  lw=1.5, label="Motif Caller CNN+BiGRU (paper ~15.9%/read)", alpha=0.7)
ax2.plot(reads, [recovery_greedy[r] for r in reads], color=GREEDY_COLOR, lw=2.5, label="Conformer — greedy (ours)")
ax2.plot(reads, [recovery_conf[r] for r in reads],   color=CONF_COLOR,   lw=2.5, label="Conformer — conf≥0.70 (ours)")

ax2.axhline(1.0, color="gray", lw=0.8, linestyle=":")
ax2.set_xlabel("Number of reads sampled", fontsize=12)
ax2.set_ylabel("Fraction of payload cycles recovered", fontsize=12)
ax2.set_title("Information Recovery vs. Number of Reads", fontsize=13)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax2.legend(fontsize=9, loc="lower right")
ax2.set_xlim(1, MAX_READS)
ax2.set_ylim(0, 1.05)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "information_recovery.png", dpi=200, bbox_inches="tight")
print(f"✓ Information recovery curve saved")

# Find reads needed for 90% and 100% recovery
for threshold_pct in [0.5, 0.75, 0.90, 1.0]:
    for r in reads:
        if recovery_greedy[r] >= threshold_pct:
            print(f"  Greedy: {threshold_pct*100:.0f}% recovery at {r} reads")
            break
    for r in reads:
        if recovery_conf[r] >= threshold_pct:
            print(f"  Conf70: {threshold_pct*100:.0f}% recovery at {r} reads")
            break

plt.show()
print(f"\n✓ All paper figures saved to {RESULTS_DIR}/")
print("  - per_cycle_accuracy.png")
print("  - information_recovery.png")
print("  - decoder_comparison.json")
print("  - top_confusions.json")
