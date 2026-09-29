"""generate_encoding_schematic.py.

================================
Generates results/paper/encoding_schematic.png

Publication-ready schematic for motif-based DNA storage encoding and decoding.
Organized into three clean panels:
  (a) In silico encoding and physical oligo assembly
  (b) Oxford Nanopore sequencing and ionic current acquisition
  (c) Decoding paradigms: Traditional two-stage pipeline vs. End-to-end Conformer

Run from project root:
    MPLCONFIGDIR=/tmp/mpl python3.12 notebooks/generate_encoding_schematic.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np

Path("results/paper").mkdir(parents=True, exist_ok=True)

FW, FH = 14.0, 9.2
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")
fig.patch.set_facecolor("white")


# -------------------------------------------------------------------------
# Drawing primitives
# -------------------------------------------------------------------------
def rbox(
    ax,
    x1,
    y1,
    x2,
    y2,
    fc,
    ec="#64748B",
    lw=1.2,
    r=0.08,
    zorder=3,
    linestyle="-",
):
  ax.add_patch(
      FancyBboxPatch(
          (x1, y1),
          x2 - x1,
          y2 - y1,
          boxstyle=f"round,pad={r}",
          facecolor=fc,
          edgecolor=ec,
          linewidth=lw,
          linestyle=linestyle,
          zorder=zorder,
      )
  )


def arr_h(ax, x1, x2, y, lw=1.4, col="#475569"):
  ax.annotate(
      "",
      xy=(x2, y),
      xytext=(x1, y),
      arrowprops=dict(
          arrowstyle="->", color=col, lw=lw, shrinkA=2, shrinkB=2
      ),
      zorder=6,
  )


def arr_v(ax, x, y1, y2, lw=1.4, col="#475569"):
  ax.annotate(
      "",
      xy=(x, y2),
      xytext=(x, y1),
      arrowprops=dict(
          arrowstyle="->", color=col, lw=lw, shrinkA=2, shrinkB=2
      ),
      zorder=6,
  )


# -------------------------------------------------------------------------
# PANEL A: ENCODING & OLIGO DESIGN
# -------------------------------------------------------------------------
ax.text(
    0.5,
    8.85,
    "a",
    fontsize=13,
    fontweight="bold",
    color="#0F172A",
    ha="left",
    va="center",
)
ax.text(
    0.75,
    8.85,
    "Encoding & Physical Library Assembly",
    fontsize=11,
    fontweight="bold",
    color="#1E293B",
    ha="left",
    va="center",
)

# 1. Digital Information Block
rbox(ax, 0.5, 7.15, 2.0, 8.35, "#F8FAFC", ec="#64748B", lw=1.3)
ax.text(
    1.25,
    7.85,
    "Binary Input",
    ha="center",
    va="center",
    fontsize=9.5,
    fontweight="bold",
    color="#0F172A",
    zorder=5,
)
ax.text(
    1.25,
    7.50,
    "Payload Data\nBitstream",
    ha="center",
    va="center",
    fontsize=8.0,
    color="#475569",
    zorder=5,
)
arr_h(ax, 2.0, 2.35, 7.75)

# 2. Payload Selection Bracket & 8 Cycle Boxes
ax.plot(
    [2.35, 2.35, 9.15, 9.15],
    [8.42, 8.49, 8.49, 8.42],
    color="#64748B",
    lw=1.0,
    zorder=5,
)
ax.text(
    5.75,
    8.62,
    "Payload Motifs (8 cycles, C3–C10)",
    ha="center",
    va="center",
    fontsize=9.0,
    fontweight="bold",
    color="#334155",
    zorder=5,
)

cycle_colors = [
    "#E0F2FE",
    "#E0E7FF",
    "#EDE9FE",
    "#FCE7F3",
    "#CCFBF1",
    "#D1FAE5",
    "#FEF3C7",
    "#FFEDD5",
]
cycle_start_x = 2.35
cycle_w = 0.80
cycle_gap = 0.05
for i in range(8):
  x1 = cycle_start_x + i * (cycle_w + cycle_gap)
  x2 = x1 + cycle_w
  rbox(ax, x1, 7.15, x2, 8.35, cycle_colors[i], ec="#94A3B8", lw=1.0)
  cx = (x1 + x2) / 2
  ax.text(
      cx,
      7.92,
      f"C{i+3}",
      ha="center",
      va="center",
      fontsize=10.5,
      fontweight="bold",
      color="#1E293B",
      zorder=5,
  )
  ax.text(
      cx,
      7.57,
      "1 of 8",
      ha="center",
      va="center",
      fontsize=7.5,
      color="#475569",
      zorder=5,
  )
  ax.text(
      cx,
      7.35,
      "variants",
      ha="center",
      va="center",
      fontsize=7.0,
      color="#64748B",
      zorder=5,
  )

ax.text(
    5.75,
    6.85,
    "Library: 64 payload motifs (8 cycles × 8 variants)   |   25 nt per motif",
    ha="center",
    va="center",
    fontsize=8.5,
    color="#475569",
    fontstyle="italic",
    zorder=5,
)

last_cycle_x = cycle_start_x + 8 * (cycle_w + cycle_gap) - cycle_gap
arr_h(ax, last_cycle_x, 9.55, 7.75)

# 3. Ligated Oligo Construct
rbox(ax, 9.55, 7.15, 13.5, 8.35, "#ECFDF5", ec="#059669", lw=1.4)
oligo_cx = (9.55 + 13.5) / 2
ax.text(
    oligo_cx,
    7.95,
    "Ligated Oligo (~625 bp)",
    ha="center",
    va="center",
    fontsize=10.0,
    fontweight="bold",
    color="#065F46",
    zorder=5,
)
ax.text(
    oligo_cx,
    7.62,
    "5′ Flank  +  [C3 ··· C10]  +  3′ Flank",
    ha="center",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    color="#047857",
    zorder=5,
)
ax.text(
    oligo_cx,
    7.34,
    "Physical construct for nanopore sequencing",
    ha="center",
    va="center",
    fontsize=7.5,
    color="#059669",
    fontstyle="italic",
    zorder=5,
)

# -------------------------------------------------------------------------
# PANEL B: SEQUENCING & RAW CURRENT ACQUISITION
# -------------------------------------------------------------------------
ax.text(
    0.5,
    6.30,
    "b",
    fontsize=13,
    fontweight="bold",
    color="#0F172A",
    ha="left",
    va="center",
)
ax.text(
    0.75,
    6.30,
    "Nanopore Translocation & Raw Current Signal",
    fontsize=11,
    fontweight="bold",
    color="#1E293B",
    ha="left",
    va="center",
)

# Direct sequencing transition from Ligated Oligo into signal
seq_box_x1, seq_box_x2 = 10.2, 12.85
rbox(ax, seq_box_x1, 6.05, seq_box_x2, 6.55, "#F1F5F9", ec="#475569", lw=1.0)
ax.text(
    (seq_box_x1 + seq_box_x2) / 2,
    6.30,
    "Oxford Nanopore Sequencing",
    ha="center",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    color="#334155",
    zorder=5,
)

arr_v(ax, oligo_cx, 7.15, 6.55)
arr_v(ax, oligo_cx, 6.05, 5.50)

# Raw Signal Container
rbox(ax, 0.5, 4.45, 13.5, 5.50, "#FAFAFA", ec="#CBD5E1", lw=1.0)
ax.text(
    0.75,
    5.30,
    "Raw Ionic Current Signal (pA)",
    fontsize=8.5,
    fontweight="bold",
    color="#475569",
    va="center",
    zorder=5,
)
ax.text(
    13.25,
    4.65,
    "Time / Translocation →",
    fontsize=8.0,
    color="#94A3B8",
    ha="right",
    va="center",
    zorder=5,
)

# Stepped k-mer blockade current trace
np.random.seed(42)
plateau_levels = [
    102,
    80,
    115,
    72,
    94,
    66,
    108,
    85,
    122,
    77,
    100,
    64,
    112,
    83,
    96,
    70,
    105,
    88,
]
samples_per_step = 60
raw_sig = []
for lvl in plateau_levels:
  noise = np.random.normal(0, 2.8, samples_per_step)
  raw_sig.extend(lvl + noise)
raw_sig = np.array(raw_sig)

sig_x = np.linspace(0.8, 13.2, len(raw_sig))
sig_y_norm = 4.65 + (raw_sig - raw_sig.min()) / (
    raw_sig.max() - raw_sig.min()
) * 0.55
ax.plot(sig_x, sig_y_norm, color="#334155", lw=1.1, alpha=0.92, zorder=5)

# Branching arrows to decoding pipelines
arr_v(ax, 3.5, 4.45, 3.85, lw=1.5, col="#475569")
arr_v(ax, 10.5, 4.45, 3.85, lw=1.5, col="#475569")

# -------------------------------------------------------------------------
# PANEL C: DECODING PARADIGMS
# -------------------------------------------------------------------------
ax.text(
    0.5,
    4.05,
    "c",
    fontsize=13,
    fontweight="bold",
    color="#0F172A",
    ha="left",
    va="center",
)
ax.text(
    0.75,
    4.05,
    "Decoding Pipelines",
    fontsize=11,
    fontweight="bold",
    color="#1E293B",
    ha="left",
    va="center",
)

# Left Branch: Traditional Two-Stage Pipeline
rbox(ax, 0.5, 1.25, 6.5, 3.85, "#FFFBEB", ec="#D97706", lw=1.3)
ax.text(
    3.5,
    3.62,
    "Traditional Pipeline (Two-Stage)",
    ha="center",
    va="center",
    fontsize=10.0,
    fontweight="bold",
    color="#92400E",
    zorder=5,
)

rbox(ax, 0.8, 2.92, 6.2, 3.42, "#FFFFFF", ec="#F59E0B", lw=1.0)
ax.text(
    3.5,
    3.25,
    "Neural Basecalling (e.g., Guppy / Dorado)",
    ha="center",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    color="#78350F",
    zorder=5,
)
ax.text(
    3.5,
    3.07,
    "Maps ionic current to nucleotide probabilities",
    ha="center",
    va="center",
    fontsize=7.5,
    color="#92400E",
    zorder=5,
)

arr_v(ax, 3.5, 2.92, 2.65, lw=1.2, col="#B45309")

rbox(ax, 1.2, 2.35, 5.8, 2.65, "#FEF3C7", ec="#FCD34D", lw=0.9, linestyle="--")
ax.text(
    3.5,
    2.50,
    "Intermediate Basecalled Read (A, C, G, T)",
    ha="center",
    va="center",
    fontsize=8.0,
    fontstyle="italic",
    color="#713F12",
    zorder=5,
)

arr_v(ax, 3.5, 2.35, 2.08, lw=1.2, col="#B45309")

rbox(ax, 0.8, 1.48, 6.2, 2.08, "#FFFFFF", ec="#F59E0B", lw=1.0)
ax.text(
    3.5,
    1.86,
    "Motif Search & Alignment (ZE / AM)",
    ha="center",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    color="#78350F",
    zorder=5,
)
ax.text(
    3.5,
    1.65,
    "Dictionary search against 64 candidate sequences",
    ha="center",
    va="center",
    fontsize=7.5,
    color="#92400E",
    zorder=5,
)

# Right Branch: ConformerCaller (End-to-End)
rbox(ax, 7.5, 1.25, 13.5, 3.85, "#EFF6FF", ec="#2563EB", lw=1.3)
ax.text(
    10.5,
    3.62,
    "ConformerCaller (End-to-End)",
    ha="center",
    va="center",
    fontsize=10.0,
    fontweight="bold",
    color="#1E40AF",
    zorder=5,
)

rbox(ax, 7.8, 2.92, 13.2, 3.42, "#FFFFFF", ec="#3B82F6", lw=1.0)
ax.text(
    10.5,
    3.25,
    "1D CNN Subsampler",
    ha="center",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    color="#1E3A8A",
    zorder=5,
)
ax.text(
    10.5,
    3.07,
    "Strided convolutions for temporal feature reduction",
    ha="center",
    va="center",
    fontsize=7.5,
    color="#1D4ED8",
    zorder=5,
)

arr_v(ax, 10.5, 2.92, 2.65, lw=1.2, col="#1D4ED8")

rbox(ax, 7.8, 2.15, 13.2, 2.65, "#FFFFFF", ec="#3B82F6", lw=1.0)
ax.text(
    10.5,
    2.48,
    "Conformer Encoder Blocks",
    ha="center",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    color="#1E3A8A",
    zorder=5,
)
ax.text(
    10.5,
    2.30,
    "Multi-Head Self-Attention + Depthwise Convolution",
    ha="center",
    va="center",
    fontsize=7.5,
    color="#1D4ED8",
    zorder=5,
)

arr_v(ax, 10.5, 2.15, 1.88, lw=1.2, col="#1D4ED8")

rbox(ax, 7.8, 1.38, 13.2, 1.88, "#FFFFFF", ec="#3B82F6", lw=1.0)
ax.text(
    10.5,
    1.71,
    "CTC Loss & Decoding",
    ha="center",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    color="#1E3A8A",
    zorder=5,
)
ax.text(
    10.5,
    1.52,
    "Direct emission of 64 motif classes + blank token",
    ha="center",
    va="center",
    fontsize=7.5,
    color="#1D4ED8",
    zorder=5,
)

# Outputs
arr_v(ax, 3.5, 1.25, 0.85, lw=1.4, col="#D97706")
arr_v(ax, 10.5, 1.25, 0.85, lw=1.4, col="#2563EB")

rbox(ax, 1.5, 0.35, 5.5, 0.85, "#FEF3C7", ec="#D97706", lw=1.1)
ax.text(
    3.5,
    0.60,
    "Decoded Payload Motifs (C3–C10)",
    ha="center",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    color="#92400E",
    zorder=5,
)

rbox(ax, 8.5, 0.35, 12.5, 0.85, "#DBEAFE", ec="#2563EB", lw=1.1)
ax.text(
    10.5,
    0.60,
    "Decoded Payload Motifs (C3–C10)",
    ha="center",
    va="center",
    fontsize=8.5,
    fontweight="bold",
    color="#1E40AF",
    zorder=5,
)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(
    "results/paper/encoding_schematic.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
print("Saved: results/paper/encoding_schematic.png (300 DPI)")