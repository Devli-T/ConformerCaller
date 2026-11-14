"""
generate_architecture_diagram.py
=================================
Generates results/paper/architecture_diagram.png

Run from project root:
    python3.12 notebooks/generate_architecture_diagram.py
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt

Path('results/paper').mkdir(parents=True, exist_ok=True)

FW, FH = 6.4, 16.5
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')
fig.patch.set_facecolor('white')

BL, BR = 1.0, 5.4
BCX = (BL + BR) / 2
BW = BR - BL

# Color scheme (Fill, Edge, Text)
C_INPUT   = ('#F4F6F7', '#546E7A', '#263238')
C_CNN     = ('#1565C0', '#0D47A1', 'white')
C_PROJ    = ('#1976D2', '#0D47A1', 'white')
C_CONF_BG = ('#F9FBE7', '#33691E', '#1B5E20')
C_FF      = ('#E3F2FD', '#1565C0', '#0D47A1')
C_MHSA    = ('#FFF8E1', '#F57F17', '#E65100')
C_CONV    = ('#FCE4EC', '#C2185B', '#880E4F')
C_HEAD    = ('#6A1B9A', '#4A148C', 'white')
C_CTC     = ('#7B1FA2', '#4A148C', 'white')
C_OUT     = ('#E8F5E9', '#2E7D32', '#1B5E20')
C_DIM     = '#616161'

def rbox(y1, y2, fc, ec, lw=1.4, r=0.12, alpha=1.0):
    """Draws a rounded box strictly within [BL, BR] and [y1, y2]."""
    ax.add_patch(FancyBboxPatch(
        (BL, y1), BW, y2 - y1,
        boxstyle=f'round,pad=0,rounding_size={r}',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3, alpha=alpha
    ))

def tc(y, s, fs=9.5, c='black', w='normal', style='normal', zorder=5):
    ax.text(BCX, y, s, ha='center', va='center',
            fontsize=fs, color=c, fontweight=w, fontstyle=style, zorder=zorder)

def darrow(y_from, y_to, dim=None):
    """Draws an arrow with clearance from borders."""
    pad = 0.06
    ax.annotate(
        '', xy=(BCX, y_to + pad), xytext=(BCX, y_from - pad),
        arrowprops=dict(arrowstyle='->', color='#37474F', lw=1.4,
                        shrinkA=0, shrinkB=0, mutation_scale=12),
        zorder=6
    )
    if dim:
        ax.text(BCX + 0.18, (y_from + y_to) / 2, dim, ha='left', va='center',
                fontsize=8.2, color=C_DIM, fontstyle='italic', zorder=6)

def sub_band(y1, y2, x_pad, fc, ec, title, spec):
    xl, xr = BL + x_pad, BR - x_pad
    ax.add_patch(FancyBboxPatch(
        (xl, y1), xr - xl, y2 - y1,
        boxstyle='round,pad=0,rounding_size=0.08',
        facecolor=fc, edgecolor=ec, linewidth=1.1, zorder=4
    ))
    scx = (xl + xr) / 2
    cy = (y1 + y2) / 2
    ax.text(scx, cy + 0.12, title, ha='center', va='center',
            fontsize=9.0, fontweight='bold', color=ec, zorder=5)
    ax.text(scx, cy - 0.14, spec, ha='center', va='center',
            fontsize=7.8, color='#37474F', zorder=5)

# --- Title ---
ax.text(BCX, 15.90, 'ConformerCaller Architecture',
        ha='center', va='center', fontsize=13.5, fontweight='bold', color='#212121', zorder=5)

# --- 1. Raw Input ---
rbox(14.70, 15.45, *C_INPUT[:2])
tc(15.18, 'Raw Current Signal Input', fs=10.5, c=C_INPUT[2], w='bold')
tc(14.90, 'Nanopore Squiggle Sequence: (B, 1, T)', fs=8.5, c='#455A64')
darrow(14.70, 13.90, '(B, 1, T)')

# --- 2. CNN Subsampler ---
rbox(12.30, 13.90, *C_CNN[:2])
tc(13.62, '1D CNN Subsampler (×5 Layers)', fs=10.5, c='white', w='bold')
tc(13.30, 'Channels: 1 → 32 → 64 → 128 → 256 → 256', fs=8.5, c='#E3F2FD')
tc(12.98, 'Kernels: [11, 9, 7, 5, 5]   ·   Strides: [1, 2, 2, 2, 2]', fs=8.5, c='#E3F2FD')
tc(12.65, '16× Temporal Downsampling', fs=9.0, c='white', w='bold')
tc(12.42, 'BatchNorm  +  SiLU activations', fs=8.0, c='#BBDEFB')
darrow(12.30, 11.45, '(B, 256, T/16)')

# --- 3. Linear Projection ---
rbox(10.75, 11.45, *C_PROJ[:2])
tc(11.18, 'Linear Projection', fs=10.0, c='white', w='bold')
tc(10.92, '256 → 256   ·   Dropout (p = 0.1)', fs=8.5, c='#E3F2FD')
darrow(10.75, 9.95, '(B, T/16, 256)')

# --- 4. Conformer Encoder Stack ---
CONF_Y1, CONF_Y2 = 4.80, 9.95
rbox(CONF_Y1, CONF_Y2, *C_CONF_BG[:2], lw=1.8, r=0.14)
tc(CONF_Y2 - 0.28, 'Conformer Encoder Stack (×6 Blocks)', fs=11.0, c=C_CONF_BG[2], w='bold')
tc(CONF_Y2 - 0.55, 'd_model = 256  ·  8 heads  ·  FFN expansion 4×  ·  Dropout = 0.1', fs=8.2, c='#2E7D32')

SB_H = 0.76
SB_GAP = 0.22
SB_PAD = 0.25
sb_top = CONF_Y2 - 0.78

sub_blocks = [
    (C_FF[0], C_FF[1], 'Feed-Forward Module 1', 'Macaron-style ½-step residual · 4× expansion · SiLU'),
    (C_MHSA[0], C_MHSA[1], 'Multi-Head Self-Attention', '8 heads · d_model = 256 · Relative positional encoding'),
    (C_CONV[0], C_CONV[1], 'Convolution Module', 'Depthwise Conv (kernel = 31) · BatchNorm · SiLU'),
    (C_FF[0], C_FF[1], 'Feed-Forward Module 2', 'Macaron-style ½-step residual · LayerNorm'),
]

for i, (fc, ec, title, spec) in enumerate(sub_blocks):
    y2 = sb_top - i * (SB_H + SB_GAP)
    y1 = y2 - SB_H
    sub_band(y1, y2, SB_PAD, fc, ec, title, spec)
    if i < 3:
        ax.annotate(
            '', xy=(BCX, y1 - SB_GAP + 0.02), xytext=(BCX, y1 - 0.02),
            arrowprops=dict(arrowstyle='->', color='#757575', lw=1.1,
                            shrinkA=0, shrinkB=0, mutation_scale=9),
            zorder=5
        )

darrow(CONF_Y1, 3.95, '(T/16, B, 256)')

# --- 5. Classification Head (Linear + Log-Softmax merged) ---
rbox(3.10, 3.95, *C_HEAD[:2])
tc(3.62, 'Linear Classification Head', fs=10.0, c='white', w='bold')
tc(3.32, 'Linear (256 → 65 motif classes)  +  Log-Softmax', fs=8.5, c='#E1BEE7')
darrow(3.10, 2.25, '(T/16, B, 65)')

# --- 6. CTC Decoder ---
rbox(1.45, 2.25, *C_CTC[:2])
tc(1.95, 'CTC Decoder', fs=10.0, c='white', w='bold')
tc(1.65, 'Greedy / Confidence Thresholding (τ = 0.70)', fs=8.5, c='#E1BEE7')
darrow(1.45, 0.85)

# --- 7. Output Badge ---
rbox(0.20, 0.85, *C_OUT[:2], lw=1.3, r=0.10)
tc(0.60, 'Decoded Motif Sequence Output', fs=9.2, c=C_OUT[2], w='bold')
tc(0.38, 'ltm8_3x1,  ltm8_4x6,  ltm8_5x2,  …', fs=8.2, c='#2E7D32', style='italic')

# --- Footer Metric ---
ax.text(BCX, -0.05, 'Total Trainable Parameters: 11.02 M',
        ha='center', va='center', fontsize=8.5, color='#616161', fontstyle='italic', zorder=5)

plt.tight_layout(rect=[0, 0.01, 1, 0.99])
plt.savefig('results/paper/architecture_diagram.png', dpi=300,
            bbox_inches='tight', facecolor='white')
print('Saved: results/paper/architecture_diagram.png')