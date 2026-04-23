"""FIG-N1: Architecture diagram of the final bio-signal deepfake detection pipeline.

Matplotlib-based clean schematic. Left-to-right horizontal flow.
Frozen components have a snowflake marker; trainable components do not.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "figures" / "fig_n1_architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Color palette (blues/grays, colorblind-safe)
FROZEN_FACE   = "#d9e4f1"
FROZEN_EDGE   = "#3a6fa5"
TRAIN_FACE    = "#f0e6d2"
TRAIN_EDGE    = "#9b6a2e"
AUX_FACE      = "#e5e5e5"
AUX_EDGE      = "#555555"
TEXT          = "#222222"
ARROW         = "#555555"


def rounded_box(ax, x, y, w, h, text, face, edge, frozen=False, fs=9):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.4, facecolor=face, edgecolor=edge, zorder=2,
    )
    ax.add_patch(box)
    label = ("\u2744  " if frozen else "") + text  # snowflake prefix
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fs, color=TEXT,
            zorder=3, wrap=True)


def arrow(ax, x1, y1, x2, y2, label=None, style="-|>", fs=8, curve=0.0):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        linewidth=1.2, color=ARROW,
        connectionstyle=f"arc3,rad={curve}", zorder=1,
    )
    ax.add_patch(a)
    if label:
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2 + 0.10
        ax.text(xm, ym, label, ha="center", va="bottom",
                fontsize=fs, color=TEXT, style="italic", zorder=3)


fig, ax = plt.subplots(figsize=(13.5, 5.8), dpi=200)
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.set_axis_off()

# Row y-coordinates
Y_MAIN   = 3.3
Y_RPPG   = 5.2
Y_BLINK  = 1.4
H_BOX    = 1.1

# (1) Input
rounded_box(ax, 0.2, Y_MAIN, 2.0, H_BOX,
            "Face video\nclip (T=16,\n224\u00d7224\u00d73)",
            AUX_FACE, AUX_EDGE, fs=8.5)

# (2) EfficientNet-B4 backbone (FROZEN)
rounded_box(ax, 2.8, Y_MAIN, 2.8, H_BOX,
            "EfficientNet-B4\nmulti-task v13\n(frozen)",
            FROZEN_FACE, FROZEN_EDGE, frozen=True, fs=9)

# (2b) rPPG branch
rounded_box(ax, 2.8, Y_RPPG, 2.8, H_BOX,
            "rPPG features\nCHROM + POS\n(12-d spectral)",
            AUX_FACE, AUX_EDGE, fs=8.5)

# (2c) Blink branch
rounded_box(ax, 2.8, Y_BLINK, 2.8, H_BOX,
            "Blink features\nMediaPipe EAR\n(16-d stats)",
            AUX_FACE, AUX_EDGE, fs=8.5)

# (3) Concatenation node
rounded_box(ax, 6.4, Y_MAIN, 2.4, H_BOX,
            "Concatenate\n\n1820-d",
            AUX_FACE, AUX_EDGE, fs=9)

# (4) Linear probe (trainable)
rounded_box(ax, 9.6, Y_MAIN, 2.4, H_BOX,
            "Linear probe\n(dense layer,\ntrained)",
            TRAIN_FACE, TRAIN_EDGE, fs=9)

# (5) Platt scaling
rounded_box(ax, 12.8, Y_MAIN, 2.6, H_BOX,
            "Platt scaling\n(post-hoc\ncalibration)",
            TRAIN_FACE, TRAIN_EDGE, fs=9)

# Arrows — main row
arrow(ax, 2.2, Y_MAIN + H_BOX / 2, 2.8, Y_MAIN + H_BOX / 2)
arrow(ax, 5.6, Y_MAIN + H_BOX / 2, 6.4, Y_MAIN + H_BOX / 2, label="1792-d")
arrow(ax, 8.8, Y_MAIN + H_BOX / 2, 9.6, Y_MAIN + H_BOX / 2, label="1820-d")
arrow(ax, 12.0, Y_MAIN + H_BOX / 2, 12.8, Y_MAIN + H_BOX / 2, label="logit")

# Branch arrows: input → rPPG/blink
arrow(ax, 1.2, Y_MAIN + H_BOX, 2.8, Y_RPPG + H_BOX / 2, curve=0.25)
arrow(ax, 1.2, Y_MAIN, 2.8, Y_BLINK + H_BOX / 2, curve=-0.25)

# rPPG/blink → concat
arrow(ax, 5.6, Y_RPPG + H_BOX / 2, 6.8, Y_MAIN + H_BOX, label="12-d", curve=-0.25)
arrow(ax, 5.6, Y_BLINK + H_BOX / 2, 6.8, Y_MAIN, label="16-d", curve=0.25)

# Final output
ax.annotate("", xy=(15.9, Y_MAIN + H_BOX / 2),
            xytext=(15.4, Y_MAIN + H_BOX / 2),
            arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.4))
ax.text(15.95, Y_MAIN + H_BOX / 2, "p(fake)",
        ha="left", va="center", fontsize=10, fontweight="bold", color=TEXT)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=FROZEN_FACE, edgecolor=FROZEN_EDGE,
                   label="Frozen (pre-trained, \u2744)"),
    mpatches.Patch(facecolor=TRAIN_FACE, edgecolor=TRAIN_EDGE,
                   label="Trained (linear probe, Platt)"),
    mpatches.Patch(facecolor=AUX_FACE, edgecolor=AUX_EDGE,
                   label="Input / feature extractor"),
]
ax.legend(handles=legend_elements, loc="lower center",
          ncol=3, frameon=False, fontsize=9,
          bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"[fig_n1] wrote {OUT}")
