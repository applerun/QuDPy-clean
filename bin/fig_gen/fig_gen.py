#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate simple double-sided Feynman diagrams for study notes.

Output figures:
- fig_3_1_basic_structure.png
- fig_3_2_single_interaction_ket_bra.png
- fig_3_3_two_interactions_LL_LR_RL_RR.png
- fig_3_4_third_order_examples.png
"""

from pathlib import Path
import matplotlib.pyplot as plt


OUTDIR = Path("feynman_figures")
OUTDIR.mkdir(exist_ok=True)


def setup_ax(ax, title=None):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)

    # ket / bra vertical lines
    x_ket = 0.35
    x_bra = 0.65
    ax.plot([x_ket, x_ket], [0.1, 0.9], linewidth=2)
    ax.plot([x_bra, x_bra], [0.1, 0.9], linewidth=2)

    # labels
    ax.text(x_ket, 0.94, "ket", ha="center", va="bottom", fontsize=11)
    ax.text(x_bra, 0.94, "bra", ha="center", va="bottom", fontsize=11)

    # time arrow
    ax.annotate(
        "", xy=(0.92, 0.88), xytext=(0.92, 0.12),
        arrowprops=dict(arrowstyle="->", linewidth=1.5)
    )
    ax.text(0.94, 0.90, "time", ha="left", va="center", fontsize=10)

    return x_ket, x_bra


def draw_interaction(ax, side="ket", y=0.3, label=None, direction="in"):
    """
    side: 'ket' or 'bra'
    direction: 'in' or 'out'
    """
    x = 0.35 if side == "ket" else 0.65
    dx = -0.12 if side == "ket" else 0.12
    x0 = x + dx
    x1 = x

    if direction == "in":
        start, end = (x0, y), (x1, y)
    else:
        start, end = (x1, y), (x0, y)

    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="->", linewidth=1.5)
    )

    if label is not None:
        tx = x0 - 0.02 if side == "ket" else x0 + 0.02
        ha = "right" if side == "ket" else "left"
        ax.text(tx, y + 0.03, label, ha=ha, va="bottom", fontsize=10)


def draw_state_label(ax, ket_text=None, bra_text=None, y=0.15):
    if ket_text is not None:
        ax.text(0.31, y, ket_text, ha="right", va="center", fontsize=10)
    if bra_text is not None:
        ax.text(0.69, y, bra_text, ha="left", va="center", fontsize=10)


def fig_basic_structure():
    fig, ax = plt.subplots(figsize=(4, 6))
    setup_ax(ax, "Basic structure of a double-sided Feynman diagram")
    draw_state_label(ax, ket_text="|...⟩", bra_text="⟨...|", y=0.08)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_3_1_basic_structure.png", dpi=300)
    plt.close(fig)


def fig_single_interaction():
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))

    # ket-side interaction
    ax = axes[0]
    setup_ax(ax, "Single interaction: ket-side")
    draw_interaction(ax, side="ket", y=0.45, label="E1", direction="in")
    draw_state_label(ax, ket_text="|g⟩ → |e⟩", bra_text="⟨g|", y=0.08)
    ax.text(0.5, 0.02, "corresponds to  μρ", ha="center", va="bottom", fontsize=10)

    # bra-side interaction
    ax = axes[1]
    setup_ax(ax, "Single interaction: bra-side")
    draw_interaction(ax, side="bra", y=0.45, label="E1", direction="in")
    draw_state_label(ax, ket_text="|g⟩", bra_text="⟨g| → ⟨e|", y=0.08)
    ax.text(0.5, 0.02, "corresponds to  -ρμ", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_3_2_single_interaction_ket_bra.png", dpi=300)
    plt.close(fig)


def fig_two_interactions():
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    panels = [
        ("LL", [("ket", 0.30, "E1"), ("ket", 0.60, "E2")],
         "μ₂ μ₁ ρ₀"),
        ("LR", [("bra", 0.30, "E1"), ("ket", 0.60, "E2")],
         "- μ₂ ρ₀ μ₁"),
        ("RL", [("ket", 0.30, "E1"), ("bra", 0.60, "E2")],
         "- μ₁ ρ₀ μ₂"),
        ("RR", [("bra", 0.30, "E1"), ("bra", 0.60, "E2")],
         "ρ₀ μ₁ μ₂"),
    ]

    for ax, (title, ops, expr) in zip(axes.flat, panels):
        setup_ax(ax, f"Two interactions: {title}")
        for side, y, lab in ops:
            draw_interaction(ax, side=side, y=y, label=lab, direction="in")
        ax.text(0.5, 0.03, expr, ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_3_3_two_interactions_LL_LR_RL_RR.png", dpi=300)
    plt.close(fig)


def fig_third_order_examples():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    examples = [
        (
            "Rephasing example",
            [("ket", 0.25, "E1"),
             ("bra", 0.50, "E2"),
             ("ket", 0.75, "E3")],
            "phase:  -φ1 + φ2 + φ3",
            "coherence → population → coherence",
        ),
        (
            "Nonrephasing example",
            [("bra", 0.25, "E1"),
             ("ket", 0.50, "E2"),
             ("ket", 0.75, "E3")],
            "phase:  +φ1 - φ2 + φ3",
            "coherence → population → coherence",
        ),
        (
            "Population pathway",
            [("ket", 0.25, "E1"),
             ("bra", 0.50, "E2"),
             ("ket", 0.75, "E3")],
            "population during t2",
            "|e⟩⟨e| or |g⟩⟨g| in waiting time",
        ),
        (
            "Double-coherence example",
            [("ket", 0.25, "E1"),
             ("ket", 0.50, "E2"),
             ("bra", 0.75, "E3")],
            "double quantum coherence",
            "|f⟩⟨g| during evolution",
        ),
    ]

    for ax, (title, ops, line1, line2) in zip(axes.flat, examples):
        setup_ax(ax, title)
        for side, y, lab in ops:
            draw_interaction(ax, side=side, y=y, label=lab, direction="in")
        ax.text(0.5, 0.05, line1, ha="center", va="bottom", fontsize=10)
        ax.text(0.5, 0.00, line2, ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_3_4_third_order_examples.png", dpi=300)
    plt.close(fig)


def main():
    fig_basic_structure()
    fig_single_interaction()
    fig_two_interactions()
    fig_third_order_examples()
    print(f"Figures saved to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()