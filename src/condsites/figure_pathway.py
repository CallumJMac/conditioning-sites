"""Headline figure for pathway_v2: what survives when prompt-level conditioning is removed.

(a) Delta speaker similarity (x-vector only minus full prompt), per system and
    dataset, grouped by where the speaker embedding enters the model.
(b) Delta naturalness (UTMOS22) for the same contrasts: the gain from dropping
    the prompt is larger on noisy Seed-TTS-eval prompts than on clean LibriTTS-R.

Speaker-level means with 95% bootstrap CIs from pathway_contrasts.csv.

Usage:
    python -m condsites.figure_pathway --contrasts results/analysis/pathway_contrasts.csv \
        --output paper/figures/pathway_headline.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LM_COLOUR = "#2a78d6"
DECODER_COLOUR = "#eb6834"
INK = "#2b2b2b"
MUTED = "#8a8a85"
SYSTEMS = [
    ("qwen06", "Qwen3-TTS 0.6B", "lm"),
    ("qwen17", "Qwen3-TTS 1.7B", "lm"),
    ("cosyvoice1", "CosyVoice v1", "lm"),
    ("cosyvoice3", "CosyVoice 3", "decoder"),
    ("xtts", "XTTS-v2", "decoder"),
    ("chatterbox", "Chatterbox$^\\dagger$", "breaks"),
]
DATASETS = [("pathway_libritts", "LibriTTS-R", "o", -0.17), ("pathway_seedtts", "Seed-TTS-eval", "s", 0.17)]
COLOURS = {"lm": LM_COLOUR, "decoder": DECODER_COLOUR, "breaks": MUTED}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contrasts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    contrasts = pd.read_csv(args.contrasts).set_index(["arm", "system", "metric"])
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    figure, (panel_a, panel_b) = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
    positions = list(range(len(SYSTEMS)))[::-1]

    for panel, metric, label, clip in ((panel_a, "sim_heldout", r"$\Delta$ speaker similarity (SIM-o)", None),
                                       (panel_b, "utmos22", r"$\Delta$ naturalness (UTMOS)", -0.25)):
        panel.axvline(0, color=MUTED, linewidth=0.8, zorder=0)
        panel.grid(axis="x", color="#e6e6e3", linewidth=0.5, zorder=0)
        panel.spines[["top", "right"]].set_visible(False)
        panel.set_xlabel(label)
        for position, (system, _, group) in zip(positions, SYSTEMS, strict=True):
            for arm, _, marker, offset in DATASETS:
                row = contrasts.loc[(arm, system, metric)]
                value, low, high = row.delta, row.ci_low, row.ci_high
                if clip is not None and value < clip:
                    panel.annotate("", xy=(clip, position + offset), xytext=(clip + 0.06, position + offset),
                                   arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1})
                    continue
                panel.errorbar(value, position + offset, xerr=[[value - low], [high - value]], fmt=marker, markersize=5,
                               color=COLOURS[group], markerfacecolor=COLOURS[group] if marker == "o" else "white",
                               markeredgewidth=1.3, elinewidth=1, capsize=0, zorder=3)
        if clip is not None:
            panel.set_xlim(left=clip)
    panel_a.set_yticks(positions)
    panel_a.set_yticklabels([name for _, name, _ in SYSTEMS])
    panel_a.set_title("(a) Identity without the prompt", loc="left", fontsize=8, color=INK)
    panel_b.set_title("(b) Naturalness without the prompt", loc="left", fontsize=8, color=INK)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=LM_COLOUR, label="embedding conditions the LM"),
        plt.Line2D([], [], marker="o", linestyle="", color=DECODER_COLOUR, label="embedding in decoder only"),
        plt.Line2D([], [], marker="o", linestyle="", color=INK, label="LibriTTS-R (clean)"),
        plt.Line2D([], [], marker="s", linestyle="", color=INK, markerfacecolor="white", label="Seed-TTS-eval (noisy)"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.55, -0.06))
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
