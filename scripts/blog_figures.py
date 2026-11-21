# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.10,<3.11", "numpy", "pandas"]
# ///
"""Blog versions of the paper's two headline figures, as theme-aware SVGs.

Reads only the released analysis tables, so the figures match the paper's
numbers exactly. Text is embedded as Helvetica outlines; colours become CSS
custom properties with a dark-mode variant matched to callumjmac.github.io.

    uv run scripts/blog_figures.py --output-dir ../CallumJMac.github.io/public/assets/conditioning-sites
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
LIGHT = {
    "surface": "#fdfdfd",
    "ink": "#1d1d1f",
    "ink-secondary": "#55565b",
    "ink-muted": "#8c8d92",
    "rule": "#b9babe",
    "series-1": "#2a78d6",
    "series-2": "#eb6834",
}
DARK = {
    "surface": "#212737",
    "ink": "#eaedf3",
    "ink-secondary": "#b8bdc9",
    "ink-muted": "#8a90a0",
    "rule": "#5a6278",
    "series-1": "#3987e5",
    "series-2": "#d95926",
}
INK, SECONDARY, MUTED = LIGHT["ink"], LIGHT["ink-secondary"], LIGHT["ink-muted"]
LM_SITE, DECODER_SITE = LIGHT["series-1"], LIGHT["series-2"]
FULL_WIDTH_INCHES = 6.0
DISPLAY_PIXELS_PER_POINT = 680 / (FULL_WIDTH_INCHES * 72)

# (key, display name, colour): ordered as in the paper.
SYSTEMS = [
    ("qwen06", "Qwen3-TTS 0.6B", LM_SITE),
    ("qwen17", "Qwen3-TTS 1.7B", LM_SITE),
    ("cosyvoice1", "CosyVoice-300M", LM_SITE),
    ("cosyvoice3", "CosyVoice 3", DECODER_SITE),
    ("xtts", "XTTS-v2", DECODER_SITE),
    ("chatterbox", "Chatterbox", MUTED),
]
DATASETS = [("pathway_libritts", "LibriTTS-R (clean)", "o", True, -0.14),
            ("pathway_seedtts", "Seed-TTS-eval (noisy)", "s", False, 0.14)]


def apply_style() -> None:
    plt.rcParams.update({
        "svg.fonttype": "path", "svg.hashsalt": "conditioning-sites-blog",
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"],
        "font.size": 8, "axes.labelsize": 8.5, "axes.labelcolor": INK,
        "axes.edgecolor": SECONDARY, "axes.linewidth": 0.6, "axes.facecolor": LIGHT["surface"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "xtick.color": SECONDARY, "ytick.color": SECONDARY, "xtick.labelcolor": INK,
        "ytick.labelcolor": INK, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "figure.facecolor": LIGHT["surface"],
        "savefig.facecolor": LIGHT["surface"], "text.color": INK, "mathtext.default": "regular",
    })


def panel_label(axis, letter: str, x_offset: float) -> None:
    axis.text(x_offset, 1.02, letter, transform=axis.transAxes, fontweight="bold",
              fontsize=9.5, color=INK, ha="left", va="bottom")


def theme_and_save(figure, path: Path, description: str) -> dict:
    figure.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.04,
                   metadata={"Date": None})
    plt.close(figure)
    svg = path.read_text()
    for token, light_hex in LIGHT.items():
        svg = re.sub(re.escape(light_hex), f"var(--{token})", svg, flags=re.IGNORECASE)
    block = lambda palette: " ".join(f"--{key}: {value};" for key, value in palette.items())  # noqa: E731
    style = (f"<style>:root {{ {block(LIGHT)} }} @media (prefers-color-scheme: dark) "
             f"{{ :root {{ {block(DARK)} }} }}</style>")
    escaped = description.replace("&", "&amp;").replace("<", "&lt;")
    svg = re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.DOTALL)
    svg = re.sub(r"(<svg[^>]*>)", lambda match: f"{match.group(1)}<desc>{escaped}</desc>{style}",
                 svg, count=1)
    path.write_text(svg)
    width, height = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
    return {"width": round(width * DISPLAY_PIXELS_PER_POINT),
            "height": round(height * DISPLAY_PIXELS_PER_POINT)}


def figure_pathway(output_dir: Path) -> dict:
    contrasts = pd.read_csv(REPO_ROOT / "results/analysis/pathway_contrasts.csv")
    contrasts = contrasts.set_index(["arm", "system", "metric"])
    figure, (left, right) = plt.subplots(1, 2, figsize=(FULL_WIDTH_INCHES, 2.55), sharey=True,
                                         gridspec_kw={"wspace": 0.08})
    rows = range(len(SYSTEMS))[::-1]
    for axis, metric, xlabel, limits in (
        (left, "sim_heldout", r"$\Delta$ speaker similarity (SIM-o)", (-0.36, 0.06)),
        (right, "utmos22", r"$\Delta$ predicted naturalness (UTMOS)", (-0.25, 0.58)),
    ):
        axis.axvline(0, color=LIGHT["rule"], linewidth=0.6, zorder=1)
        for row, (key, _name, colour) in zip(rows, SYSTEMS, strict=True):
            for arm, _label, marker, filled, offset in DATASETS:
                item = contrasts.loc[(arm, key, metric)]
                value, low, high = item["delta"], item["ci_low"], item["ci_high"]
                if value < limits[0]:
                    axis.annotate(f"{value:.1f}".replace("-", "\u2212"), xy=(limits[0], row + offset),
                                  xytext=(limits[0] + 0.07, row + offset), fontsize=6.5,
                                  color=SECONDARY, va="center",
                                  arrowprops={"arrowstyle": "->", "color": MUTED,
                                              "linewidth": 0.7})
                    continue
                axis.errorbar(value, row + offset, xerr=[[value - low], [high - value]],
                              color=colour, marker=marker, markersize=4,
                              markerfacecolor=colour if filled else LIGHT["surface"],
                              markeredgewidth=0.9, elinewidth=0.8, linestyle="none", zorder=3)
        axis.set_xlim(*limits)
        axis.set_xlabel(xlabel)
        axis.tick_params(axis="y", length=0)
    left.set_yticks(list(rows), [name for _, name, _ in SYSTEMS])
    left.set_ylim(-0.6, len(SYSTEMS) - 0.4)
    for axis in (left, right):
        axis.spines["left"].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=LM_SITE, markersize=4,
                   label="embedding reaches the LM"),
        plt.Line2D([], [], marker="o", linestyle="", color=DECODER_SITE, markersize=4,
                   label="embedding in decoder only"),
        plt.Line2D([], [], marker="o", linestyle="", color=INK, markersize=4,
                   label="LibriTTS-R (clean)"),
        plt.Line2D([], [], marker="s", linestyle="", color=INK, markerfacecolor=LIGHT["surface"],
                   markersize=4, label="Seed-TTS-eval (noisy)"),
    ]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 1.0), ncol=4,
                  fontsize=7, handletextpad=0.2, columnspacing=1.4)
    panel_label(left, "a", -0.02)
    panel_label(right, "b", -0.02)
    return theme_and_save(
        figure, output_dir / "pathway-contrasts.svg",
        "Change in speaker similarity and predicted naturalness when the acoustic prompt is "
        "removed and only the speaker embedding is kept, for six systems on two test sets. "
        "Systems whose language model receives the embedding lose 0.06 to 0.11 similarity or "
        "nothing; decoder-only systems lose 0.18 to 0.24. Naturalness rises most on noisy "
        "prompts. Chatterbox stops producing intelligible speech.",
    )


def figure_factorial(output_dir: Path) -> dict:
    summary = pd.read_csv(REPO_ROOT / "paper/tables/sites_summary.csv")
    summary = summary.set_index(["arm", "condition"])
    cells = [("L1F1", "LM +\ndecoder"), ("L0F1", "decoder\nonly"), ("L1F0", "LM\nonly"),
             ("L0F0", "none")]
    figure, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_INCHES, 2.5), sharey=True,
                                gridspec_kw={"wspace": 0.1})
    for axis, arm, letter in zip(axes, ("LibriTTS-R", "Seed-TTS-eval"), "ab", strict=True):
        for prompt, name, colour, marker in (("P1", "prompt kept", LM_SITE, "o"),
                                             ("P0", "prompt removed", DECODER_SITE, "s")):
            rows = [summary.loc[(arm, f"site:{prompt}{cell}")] for cell, _ in cells]
            values = [row["sim"] for row in rows]
            axis.errorbar(range(len(cells)), values,
                          yerr=[[row["sim"] - row["low"] for row in rows],
                                [row["high"] - row["sim"] for row in rows]],
                          color=colour, marker=marker, markersize=4, linewidth=1.0,
                          elinewidth=0.8, zorder=3)
            if letter == "b":
                axis.annotate(name, xy=(len(cells) - 1, values[-1]), xytext=(6, 0),
                              textcoords="offset points", va="center", fontsize=7.5,
                              color=colour)
        axis.set_xticks(range(len(cells)), [label for _, label in cells])
        axis.set_xlim(-0.35, len(cells) - 0.4 + (0.9 if letter == "b" else 0))
        axis.spines["bottom"].set_bounds(-0.35, len(cells) - 0.65)
        axis.tick_params(axis="x", length=0)
        axis.set_xlabel("Speaker embedding kept at")
        axis.text(0.02, 1.0, arm, transform=axis.transAxes, fontsize=8, color=INK, va="top")
        panel_label(axis, letter, -0.14 if letter == "a" else -0.04)
    axes[0].set_ylim(0, 0.52)
    axes[0].set_ylabel("Speaker similarity (SIM-o)")
    return theme_and_save(
        figure, output_dir / "site-factorial.svg",
        "CosyVoice-300M factorial over the acoustic prompt and the speaker embedding at the "
        "language model and the decoder. Without the prompt, removing either site costs "
        "similarity and the losses add, down to 0.14 with neither. With the prompt, removing "
        "the language-model embedding changes nothing while removing the decoder embedding "
        "still costs about 0.06 to 0.07.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    sizes = {"pathway-contrasts": figure_pathway(args.output_dir),
             "site-factorial": figure_factorial(args.output_dir)}
    print(json.dumps(sizes))


if __name__ == "__main__":
    main()
