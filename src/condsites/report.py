"""Generate figures and LaTeX tables for the long (TMLR) paper from result files.

Every number in the generated tables comes from the analysis outputs, so the
paper can be rebuilt from the committed results:
- results/analysis/{pathway_contrasts,condition_means,refcount_effects}.csv
- results/scores/factorial_scores.csv with jobs/factorial.jsonl
- results/scores/pathway_status.jsonl with jobs/pathway.jsonl

Usage:
    python -m condsites.report --results results --output paper
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

JOBS = Path(__file__).resolve().parents[2] / "jobs"

SYSTEM_NAMES = {
    "qwen06": "Qwen3-TTS 0.6B", "qwen17": "Qwen3-TTS 1.7B", "cosyvoice1": "CosyVoice-300M",
    "cosyvoice3": "CosyVoice 3", "xtts": "XTTS-v2", "chatterbox": "Chatterbox", "f5tts": "F5-TTS",
}
ARM_NAMES = {"pathway_libritts": "LibriTTS-R", "pathway_seedtts": "Seed-TTS-eval"}
SYSTEM_ORDER = ["qwen06", "qwen17", "cosyvoice1", "cosyvoice3", "xtts", "chatterbox"]
INK, MUTED = "#2b2b2b", "#8a8a85"
PROMPT_ON, PROMPT_OFF = "#2a78d6", "#eb6834"


def fmt(value: float, low: float, high: float, significant: bool) -> str:
    star = "$^\\ast$" if significant else ""
    return f"${value:+.3f}${star} \\scriptsize{{[{low:+.3f}, {high:+.3f}]}}"


def contrasts_table(results: Path, output: Path) -> None:
    contrasts = pd.read_csv(results / "analysis" / "pathway_contrasts.csv").set_index(["arm", "system", "metric"])
    lines = [
        "\\begin{tabular}{llcccc}", "\\toprule",
        "\\textbf{Data} & \\textbf{System} & $\\Delta$\\textbf{SIM-o} & $\\Delta$\\textbf{SIM-p} & $\\Delta$\\textbf{UTMOS} & $\\Delta$\\textbf{WER} \\\\", "\\midrule",
    ]
    for arm in ARM_NAMES:
        for index, system in enumerate(SYSTEM_ORDER):
            cells = []
            for metric in ("sim_heldout", "sim_prompt", "utmos22", "wer"):
                row = contrasts.loc[(arm, system, metric)]
                cells.append(fmt(row.delta, row.ci_low, row.ci_high, row.p_holm < 0.05))
            data_label = ARM_NAMES[arm] if index == 0 else ""
            lines.append(f"{data_label} & {SYSTEM_NAMES[system]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule" if arm == "pathway_libritts" else "\\bottomrule")
    lines.append("\\end{tabular}")
    (output / "tables" / "pathway_contrasts.tex").write_text("\n".join(lines) + "\n")


def means_table(results: Path, output: Path) -> None:
    means = pd.read_csv(results / "analysis" / "condition_means.csv").set_index(["arm", "system", "condition"])
    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "\\textbf{Data} & \\textbf{System} & \\textbf{Condition} & \\textbf{SIM-o} & \\textbf{UTMOS} & \\textbf{WER} \\\\", "\\midrule"]
    for arm in ARM_NAMES:
        first = True
        for system in SYSTEM_ORDER + ["f5tts"]:
            for condition in ("full", "xvector_only"):
                if (arm, system, condition) not in means.index:
                    continue
                row = means.loc[(arm, system, condition)]
                label = ARM_NAMES[arm] if first else ""
                first = False
                condition_label = "full" if condition == "full" else "x-vector only"
                lines.append(f"{label} & {SYSTEM_NAMES[system]} & {condition_label} & {row.sim_heldout:.3f} & {row.utmos22:.3f} & {100 * row.wer:.1f}\\% \\\\")
        lines.append("\\midrule" if arm == "pathway_libritts" else "\\bottomrule")
    lines.append("\\end{tabular}")
    (output / "tables" / "condition_means.tex").write_text("\n".join(lines) + "\n")


def refcount_table(results: Path, output: Path) -> None:
    effects = pd.read_csv(results / "analysis" / "refcount_effects.csv")
    effects = effects[effects.metric.isin(["sim_heldout", "utmos22"])]
    lines = ["\\begin{tabular}{lllcc}", "\\toprule",
             "\\textbf{System} & \\textbf{Route} & \\textbf{Selection} & $\\Delta$\\textbf{SIM-o} & $\\Delta$\\textbf{UTMOS} \\\\", "\\midrule"]
    for system in ("qwen06", "qwen17", "cosyvoice3", "f5tts"):
        for condition in ("full", "xvector_only"):
            for selection in ("longest", "random"):
                subset = effects[(effects.system == system) & (effects.condition == condition) & (effects.selection == selection)].set_index("metric")
                if subset.empty:
                    continue
                cells = [fmt(subset.loc[m].delta, subset.loc[m].ci_low, subset.loc[m].ci_high, subset.loc[m].p_holm < 0.05) for m in ("sim_heldout", "utmos22")]
                route = "prompt (concat)" if condition == "full" else "x-vector (avg)"
                lines.append(f"{SYSTEM_NAMES[system]} & {route} & {selection} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (output / "tables" / "refcount_full.tex").write_text("\n".join(lines) + "\n")


def sites_figure_and_table(results: Path, output: Path) -> None:
    jobs = pd.DataFrame([json.loads(line) for line in (JOBS / "factorial.jsonl").read_text().splitlines()])
    scores = pd.read_csv(results / "scores" / "factorial_scores.csv").rename(columns={"clip_id": "job_id"})
    data = jobs.merge(scores, on="job_id")
    for column in ("sim_heldout", "utmos22", "wer"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["wer"] = data["wer"].clip(upper=1.0)
    rng = np.random.default_rng(20261009)
    rows = []
    for arm, arm_label in (("sites_libritts", "LibriTTS-R"), ("sites_seedtts", "Seed-TTS-eval")):
        subset = data[data.arm == arm]
        speaker_means = subset.groupby(["condition", "speaker_id"])[["sim_heldout", "utmos22", "wer"]].mean()
        for condition, group in speaker_means.groupby(level="condition"):
            values = group.sim_heldout.to_numpy()
            boot = [rng.choice(values, len(values)).mean() for _ in range(2000)]
            rows.append({"arm": arm_label, "condition": condition, "sim": values.mean(), "low": np.percentile(boot, 2.5), "high": np.percentile(boot, 97.5),
                         "utmos": group.utmos22.mean(), "wer": group.wer.mean(), "n": len(values)})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "tables" / "sites_summary.csv", index=False)

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
    states = [("L1F1", "LM + flow"), ("L0F1", "flow only"), ("L1F0", "LM only"), ("L0F0", "none")]
    width = 0.38
    for axis, arm_label in zip(axes, ("LibriTTS-R", "Seed-TTS-eval"), strict=True):
        part = summary[summary.arm == arm_label].set_index("condition")
        for offset, prompt, colour, label in ((-width / 2, "P1", PROMPT_ON, "with prompt"), (width / 2, "P0", PROMPT_OFF, "prompt removed")):
            positions = np.arange(len(states)) + offset
            values = [part.loc[f"site:{prompt}{state}"].sim for state, _ in states]
            errors = [[part.loc[f"site:{prompt}{state}"].sim - part.loc[f"site:{prompt}{state}"].low for state, _ in states],
                      [part.loc[f"site:{prompt}{state}"].high - part.loc[f"site:{prompt}{state}"].sim for state, _ in states]]
            axis.bar(positions, values, width=width - 0.04, color=colour, label=label, zorder=2)
            axis.errorbar(positions, values, yerr=errors, fmt="none", ecolor=INK, elinewidth=0.8, zorder=3)
        axis.set_xticks(np.arange(len(states)))
        axis.set_xticklabels([label for _, label in states])
        axis.set_xlabel("embedding kept at")
        axis.set_title(arm_label, fontsize=8, color=INK, loc="left")
        axis.grid(axis="y", color="#e6e6e3", linewidth=0.5, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("SIM-o")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    figure.tight_layout()
    (output / "figures").mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "figures" / "sites_factorial.pdf", bbox_inches="tight")
    figure.savefig(output / "figures" / "sites_factorial.png", dpi=200, bbox_inches="tight")


def failures_table(results: Path, output: Path) -> None:
    status = pd.read_json(results / "scores" / "pathway_status.jsonl", lines=True)
    jobs = pd.DataFrame([json.loads(line) for line in (JOBS / "pathway.jsonl").read_text().splitlines()])[["job_id", "arm"]]
    status = status.merge(jobs, on="job_id")
    table = status.groupby(["system", "status"]).size().unstack(fill_value=0)
    lines = ["\\begin{tabular}{lrr}", "\\toprule", "\\textbf{System} & \\textbf{Synthesised} & \\textbf{Failed} \\\\", "\\midrule"]
    for system in SYSTEM_ORDER + ["f5tts"]:
        success = int(table.loc[system].get("success", 0)) if system in table.index else 0
        failed = int(table.loc[system].get("failed", 0)) if system in table.index else 0
        lines.append(f"{SYSTEM_NAMES[system]} & {success:,} & {failed} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (output / "tables" / "failures.tex").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    (args.output / "tables").mkdir(parents=True, exist_ok=True)
    contrasts_table(args.results, args.output)
    means_table(args.results, args.output)
    refcount_table(args.results, args.output)
    sites_figure_and_table(args.results, args.output)
    failures_table(args.results, args.output)
    print("assets written to", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
