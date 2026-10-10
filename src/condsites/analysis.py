"""Analyse pathway_v2 scores: speaker-level pathway contrasts and reference-count effects.

Inputs: jobs.jsonl (design) and a scores CSV from evaluate_modern.py whose
clip_id is the job_id. Speakers (LibriTTS-R) or prompt speakers (Seed-TTS-eval
Common Voice client id) are the unit of analysis: repetitions are averaged
within speaker x system x condition before paired contrasts.

Outputs (to --output-dir):
- pathway_contrasts.csv: x-vector-only minus full, per arm x system x metric,
  with speaker-level mean, 95% bootstrap CI, paired d_z, Wilcoxon p and
  Holm-adjusted p (family: all pathway contrasts);
- refcount_effects.csv: n=3 minus n=1 within pathway x selection x system;
- condition_means.csv: speaker-level means per arm x system x condition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ("sim_heldout", "sim_prompt", "utmos22", "utmosv2", "wer")


def holm(p_values: pd.Series) -> pd.Series:
    order = p_values.sort_values().index
    adjusted = pd.Series(index=p_values.index, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def contrast(differences: np.ndarray, rng: np.random.Generator) -> dict:
    differences = differences[np.isfinite(differences)]
    boot = [rng.choice(differences, len(differences)).mean() for _ in range(2000)]
    nonzero = differences[differences != 0]
    return {
        "n_speakers": len(differences),
        "delta": differences.mean(),
        "ci_low": np.percentile(boot, 2.5),
        "ci_high": np.percentile(boot, 97.5),
        "d_z": differences.mean() / differences.std(ddof=1) if differences.std(ddof=1) > 0 else np.nan,
        "p": wilcoxon(nonzero).pvalue if len(nonzero) >= 6 else np.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    jobs = pd.DataFrame([json.loads(line) for line in args.jobs.read_text().splitlines()])
    scores = pd.read_csv(args.scores).rename(columns={"clip_id": "job_id"})
    data = jobs.merge(scores, on="job_id", how="inner")
    for metric in METRICS:
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data["wer"] = data["wer"].clip(upper=1.0)  # cap runaway insertions so one loop cannot dominate a speaker mean
    print(f"scored jobs: {len(data)} of {len(jobs)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261008)

    pathway = data[data.arm.isin(["pathway_libritts", "pathway_seedtts"])]
    means = pathway.groupby(["arm", "system", "condition", "speaker_id"])[list(METRICS)].mean()
    means.groupby(["arm", "system", "condition"]).mean().to_csv(args.output_dir / "condition_means.csv")
    rows = []
    for (arm, system), group in means.groupby(level=["arm", "system"]):
        wide = group.droplevel(["arm", "system"]).unstack("condition")
        if "xvector_only" not in wide.columns.get_level_values(1):
            continue
        for metric in METRICS:
            if (metric, "full") not in wide.columns:
                continue
            differences = (wide[(metric, "xvector_only")] - wide[(metric, "full")]).to_numpy()
            rows.append({"arm": arm, "system": system, "metric": metric, **contrast(differences, rng)})
    contrasts = pd.DataFrame(rows, columns=["arm", "system", "metric", "n_speakers", "delta", "ci_low", "ci_high", "d_z", "p"])
    contrasts["p_holm"] = holm(contrasts["p"].fillna(1.0))
    contrasts.to_csv(args.output_dir / "pathway_contrasts.csv", index=False)

    refcount = data[data.arm == "refcount_libritts"]
    rows = []
    speaker_means = refcount.groupby(["system", "condition", "selection", "n_refs", "speaker_id"])[list(METRICS)].mean()
    for (system, condition, selection), group in speaker_means.groupby(level=["system", "condition", "selection"]):
        wide = group.droplevel(["system", "condition", "selection"]).unstack("n_refs")
        for metric in METRICS:
            if (metric, 3) not in wide.columns or (metric, 1) not in wide.columns:
                continue
            differences = (wide[(metric, 3)] - wide[(metric, 1)]).to_numpy()
            rows.append({"system": system, "condition": condition, "selection": selection, "metric": metric, **contrast(differences, rng)})
    effects = pd.DataFrame(rows, columns=["system", "condition", "selection", "metric", "n_speakers", "delta", "ci_low", "ci_high", "d_z", "p"])
    effects["p_holm"] = holm(effects["p"].fillna(1.0))
    effects.to_csv(args.output_dir / "refcount_effects.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(contrasts[contrasts.metric.isin(["sim_heldout", "utmosv2", "wer"])].round(4).to_string(index=False))
        print(effects[effects.metric.isin(["sim_heldout", "utmosv2"])].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
