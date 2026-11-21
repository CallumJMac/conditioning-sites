"""
Analyse paired-comparison listening study results (pre-registered pipeline).

Input: a directory of results_*.json files (dummy tests) and/or a CSV export of
the "Raw" sheet (column "json"). Scores are recoded so that positive = x-vector
pathway preferred (CMOS for naturalness, CSMOS for speaker match).

Exclusions (applied in order, all reported):
1. screened out at the headphone check (Huggins pitch, < 5/6 correct);
2. failed traps: more than one of (degraded trap preferring the degraded clip,
   i.e. score not >= 1 towards natural; identical trap with |score| >= 2);
3. completion time under one third of the median.

Analysis per axis: linear mixed model score ~ 1 with crossed random intercepts
for listener and pair (statsmodels variance components); the intercept is the
mean preference with its 95% CI. Robustness: sign test on per-pair means.
Bonferroni across the two axes.

Headline test (pathway_v2 design): per system x dataset cell means, and a mixed
model score ~ C(system) on the similarity axis, testing whether removing the
prompt costs more perceived identity in CosyVoice 3 (embedding in the decoder
only) than in Qwen3-TTS (embedding conditions the language model).

Sequential stopping rule (fixed before data collection): analyse once 40
listeners remain after exclusions. If the headline system effect has p < 0.0294,
stop; otherwise recruit 40 more and test once more at p < 0.0294 (Pocock
boundary, overall alpha 0.05 over two looks). Report both looks.

Usage:
    python analyze_results.py --results path/to/dir_or_raw.csv --config configs/study.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def load_submissions(source: Path) -> list[dict]:
    if source.is_dir():
        return [json.loads(path.read_text()) for path in sorted(source.glob("results_*.json"))]
    raw = pd.read_csv(source)
    return [json.loads(text) for text in raw["json"]]


def trap_failures(responses: list[dict]) -> int:
    failures = 0
    for response in responses:
        if response["kind"] == "trap_degraded":
            towards_natural = -response["score"] if response["a"] == "natural" else response["score"]
            failures += towards_natural < 1
        elif response["kind"] == "trap_identical":
            failures += abs(response["score"]) >= 2
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    submissions = load_submissions(args.results)
    print(f"submissions: {len(submissions)}")
    kept = [item for item in submissions if not item["screenedOut"]]
    print(f"excluded at headphone check: {len(submissions) - len(kept)}")
    kept_after_traps = [item for item in kept if trap_failures(item["responses"]) <= 1]
    print(f"excluded for trap failures: {len(kept) - len(kept_after_traps)}")
    if kept_after_traps:
        median = np.median([item["durationMs"] for item in kept_after_traps])
        final = [item for item in kept_after_traps if item["durationMs"] >= median / 3]
    else:
        final = []
    print(f"excluded as too fast: {len(kept_after_traps) - len(final)}")
    print(f"listeners analysed: {len(final)}  (lists: {pd.Series([item['list'] for item in final]).value_counts().to_dict()})")
    if not final:
        return 0

    config = json.loads(args.config.read_text())
    rows = []
    for item in final:
        for response in item["responses"]:
            if response["kind"] != "pair":
                continue
            # Scale: negative = A preferred, positive = B preferred. Recode to x-vector preferred.
            score = response["score"] if response["b"] == "xvec" else -response["score"]
            pair = config["pairs"][response["pairId"]]
            rows.append({"listener": item["participantId"], "axis": response["axis"], "pair": response["pairId"], "score": score,
                         "system": pair.get("system", ""), "arm": pair.get("arm", "")})
    ratings = pd.DataFrame(rows)

    import statsmodels.formula.api as smf

    for axis in ("naturalness", "similarity"):
        data = ratings[ratings.axis == axis].copy()
        if data.empty:
            continue
        data["group"] = 1
        print(f"\n== {axis}: {len(data)} ratings, {data.listener.nunique()} listeners, {data.pair.nunique()} pairs")
        print(f"mean score (positive = x-vector preferred): {data.score.mean():.3f}")
        try:
            model = smf.mixedlm(
                "score ~ 1", data, groups="group",
                vc_formula={"listener": "0 + C(listener)", "pair": "0 + C(pair)"},
            ).fit(reml=True)
            estimate = model.params["Intercept"]
            low, high = model.conf_int().loc["Intercept"]
            p_value = model.pvalues["Intercept"]
            print(f"LMM intercept {estimate:.3f} [95% CI {low:.3f}, {high:.3f}], p={p_value:.3g}, Bonferroni p={min(1.0, 2 * p_value):.3g}")
        except Exception as error:  # small dummy datasets can fail to converge
            print(f"LMM not estimable: {error}")
        pair_means = data.groupby("pair").score.mean()
        nonzero = pair_means[pair_means != 0]
        if len(nonzero):
            sign = binomtest(int((nonzero > 0).sum()), len(nonzero))
            print(f"sign test over pairs: {(nonzero > 0).sum()}/{len(nonzero)} favour x-vector, p={sign.pvalue:.3g}")
        if data.system.nunique() > 1:
            print("per cell (mean score, positive = prompt-free preferred):")
            print(data.groupby(["system", "arm"]).score.agg(["mean", "count"]).round(3).to_string())
            try:
                model = smf.mixedlm(
                    "score ~ C(system)", data, groups="group",
                    vc_formula={"listener": "0 + C(listener)", "pair": "0 + C(pair)"},
                ).fit(reml=True)
                term = [name for name in model.params.index if name.startswith("C(system)")][0]
                print(f"system effect {term}: {model.params[term]:.3f} [95% CI {model.conf_int().loc[term, 0]:.3f}, {model.conf_int().loc[term, 1]:.3f}], p={model.pvalues[term]:.3g}")
            except Exception as error:
                print(f"system-effect model not estimable: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
