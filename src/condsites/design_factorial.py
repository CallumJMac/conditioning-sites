"""Build the CosyVoice-300M embedding-site factorial jobs from the pathway_v2 design.

Cells: prompt {on, off} x language-model embedding {on, off} x flow embedding {on, off}
on every LibriTTS-R pathway cell (34 speakers x 5 targets x 3 seeds) and a fixed random
300-item subset of Seed-TTS-eval test-en (seed 42). Targets, references and held-out
recordings are identical to pathway_v2, so the P1L1F1 and P0L1F1 cells replicate its
CosyVoice-300M full and x-vector-only conditions.
"""

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

SITES = [f"site:P{prompt}L{lm}F{flow}" for prompt, lm, flow in itertools.product("10", "10", "10")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pathway-jobs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seedtts-items", type=int, default=300)
    args = parser.parse_args()

    base = [json.loads(line) for line in args.pathway_jobs.read_text().splitlines()]
    cells = [job for job in base if job["system"] == "cosyvoice1" and job["condition"] == "full" and job["arm"] in ("pathway_libritts", "pathway_seedtts")]
    libritts = [job for job in cells if job["arm"] == "pathway_libritts"]
    seedtts = sorted((job for job in cells if job["arm"] == "pathway_seedtts"), key=lambda job: job["target_id"])
    seedtts = random.Random(20261009).sample(seedtts, args.seedtts_items)
    jobs = []
    for cell in libritts + seedtts:
        for site in SITES:
            job = {**cell, "arm": cell["arm"].replace("pathway", "sites"), "system": "cosyvoice1sites", "condition": site}
            job["job_id"] = hashlib.sha256(f"{cell['job_id']}|{site}".encode()).hexdigest()[:16]
            jobs.append(job)
    with args.output.open("w") as handle:
        for job in jobs:
            handle.write(json.dumps(job) + "\n")
    print(len(libritts), "libritts cells,", len(seedtts), "seedtts cells,", len(jobs), "jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
