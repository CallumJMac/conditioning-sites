"""Build the pathway_v2 synthesis job list.

Arms:
- pathway_libritts: full vs xvector_only, one fixed 6 s reference, the
  confirmatory 34-speaker x 5-target matrix, seeds 42-44;
- pathway_seedtts: full vs xvector_only on Seed-TTS-eval test-en (1,088 items,
  each with its own Common Voice prompt), seed 42; held-out = ground-truth
  recording of the target sentence;
- refcount_libritts: n in {1, 3} x selection in {random, longest} x pathway,
  references drawn from each speaker's pool of 3-9.5 s clips excluding targets
  and enrollment (so a three-reference prompt stays under CosyVoice's 30 s
  tokeniser limit), seed 42.

F5-TTS has no embedding pathway, so it only receives full-condition jobs.
Paths are written relative to the bundle root (data/...).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import soundfile as sf

PATHWAY_SYSTEMS = ("qwen06", "qwen17", "cosyvoice1", "cosyvoice3", "xtts", "chatterbox")
REFCOUNT_SYSTEMS = ("qwen06", "qwen17", "cosyvoice3", "f5tts")


def _job_id(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _relative(remote: str) -> str:
    remote_path = Path(remote)
    return str(Path("data/libritts_r_aligned") / remote_path.relative_to(remote_path.parents[2]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qwen-matrix", required=True, type=Path, help="confirmatory qwen run_matrix.json")
    parser.add_argument("--manifest", required=True, type=Path, help="manifest_40spk.json")
    parser.add_argument("--libritts-root", required=True, type=Path, help="local libritts_r_aligned dir")
    parser.add_argument("--seedtts-meta", required=True, type=Path, help="seedtts_testset/en/meta.lst")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    jobs: list[dict] = []

    def add(arm, system, condition, refs, target_text, target_id, speaker_id, seed, heldout, extra=None):
        jobs.append({
            "job_id": _job_id(arm, system, condition, target_id, seed, *(ref["id"] for ref in refs), json.dumps(extra or {}, sort_keys=True)),
            "arm": arm, "system": system, "condition": condition, "refs": refs,
            "target_text": target_text, "target_id": target_id, "speaker_id": speaker_id,
            "seed": seed, "heldout": heldout, **(extra or {}),
        })

    matrix = [entry for entry in json.loads(args.qwen_matrix.read_text()) if entry["duration_regime"] == "fixed"]
    cells = {(entry["speaker_id"], entry["target_id"], entry["seed"]): entry for entry in matrix if entry["condition"] == "full"}
    for (speaker, target, seed), entry in sorted(cells.items()):
        refs = [{"id": ref_id, "path": _relative(path), "text": text, "duration": duration}
                for ref_id, path, text, duration in zip(entry["reference_ids"], entry["reference_paths"], entry["reference_texts"], entry["reference_durations"], strict=True)]
        for system in PATHWAY_SYSTEMS + ("f5tts",):
            for condition in (("full",) if system == "f5tts" else ("full", "xvector_only")):
                add("pathway_libritts", system, condition, refs, entry["target_text"], target, speaker, seed, _relative(entry["enrollment_audio"]))

    for line in args.seedtts_meta.read_text(encoding="utf-8").splitlines():
        item_id, prompt_text, prompt_wav, target_text = line.split("|")[:4]
        refs = [{"id": Path(prompt_wav).stem, "path": f"data/seedtts_en/{prompt_wav}", "text": prompt_text, "duration": None}]
        for system in PATHWAY_SYSTEMS + ("f5tts",):
            for condition in (("full",) if system == "f5tts" else ("full", "xvector_only")):
                add("pathway_seedtts", system, condition, refs, target_text, item_id, item_id.split("-")[0], 42, f"data/seedtts_en/wavs/{item_id}.wav")

    excluded: dict[str, set[str]] = {}
    for entry in matrix:
        excluded.setdefault(entry["speaker_id"], set()).update({entry["target_id"], entry["enrollment_id"]})
    pools: dict[str, list[dict]] = {}
    for item in json.loads(args.manifest.read_text()):
        if item["speaker_id"] in excluded and item["id"] not in excluded[item["speaker_id"]]:
            duration = sf.info(str(args.libritts_root / item["path"])).duration
            if not 3.0 <= duration <= 9.5:
                continue
            pools.setdefault(item["speaker_id"], []).append(
                {"id": item["id"], "path": str(Path("data/libritts_r_aligned") / item["path"]), "text": item["text_normalized"], "duration": round(duration, 3)}
            )
    seed42 = {(speaker, target): entry for (speaker, target, seed), entry in cells.items() if seed == 42}
    for (speaker, target), entry in sorted(seed42.items()):
        pool = sorted(pools[speaker], key=lambda item: item["id"])
        longest = sorted(pool, key=lambda item: -item["duration"])
        rng = random.Random(f"{speaker}|{target}")
        shuffled = rng.sample(pool, len(pool))
        for n_refs in (1, 3):
            for selection, ordered in (("longest", longest), ("random", shuffled)):
                refs = ordered[:n_refs]
                for system in REFCOUNT_SYSTEMS:
                    for condition in (("full",) if system == "f5tts" else ("full", "xvector_only")):
                        add("refcount_libritts", system, condition, refs, entry["target_text"], target, speaker, 42,
                            _relative(entry["enrollment_audio"]), {"n_refs": n_refs, "selection": selection})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job) + "\n")
    counts: dict[tuple[str, str], int] = {}
    for job in jobs:
        counts[(job["arm"], job["system"])] = counts.get((job["arm"], job["system"]), 0) + 1
    for key, value in sorted(counts.items()):
        print(*key, value)
    print("total", len(jobs), "unique ids", len({job["job_id"] for job in jobs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
