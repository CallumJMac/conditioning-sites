"""Build the curated audio samples for docs/ (the GitHub Pages demo).

Items are drawn at random with a fixed seed, not chosen for effect: two
LibriTTS-R speakers (one female, one male, seed-42 cells) and two Seed-TTS-eval
items. For each item the page shows the prompt, the held-out recording, every
system with and without the prompt, and all eight cells of the CosyVoice
factorial. Audio is loudness-normalised to -23 LUFS and encoded as 64 kb/s MP3.

Usage:
    python scripts/build_samples.py --audio-dir /path/to/audio   # contains <system>/<job_id>.wav
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261010
SYSTEMS = ["qwen06", "qwen17", "cosyvoice1", "cosyvoice3", "xtts", "chatterbox", "f5tts"]
SITE_LABELS = {
    "site:P1L1F1": "prompt, LM and flow embedding",
    "site:P1L0F1": "prompt, flow embedding only",
    "site:P1L1F0": "prompt, LM embedding only",
    "site:P1L0F0": "prompt only",
    "site:P0L1F1": "LM and flow embedding",
    "site:P0L0F1": "flow embedding only",
    "site:P0L1F0": "LM embedding only",
    "site:P0L0F0": "no speaker information",
}


def encode(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-af", "loudnorm=I=-23:TP=-2:LRA=11",
                    "-ac", "1", "-ar", "24000", "-b:a", "64k", str(destination)], check=True)


def data_path(relative: str) -> Path:
    return ROOT / relative


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "samples")
    args = parser.parse_args()

    rng = random.Random(SEED)
    pathway = [json.loads(line) for line in (ROOT / "jobs/pathway.jsonl").read_text().splitlines()]
    factorial = [json.loads(line) for line in (ROOT / "jobs/factorial.jsonl").read_text().splitlines()]
    manifest = json.loads((ROOT / "data/libritts_r_aligned/manifest_40spk.json").read_text())
    sex = {item["speaker_id"]: item["sex"] for item in manifest}

    def pathway_jobs(arm, target):
        return {(job["system"], job["condition"]): job for job in pathway if job["arm"] == arm and job["target_id"] == target and job["seed"] == 42}

    factorial_targets = {job["target_id"] for job in factorial if job["seed"] == 42}
    libritts_targets = sorted({job["target_id"] for job in pathway if job["arm"] == "pathway_libritts" and job["seed"] == 42} & factorial_targets)
    female = rng.choice([t for t in libritts_targets if sex[t.split("_")[0]] == "F"])
    male = rng.choice([t for t in libritts_targets if sex[t.split("_")[0]] == "M"])
    seedtts_targets = sorted({job["target_id"] for job in factorial if job["arm"] == "sites_seedtts"})
    seed_items = rng.sample(seedtts_targets, 2)

    items = []
    for arm, target in [("pathway_libritts", female), ("pathway_libritts", male), ("pathway_seedtts", seed_items[0]), ("pathway_seedtts", seed_items[1])]:
        jobs = pathway_jobs(arm, target)
        anchor = next(iter(jobs.values()))
        slug = target.replace("common_voice_en_", "cv_")
        item = {"id": slug, "dataset": "LibriTTS-R" if arm == "pathway_libritts" else "Seed-TTS-eval", "text": anchor["target_text"],
                "prompt": f"samples/{slug}/prompt.mp3", "heldout": f"samples/{slug}/heldout.mp3", "systems": [], "factorial": []}
        encode(data_path(anchor["refs"][0]["path"]), args.output / slug / "prompt.mp3")
        encode(data_path(anchor["heldout"]), args.output / slug / "heldout.mp3")
        for system in SYSTEMS:
            row = {"system": system}
            for condition in ("full", "xvector_only"):
                job = jobs.get((system, condition))
                wav = args.audio_dir / system / f"{job['job_id']}.wav" if job else None
                if wav and wav.exists():
                    encode(wav, args.output / slug / f"{system}_{condition}.mp3")
                    row[condition] = f"samples/{slug}/{system}_{condition}.mp3"
            item["systems"].append(row)
        for job in factorial:
            if job["target_id"] == target and job["seed"] == 42:
                wav = args.audio_dir / "cosyvoice1sites" / f"{job['job_id']}.wav"
                if wav.exists():
                    name = job["condition"].replace("site:", "")
                    encode(wav, args.output / slug / f"factorial_{name}.mp3")
                    item["factorial"].append({"cell": name, "label": SITE_LABELS[job["condition"]], "audio": f"samples/{slug}/factorial_{name}.mp3"})
        item["factorial"].sort(key=lambda cell: list(SITE_LABELS).index("site:" + cell["cell"]))
        items.append(item)
    (args.output.parent / "samples.json").write_text(json.dumps(items, indent=2))
    print(f"{len(items)} items written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
