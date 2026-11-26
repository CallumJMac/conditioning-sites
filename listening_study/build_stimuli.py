"""
Build stimuli and config for the paired-comparison (CMOS / CSMOS) listening study.

Design (see DEPLOY.md):
- Contrast: each system's own output with the acoustic prompt ("full") vs with the
  prompt removed and the speaker embedding kept ("xvector_only"), from pathway_v2.
- Systems: Qwen3-TTS 1.7B (embedding conditions the language model; identity is
  predicted to survive) and CosyVoice 3 (embedding in the decoder only; identity
  is predicted to collapse).
- 60 pairs = 2 systems x 2 datasets (LibriTTS-R, Seed-TTS-eval) x 15 items, drawn
  at random with a fixed seed (no metric-based selection); LibriTTS-R items use
  distinct speakers and seed 42.
- Speaker reference for CSMOS is the held-out recording used by SIM-o (a >9.5 s
  LibriTTS-R clip never used as prompt/target, or the Seed-TTS-eval ground truth).
- Counterbalanced halves: list A rates pairs 0-29 for naturalness and 30-59 for
  similarity; list B the reverse. Each listener hears each pair once.
- Traps per block: an identical pair (expected 0) and a natural vs heavily
  degraded pair (expected a clear preference for natural).
- Huggins-pitch headphone screen (Milne et al., 2021): 12 pre-generated 3AFC
  trials, 1 s white noise per interval, +-6% band around 600 Hz phase-inverted
  in the right channel.
- All speech is loudness-normalised to -23 LUFS (ffmpeg loudnorm, BS.1770).

Usage:
    python build_stimuli.py --jobs <pathway_v2 jobs.jsonl> --audio-dir <dir with <system>/<job_id>.wav> \
        --bundle-root <dir that resolves job data/ paths> --seedtts-root <seedtts_testset/en> --output-dir .

Requirements: numpy, soundfile, ffmpeg on PATH.
"""

import argparse
import csv
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SEED = 20261009
SYSTEMS = ("qwen17", "cosyvoice3")
ARMS = ("pathway_libritts", "pathway_seedtts")
ITEMS_PER_CELL = 15
SAMPLE_RATE = 24000
HUGGINS_RATE = 44100


def _data_path(bundle_root: Path, seedtts_root: Path, relative: str) -> Path:
    # Job paths are relative to the worker bundle: data/libritts_r_aligned/... or data/seedtts_en/...
    if relative.startswith("data/seedtts_en/"):
        return seedtts_root / relative.removeprefix("data/seedtts_en/")
    return bundle_root / relative


def loudnorm(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
         "-af", "loudnorm=I=-23:TP=-2:LRA=11", "-ar", str(SAMPLE_RATE), "-ac", "1", str(destination)],
        check=True,
    )


def degrade(source: Path, destination: Path) -> None:
    """Clearly degraded trap stimulus: 8 kHz telephone band, 4-bit-like noise, low SNR."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
         "-af", "highpass=f=300,lowpass=f=1500,acrusher=bits=4:mode=lin,loudnorm=I=-23:TP=-2:LRA=11",
         "-ar", str(SAMPLE_RATE), "-ac", "1", str(destination)],
        check=True,
    )


def huggins_trial(rng: np.random.Generator, target_interval: int) -> np.ndarray:
    """Three 1 s white-noise intervals; the target interval carries Huggins pitch."""
    n_samples = HUGGINS_RATE
    gap = np.zeros((int(0.5 * HUGGINS_RATE), 2))
    ramp = int(0.02 * HUGGINS_RATE)
    window = np.ones(n_samples)
    window[:ramp] = np.linspace(0, 1, ramp)
    window[-ramp:] = np.linspace(1, 0, ramp)
    intervals = []
    for index in range(3):
        noise = rng.standard_normal(n_samples)
        left = noise
        right = noise.copy()
        if index == target_interval:
            spectrum = np.fft.rfft(noise)
            frequencies = np.fft.rfftfreq(n_samples, 1 / HUGGINS_RATE)
            band = (frequencies >= 600 * 0.94) & (frequencies <= 600 * 1.06)
            spectrum[band] *= -1
            right = np.fft.irfft(spectrum, n_samples)
        stereo = np.stack([left, right], axis=1) * window[:, None]
        intervals.extend([stereo, gap])
    signal = np.concatenate(intervals[:-1])
    return (signal / np.abs(signal).max() * 0.3).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--seedtts-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rng = random.Random(SEED)
    audio_dir = args.output_dir / "audio"
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    data = lambda relative: _data_path(args.bundle_root, args.seedtts_root, relative)

    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines()]
    cells: dict[tuple[str, str, str, int], dict[str, dict]] = {}
    for job in jobs:
        if job["system"] in SYSTEMS and job["arm"] in ARMS and job["seed"] == 42:
            wav = args.audio_dir / job["system"] / f"{job['job_id']}.wav"
            if wav.exists():
                cells.setdefault((job["system"], job["arm"], job["target_id"], job["seed"]), {})[job["condition"]] = {**job, "wav": wav}
    complete = {key: value for key, value in cells.items() if {"full", "xvector_only"} <= value.keys()}

    pairs, used = [], set()
    for system in SYSTEMS:
        for arm in ARMS:
            candidates = sorted(key for key in complete if key[0] == system and key[1] == arm)
            rng.shuffle(candidates)
            chosen, speakers = [], set()
            for key in candidates:
                speaker = complete[key]["full"]["speaker_id"]
                if arm == "pathway_libritts" and speaker in speakers:
                    continue
                chosen.append(key); speakers.add(speaker); used.add(key[2])
                if len(chosen) == ITEMS_PER_CELL:
                    break
            for key in chosen:
                full, xvec = complete[key]["full"], complete[key]["xvector_only"]
                pair_id = f"p{len(pairs):02d}"
                pair_dir = audio_dir / "pairs" / pair_id
                loudnorm(full["wav"], pair_dir / "icl.wav")
                loudnorm(xvec["wav"], pair_dir / "xvec.wav")
                loudnorm(data(full["heldout"]), pair_dir / "reference.wav")
                pairs.append({
                    "id": pair_id, "system": system, "arm": arm, "speaker_id": full["speaker_id"],
                    "target_id": full["target_id"], "text": full["target_text"],
                    "icl": f"audio/pairs/{pair_id}/icl.wav", "xvec": f"audio/pairs/{pair_id}/xvec.wav",
                    "reference": f"audio/pairs/{pair_id}/reference.wav",
                    "icl_job_id": full["job_id"], "xvec_job_id": xvec["job_id"],
                })
    # Stratified eighths: every eighth item within each system x dataset cell, so
    # every list rates every cell on both axes and each listener hears each pair once
    # (about 20 trials, 12 minutes); order within lists is randomised in the browser.
    eighths = [[] for _ in range(8)]
    for cell in sorted({(pair["system"], pair["arm"]) for pair in pairs}):
        members = [pair["id"] for pair in pairs if (pair["system"], pair["arm"]) == cell]
        for index in range(8):
            eighths[index] += members[index::8]

    # Traps and practice use LibriTTS-R items not in the main set (Qwen3-TTS 1.7B).
    spare = [key for key in sorted(complete) if key[0] == "qwen17" and key[1] == "pathway_libritts" and key[2] not in used][:3]
    traps = []
    for index, key in enumerate(spare[:2]):
        full, xvec = complete[key]["full"], complete[key]["xvector_only"]
        trap_dir = audio_dir / "traps" / f"t{index}"
        natural = data(full["heldout"])
        loudnorm(natural, trap_dir / "natural.wav")
        degrade(natural, trap_dir / "degraded.wav")
        loudnorm(xvec["wav"], trap_dir / "same.wav")
        loudnorm(data(full["refs"][0]["path"]), trap_dir / "reference.wav")
        traps.append({
            "id": f"t{index}",
            "natural": f"audio/traps/t{index}/natural.wav",
            "degraded": f"audio/traps/t{index}/degraded.wav",
            "same": f"audio/traps/t{index}/same.wav",
            "reference": f"audio/traps/t{index}/reference.wav",
        })
    practice = complete[spare[2]]
    practice_dir = audio_dir / "practice"
    loudnorm(practice["full"]["wav"], practice_dir / "a.wav")
    loudnorm(practice["xvector_only"]["wav"], practice_dir / "b.wav")
    loudnorm(data(practice["full"]["heldout"]), practice_dir / "reference.wav")

    noise_rng = np.random.default_rng(SEED)
    huggins = []
    for index in range(12):
        target = int(noise_rng.integers(3))
        path = audio_dir / "headphone" / f"hp{index:02d}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, huggins_trial(noise_rng, target), HUGGINS_RATE)
        huggins.append({"file": f"audio/headphone/hp{index:02d}.wav", "answer": target + 1})

    config = {
        "testId": "pathway_v2_cmos_v1",
        "resultsEndpoint": "",
        "prolificCompletionUrl": "https://app.prolific.com/submissions/complete?cc=REPLACE_ME",
        "headphone": {"trials": huggins, "nTrials": 6, "passThreshold": 5},
        "practice": {"a": "audio/practice/a.wav", "b": "audio/practice/b.wav", "reference": "audio/practice/reference.wav"},
        # List k rates eighth k for naturalness and eighth k+4 for speaker match.
        "lists": {f"L{index + 1}": {"naturalness": eighths[index], "similarity": eighths[(index + 4) % 8]} for index in range(8)},
        "pairs": {pair["id"]: pair for pair in pairs},
        "traps": traps,
    }
    (args.output_dir / "configs").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "configs" / "study.json").write_text(json.dumps(config, indent=2))
    print(f"{len(pairs)} pairs, {len(traps)} traps, {len(huggins)} headphone trials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
