"""Score synthesised clips with the late-2026 standard metric suite.

Metrics (one row per clip, resumable):
- sim_prompt / sim_heldout: cosine similarity of WavLM-large speaker-verification
  embeddings (Seed-TTS-eval checkpoint, UniSpeech ECAPA-TDNN head) against the
  cloning prompt and against a held-out recording of the speaker (SIM-o style);
- wer: Whisper-large-v3 (official weights: Hugging Face pipeline in fp16 on CUDA,
  or the MLX port on Apple Silicon) with the Seed-TTS-eval English normalisation
  (strip punctuation except apostrophes, lowercase);
- utmos22 (SpeechMOS utmos22_strong) and utmosv2 (sarulab UTMOSv2 fusion model).
  UTMOSv2 scores random 4 s crops, so it is seeded per clip and averaged over
  5 crops to make scores deterministic and less noisy.

Input manifest CSV columns: clip_id, wav, text, prompt_wav, heldout_wav (may be empty).

Usage:
    python -m condsites.scoring --manifest m.csv --output scores.csv \
        --unispeech-dir .external/eval/seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification \
        --sv-checkpoint .external/eval/wavlm_large_finetune.pth
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import string
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from jiwer import wer
from torch.nn import functional

WHISPER_REPO = "mlx-community/whisper-large-v3-mlx"
PUNCTUATION = "".join(character for character in string.punctuation if character != "'")


def seed_normalise(text: str) -> str:
    for character in PUNCTUATION:
        text = text.replace(character, "")
    return text.replace("  ", " ").strip().lower()


class Scorer:
    def __init__(self, unispeech_dir: Path, sv_checkpoint: Path, device: str = "cpu", skip_utmosv2: bool = False):
        sys.path.insert(0, str(unispeech_dir))
        from verification import init_model  # UniSpeech speaker-verification code

        self.device = device
        self.sv_model = init_model("wavlm_large", str(sv_checkpoint)).eval().to(device)
        self.utmos22 = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval().to(device)
        self.utmosv2 = None
        if not skip_utmosv2:
            import utmosv2

            self.utmosv2 = utmosv2.create_model(pretrained=True, device=device)
        if device.startswith("cuda"):
            from transformers import pipeline

            # Official openai/whisper-large-v3 weights in fp16 (~3 GB) so scoring fits beside synthesis.
            self.whisper = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16, device=device)
        else:
            self.whisper = None
        self.embedding_cache: dict[str, torch.Tensor] = {}

    def transcribe(self, path: str) -> str:
        if self.whisper is not None:
            audio, _ = librosa.load(path, sr=16000, mono=True)
            # Runaway generations can exceed Whisper's 30 s window; long-form decoding needs timestamps.
            long_form = len(audio) > 30 * 16000
            return self.whisper(
                {"raw": audio, "sampling_rate": 16000},
                return_timestamps=long_form,
                generate_kwargs={"language": "en", "task": "transcribe"},
            )["text"].strip()
        import mlx_whisper

        return mlx_whisper.transcribe(path, path_or_hf_repo=WHISPER_REPO, language="en")["text"].strip()

    def embedding(self, path: str) -> torch.Tensor:
        if path not in self.embedding_cache:
            audio, _ = librosa.load(path, sr=16000, mono=True)
            with torch.no_grad():
                self.embedding_cache[path] = self.sv_model(torch.from_numpy(audio).float().unsqueeze(0).to(self.device)).cpu()
        return self.embedding_cache[path]

    def similarity(self, first: str, second: str) -> float:
        return round(float(functional.cosine_similarity(self.embedding(first), self.embedding(second))[0]), 4)

    def score(self, row: dict) -> dict:
        audio_16k, _ = librosa.load(row["wav"], sr=16000, mono=True)
        with torch.no_grad():
            utmos22 = float(self.utmos22(torch.from_numpy(audio_16k).unsqueeze(0).to(self.device), 16000)[0])
        utmosv2 = float("nan")
        clip_seed = int(hashlib.sha256(row["clip_id"].encode()).hexdigest()[:8], 16)
        torch.manual_seed(clip_seed)
        np.random.seed(clip_seed)
        random.seed(clip_seed)
        if self.utmosv2 is not None:
            utmosv2 = float(np.asarray(self.utmosv2.predict(
                data=audio_16k.astype(np.float32), sr=16000, device=self.device, num_workers=0, num_repetitions=5, verbose=False
            )).reshape(-1)[0])
        transcript = self.transcribe(row["wav"])
        reference = seed_normalise(row["text"])
        result = {
            "clip_id": row["clip_id"],
            "sim_prompt": self.similarity(row["wav"], row["prompt_wav"]) if row.get("prompt_wav") else "",
            "sim_heldout": self.similarity(row["wav"], row["heldout_wav"]) if row.get("heldout_wav") else "",
            "wer": round(float(wer(reference, seed_normalise(transcript))), 4) if reference else "",
            "utmos22": round(utmos22, 4),
            "utmosv2": round(utmosv2, 4),
            "transcript": transcript,
        }
        self.embedding_cache.pop(row["wav"], None)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--unispeech-dir", required=True, type=Path)
    parser.add_argument("--sv-checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-utmosv2", action="store_true", help="score UTMOSv2 separately in batched mode (much faster)")
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    done: set[str] = set()
    if args.output.exists():
        with args.output.open(encoding="utf-8", newline="") as handle:
            done = {row["clip_id"] for row in csv.DictReader(handle)}
    scorer = Scorer(args.unispeech_dir, args.sv_checkpoint, args.device, args.skip_utmosv2)
    fields = ["clip_id", "sim_prompt", "sim_heldout", "wer", "utmos22", "utmosv2", "transcript"]
    write_header = not args.output.exists()
    with args.output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        if write_header:
            writer.writeheader()
        for index, row in enumerate(rows):
            if row["clip_id"] in done:
                continue
            try:
                writer.writerow(scorer.score(row))
            except Exception as error:  # one bad clip must not stop the run; failures are reported, not hidden
                writer.writerow({"clip_id": row["clip_id"], "transcript": f"SCORING_ERROR {type(error).__name__}: {error}"[:300]})
            handle.flush()
            if (index + 1) % 100 == 0:
                print(f"{index + 1}/{len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
