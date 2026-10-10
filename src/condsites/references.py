"""Reference utterances and prompt construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass
class RefItem:
    """One reference recording with its transcript; ``duration`` is in seconds."""

    id: str
    speaker_id: str
    text: str
    path: str
    sample_rate: int
    duration: float
    sex: str | None = None


def load_manifest(manifest_path: str | Path) -> list[RefItem]:
    """Read a LibriTTS-R manifest (``data/libritts_r_aligned/manifest_40spk.json``).

    Relative audio paths resolve against the manifest's directory. A missing or
    unreadable audio file raises, because a silent zero duration would corrupt
    the duration-based reference selection.
    """
    manifest_path = Path(manifest_path)
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = []
    for entry in entries:
        audio_path = Path(entry["path"])
        if not audio_path.is_absolute():
            audio_path = (manifest_path.parent / audio_path).resolve()
        sex = str(entry["sex"]).strip() if entry.get("sex") not in (None, "") else None
        items.append(RefItem(
            id=entry["id"],
            speaker_id=entry["speaker_id"],
            text=entry["text_normalized"],
            path=str(audio_path),
            sample_rate=entry.get("sample_rate", 24000),
            duration=sf.info(audio_path).duration,
            sex=sex,
        ))
    return items


def concatenate_references(
    refs: tuple[RefItem, ...],
    output: Path,
    silence_ms: int = 300,
    sample_rate: int = 16000,
) -> tuple[str, float]:
    """Join reference recordings with ``silence_ms`` gaps into one prompt WAV.

    Returns the joined transcript and the prompt duration in seconds. The
    16 kHz default reproduces the July 2026 confirmatory run; every experiment
    in the paper passes ``sample_rate=24000``.
    """
    silence = np.zeros(round(sample_rate * silence_ms / 1000), dtype=np.float32)
    segments: list[np.ndarray] = []
    for item in refs:
        audio, _ = librosa.load(item.path, sr=sample_rate, mono=True)
        if segments:
            segments.append(silence)
        segments.append(audio.astype(np.float32))
    waveform = np.concatenate(segments)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, waveform, sample_rate)
    return " ".join(item.text.strip() for item in refs), len(waveform) / sample_rate
