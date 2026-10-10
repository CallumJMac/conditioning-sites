"""F5-TTS backend (prompt-only control).

F5-TTS conditions solely on an in-context acoustic prompt (no separate speaker
embedding), so it supports only the full condition; multi-reference input is
concatenated into one prompt, as for the other systems' ICL pathway.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from condsites.backends.base import Condition, GenerationResult, PreparedReference
from condsites.references import RefItem, concatenate_references


class F5TTSBackend:
    name = "f5tts"

    def __init__(self, model: str = "F5TTS_v1_Base", silence_ms: int = 300, device: str | None = None):
        from f5_tts.api import F5TTS

        self.model = F5TTS(model=model, device=device)
        self.silence_ms = silence_ms

    def prepare_reference(self, refs: tuple[RefItem, ...], condition: Condition, work_dir: Path) -> PreparedReference:
        if condition is not Condition.FULL:
            raise ValueError("F5-TTS has no speaker-embedding pathway; only the full condition exists")
        prompt_path = work_dir / "prompt.wav"
        prompt_text, _ = concatenate_references(refs, prompt_path, silence_ms=self.silence_ms, sample_rate=24000)
        return PreparedReference(condition=condition, refs=refs, prompt_path=prompt_path, prompt_text=prompt_text, payload={}, metadata={})

    def synthesize(self, prepared: PreparedReference, target_text: str, seed: int) -> GenerationResult:
        waveform, sample_rate, _ = self.model.infer(
            ref_file=str(prepared.prompt_path), ref_text=prepared.prompt_text, gen_text=target_text, seed=seed
        )
        return GenerationResult(
            waveform=np.asarray(waveform, dtype=np.float32).reshape(-1),
            sample_rate=int(sample_rate),
            metadata={"condition": prepared.condition.value, "seed": seed},
        )
