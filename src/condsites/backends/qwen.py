from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from condsites.backends.base import (
    Condition,
    GenerationResult,
    PreparedReference,
)
from condsites.references import RefItem, concatenate_references


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QwenBackend:
    name = "qwen"

    def __init__(
        self,
        model_id: str,
        revision: str,
        language: str = "English",
        silence_ms: int = 300,
        **model_kwargs: Any,
    ):
        from qwen_tts import Qwen3TTSModel

        self.revision = revision
        self.language = language
        self.silence_ms = silence_ms
        model_kwargs.setdefault("device_map", "cuda:0")
        model_kwargs.setdefault("dtype", torch.bfloat16)
        self.model = Qwen3TTSModel.from_pretrained(
            model_id, revision=revision, **model_kwargs
        )

    @staticmethod
    def average_embeddings(embeddings: list[torch.Tensor]) -> torch.Tensor:
        if not embeddings:
            raise ValueError("At least one embedding is required")
        # Qwen's speaker encoder output is not unit-norm (norm ~10) and the
        # decoder is sensitive to its magnitude; L2-normalising it degrades
        # x-vector-only synthesis. Take the raw mean, as in combiners.py.
        return torch.stack(embeddings).mean(dim=0)

    def prepare_reference(
        self,
        refs: tuple[RefItem, ...],
        condition: Condition,
        work_dir: Path,
    ) -> PreparedReference:
        if condition in {Condition.SEQUENCE_ONLY, Condition.NEITHER}:
            raise NotImplementedError(
                f"Qwen does not safely expose the {condition.value} condition"
            )
        if not refs:
            raise ValueError("At least one reference is required")

        prompt_text = " ".join(item.text.strip() for item in refs)
        prompt_path: Path | None = None
        payload: dict[str, Any]
        metadata = {
            "backend_revision": self.revision,
            "prompt_duration": sum(item.duration for item in refs),
        }

        if condition is Condition.FULL:
            prompt_path = work_dir / "prompt.wav"
            prompt_text, prompt_duration = concatenate_references(
                refs,
                prompt_path,
                silence_ms=getattr(self, "silence_ms", 300),
                # 16 kHz reproduces the July confirmatory run; later runs pass 24 kHz.
                sample_rate=getattr(self, "prompt_sample_rate", 16000),
            )
            voice_clone_prompt = self.model.create_voice_clone_prompt(
                ref_audio=str(prompt_path),
                ref_text=prompt_text,
                x_vector_only_mode=False,
            )
            payload = {"voice_clone_prompt": voice_clone_prompt}
            metadata["prompt_duration"] = prompt_duration
        elif condition is Condition.XVECTOR_ONLY:
            prompt_items = []
            for item in refs:
                items = self.model.create_voice_clone_prompt(
                    ref_audio=item.path,
                    ref_text=item.text,
                    x_vector_only_mode=True,
                )
                if not items:
                    raise RuntimeError(
                        f"Qwen returned no prompt item for reference {item.id}"
                    )
                prompt_items.append(items[0])

            average = self.average_embeddings(
                [torch.as_tensor(item.ref_spk_embedding) for item in prompt_items]
            )
            prompt_type = type(prompt_items[0])
            combined_item = prompt_type(
                ref_code=None,
                ref_spk_embedding=average,
                x_vector_only_mode=True,
                icl_mode=False,
            )
            payload = {"voice_clone_prompt": [combined_item]}
        else:
            raise ValueError(f"Unknown condition: {condition}")

        return PreparedReference(
            condition=condition,
            refs=refs,
            prompt_path=prompt_path,
            prompt_text=prompt_text,
            payload=payload,
            metadata=metadata,
        )

    def synthesize(
        self,
        prepared: PreparedReference,
        target_text: str,
        seed: int,
    ) -> GenerationResult:
        if prepared.condition not in {Condition.FULL, Condition.XVECTOR_ONLY}:
            raise NotImplementedError(
                f"Qwen does not safely expose the {prepared.condition.value} condition"
            )
        seed_everything(seed)
        waveforms, sample_rate = self.model.generate_voice_clone(
            text=target_text,
            language=self.language,
            voice_clone_prompt=prepared.payload["voice_clone_prompt"],
        )
        if not waveforms:
            raise RuntimeError("Qwen produced no audio")
        waveform = np.asarray(waveforms[0], dtype=np.float32).reshape(-1)
        return GenerationResult(
            waveform=waveform,
            sample_rate=int(sample_rate),
            metadata={
                "backend_revision": self.revision,
                "condition": prepared.condition.value,
                "seed": seed,
            },
        )
