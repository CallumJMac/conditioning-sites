"""Fun-CosyVoice3-0.5B backend.

CosyVoice 2/3 differ from CosyVoice-300M in where the speaker embedding enters:
the Qwen-based language model ignores it, and only the flow-matching decoder is
conditioned on it. The pathway conditions are therefore:

- full: language model and flow both receive the prompt speech tokens (and the
  flow its prompt features); the flow also receives the CAM++ embedding;
- xvector_only: no prompt tokens or features anywhere; the flow receives only the
  (averaged) embedding, so the language model has no speaker information.

CosyVoice 3 expects a system-prompt prefix on the text it conditions on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from condsites.backends.base import Condition, GenerationResult, PreparedReference
from condsites.backends.cosyvoice import seed_everything
from condsites.references import RefItem, concatenate_references

SYSTEM_PROMPT = "You are a helpful assistant.<|endofprompt|>"
PROMPT_KEYS = (
    "prompt_text",
    "prompt_text_len",
    "llm_prompt_speech_token",
    "llm_prompt_speech_token_len",
    "flow_prompt_speech_token",
    "flow_prompt_speech_token_len",
    "prompt_speech_feat",
    "prompt_speech_feat_len",
)


class CosyVoice3Backend:
    name = "cosyvoice3"

    def __init__(self, model_dir: Path, source_dir: Path, silence_ms: int = 300):
        for path in (source_dir, source_dir / "third_party" / "Matcha-TTS"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from cosyvoice.cli.cosyvoice import CosyVoice3

        self.model = CosyVoice3(str(model_dir))
        self.silence_ms = silence_ms

    @staticmethod
    def average_embeddings(embeddings: list[torch.Tensor]) -> torch.Tensor:
        # The flow L2-normalises the embedding itself, so the mean's scale is irrelevant.
        return torch.stack(embeddings).mean(dim=0)

    def prepare_reference(self, refs: tuple[RefItem, ...], condition: Condition, work_dir: Path) -> PreparedReference:
        if not refs:
            raise ValueError("At least one reference is required")
        payload: dict[str, Any] = {}
        prompt_path: Path | None = None
        prompt_text = " ".join(item.text.strip() for item in refs)
        if condition is Condition.FULL:
            prompt_path = work_dir / "prompt.wav"
            prompt_text, _ = concatenate_references(refs, prompt_path, silence_ms=self.silence_ms, sample_rate=24000)
            payload["model_input"] = self.model.frontend.frontend_zero_shot(
                "", SYSTEM_PROMPT + prompt_text, str(prompt_path), self.model.sample_rate, ""
            )
        elif condition is Condition.XVECTOR_ONLY:
            payload["embedding"] = self.average_embeddings(
                [torch.as_tensor(self.model.frontend._extract_spk_embedding(item.path)) for item in refs]
            )
        else:
            raise ValueError(f"CosyVoice3 backend supports full and xvector_only, not {condition.value}")
        return PreparedReference(condition=condition, refs=refs, prompt_path=prompt_path, prompt_text=prompt_text, payload=payload, metadata={})

    def synthesize(self, prepared: PreparedReference, target_text: str, seed: int) -> GenerationResult:
        seed_everything(seed)
        if prepared.condition is Condition.FULL:
            text, text_len = self.model.frontend._extract_text_token(target_text)
            model_input = {**prepared.payload["model_input"], "text": text, "text_len": text_len}
        else:
            text, text_len = self.model.frontend._extract_text_token(SYSTEM_PROMPT + target_text)
            model_input = {"text": text, "text_len": text_len, "flow_embedding": prepared.payload["embedding"]}
            for key in PROMPT_KEYS:
                model_input.pop(key, None)
        chunks = []
        with torch.inference_mode():
            for output in self.model.model.tts(**model_input):
                chunks.append(torch.as_tensor(output["tts_speech"]))
        if not chunks:
            raise RuntimeError("CosyVoice3 produced no audio chunks")
        waveform = torch.cat(chunks, dim=-1).detach().cpu().to(torch.float32).reshape(-1).numpy()
        return GenerationResult(
            waveform=waveform,
            sample_rate=int(self.model.sample_rate),
            metadata={"condition": prepared.condition.value, "seed": seed},
        )
