"""CosyVoice-300M embedding-site factorial backend (causal test of where identity enters).

CosyVoice-300M conditions both its language model (``llm_embedding``) and its
flow decoder (``flow_embedding``) on the CAM++ x-vector. This backend crosses
three switches within one model:

- prompt: prompt speech tokens/features present (as in the full condition) or absent;
- lm: the language-model embedding kept or zeroed;
- flow: the flow-decoder embedding kept or zeroed.

A site condition is written ``site:P{0,1}L{0,1}F{0,1}``. Zeroing follows the
July confirmatory "zero" ablation: both modules L2-normalise the embedding, so a
zero vector removes the speaker signal and leaves only the affine bias.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import torch

from condsites.backends.base import Condition, GenerationResult, PreparedReference
from condsites.backends.cosyvoice import CosyVoiceBackend, seed_everything
from condsites.references import RefItem

SITE_PATTERN = re.compile(r"site:P([01])L([01])F([01])\Z")
EMBEDDING_SHAPE = (1, 192)


def parse_site(condition: str) -> tuple[bool, bool, bool]:
    match = SITE_PATTERN.match(condition)
    if not match:
        raise ValueError(f"Not a site condition: {condition}")
    return tuple(flag == "1" for flag in match.groups())


class CosyVoiceSitesBackend(CosyVoiceBackend):
    name = "cosyvoice1sites"

    def prepare_site(self, refs: tuple[RefItem, ...], site: str, work_dir: Path) -> PreparedReference:
        prompt_on, _, _ = parse_site(site)
        prepared = self.prepare_reference(refs, Condition.FULL if prompt_on else Condition.XVECTOR_ONLY, work_dir)
        return replace(prepared, metadata={**prepared.metadata, "site": site})

    def synthesize(self, prepared: PreparedReference, target_text: str, seed: int) -> GenerationResult:
        prompt_on, lm_on, flow_on = parse_site(prepared.metadata["site"])
        seed_everything(seed)
        extracted = self.model.frontend._extract_text_token(target_text)
        target_input = {"text": extracted[0], "text_len": extracted[1]} if isinstance(extracted, tuple) else {"text": extracted}
        if prompt_on:
            model_input = {**prepared.payload["model_input"], **target_input}
        else:
            embedding = prepared.payload["embedding"]
            model_input = {**target_input, "llm_embedding": embedding, "flow_embedding": embedding}
        if not lm_on:
            model_input["llm_embedding"] = torch.zeros(EMBEDDING_SHAPE, dtype=torch.float32)
        if not flow_on:
            model_input["flow_embedding"] = torch.zeros(EMBEDDING_SHAPE, dtype=torch.float32)
        chunks = []
        with torch.inference_mode():
            for output in self.model.model.tts(**model_input):
                chunks.append(torch.as_tensor(output["tts_speech"]))
        if not chunks:
            raise RuntimeError("CosyVoice produced no audio chunks")
        waveform = torch.cat(chunks, dim=-1).detach().cpu().to(torch.float32).reshape(-1).numpy()
        return GenerationResult(waveform=waveform, sample_rate=int(self.model.sample_rate), metadata={"site": prepared.metadata["site"], "seed": seed})
