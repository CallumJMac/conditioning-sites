from __future__ import annotations

import random
import subprocess
import sys
from hashlib import sha256
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

EXPECTED_MODEL_ARTIFACT_SHA256 = {
    "cosyvoice.yaml": (
        "5475b9b2f788f3ddd3798898041197e076d20cb2ea783ef9638e6eed36ba16db"
    ),
    "llm.pt": (
        "59f7eb172d2f33f6c8c02709d829c83960f97876fa9886c7ea232597b51af976"
    ),
    "flow.pt": (
        "fd80b089444a95e52956c57cdf177d7f6017a5af13b8a697717628a1d2be6b55"
    ),
    "hift.pt": (
        "91e679b6ca1eff71187ffb4f3ab0444935594cdcc20a9bd12afad111ef8d6012"
    ),
    "campplus.onnx": (
        "a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73"
    ),
    "speech_tokenizer_v1.onnx": (
        "23b5a723ed9143aebfd9ffda14ac4c21231f31c35ef837b6a13bb9e5488abb1e"
    ),
}
REQUIRED_MODEL_ARTIFACTS = tuple(EXPECTED_MODEL_ARTIFACT_SHA256)


def _artifact_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_integrity(model_dir: Path) -> None:
    missing = [
        name for name in EXPECTED_MODEL_ARTIFACT_SHA256
        if not (model_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "CosyVoice model artifacts missing: " + ", ".join(missing)
        )
    mismatched = [
        name for name, expected_hash in EXPECTED_MODEL_ARTIFACT_SHA256.items()
        if _artifact_sha256(model_dir / name) != expected_hash
    ]
    if mismatched:
        raise RuntimeError(
            "CosyVoice artifact integrity mismatch: " + ", ".join(mismatched)
        )


def validate_cosyvoice_runtime(
    source_dir: Path,
    model_dir: Path,
    source_revision: str,
    model_revision: str,
) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"CosyVoice source directory missing: {source_dir}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"CosyVoice model directory missing: {model_dir}")

    def git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(source_dir), *args],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"CosyVoice source revision validation failed: {exc.output.strip()}"
            ) from exc

    actual_revision = git("rev-parse", "HEAD")
    if actual_revision != source_revision:
        raise RuntimeError(
            f"CosyVoice source revision mismatch: expected {source_revision}, "
            f"found {actual_revision}"
        )
    if git("status", "--porcelain"):
        raise RuntimeError("CosyVoice source checkout is dirty")
    submodules = git("submodule", "status", "--recursive")
    invalid = [
        line for line in submodules.splitlines()
        if line and line[0] in {"-", "+", "U"}
    ]
    if invalid:
        raise RuntimeError("CosyVoice submodule state is invalid: " + "; ".join(invalid))

    marker = model_dir / ".model_revision"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != model_revision:
        raise RuntimeError(
            f"CosyVoice model revision marker mismatch: expected {model_revision}"
        )
    verify_artifact_integrity(model_dir)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CosyVoiceBackend:
    name = "cosyvoice"

    def __init__(
        self,
        model_dir: Path,
        source_dir: Path,
        source_revision: str,
        model_revision: str,
        silence_ms: int = 300,
        embedding_ablation_mode: str = "zero",
    ):
        validate_cosyvoice_runtime(
            source_dir,
            model_dir,
            source_revision,
            model_revision,
        )
        self.source_revision = source_revision
        self.model_revision = model_revision
        self.revision = (
            f"source:{source_revision};model:{model_revision}"
        )
        self.silence_ms = silence_ms
        self.embedding_ablation_mode = embedding_ablation_mode
        for path in (source_dir, source_dir / "third_party" / "Matcha-TTS"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from cosyvoice.cli.cosyvoice import CosyVoice

        self.model = CosyVoice(str(model_dir))

    @staticmethod
    def apply_condition(
        model_input: dict[str, Any],
        condition: Condition,
        embedding_ablation_mode: str = "zero",
    ) -> dict:
        output = dict(model_input)
        sequence_keys = {
            "text_len",
            "prompt_text",
            "prompt_text_len",
            "llm_prompt_speech_token",
            "llm_prompt_speech_token_len",
            "flow_prompt_speech_token",
            "flow_prompt_speech_token_len",
            "prompt_speech_feat",
            "prompt_speech_feat_len",
        }
        embedding_keys = {"llm_embedding", "flow_embedding"}
        if condition in {Condition.XVECTOR_ONLY, Condition.NEITHER}:
            for key in sequence_keys:
                output.pop(key, None)
        if condition in {Condition.SEQUENCE_ONLY, Condition.NEITHER}:
            if embedding_ablation_mode == "zero":
                for key in embedding_keys:
                    output[key] = torch.zeros((1, 192), dtype=torch.float32)
            elif embedding_ablation_mode == "native_empty_sensitivity":
                for key in embedding_keys:
                    output.pop(key, None)
            else:
                raise ValueError(
                    f"Unsupported embedding ablation mode: {embedding_ablation_mode}"
                )
        return output

    @staticmethod
    def average_embeddings(embeddings: list[torch.Tensor]) -> torch.Tensor:
        if not embeddings:
            raise ValueError("At least one embedding is required")
        normalized = [
            torch.nn.functional.normalize(item, dim=-1) for item in embeddings
        ]
        mean = torch.stack(normalized).mean(dim=0)
        return torch.nn.functional.normalize(mean, dim=-1)

    def prepare_reference(
        self,
        refs: tuple[RefItem, ...],
        condition: Condition,
        work_dir: Path,
    ) -> PreparedReference:
        if not refs:
            raise ValueError("At least one reference is required")

        prompt_text = " ".join(item.text.strip() for item in refs)
        payload: dict[str, Any] = {}
        prompt_path: Path | None = None
        metadata: dict[str, Any] = {
            "backend_revision": self.revision,
            "prompt_duration": sum(item.duration for item in refs),
        }

        if condition in {Condition.FULL, Condition.SEQUENCE_ONLY}:
            prompt_path = work_dir / "prompt.wav"
            prompt_text, prompt_duration = concatenate_references(
                refs,
                prompt_path,
                silence_ms=getattr(self, "silence_ms", 300),
                # 16 kHz reproduces the July confirmatory run; later runs pass 24 kHz.
                sample_rate=getattr(self, "prompt_sample_rate", 16000),
            )
            model_input = self.model.frontend.frontend_zero_shot(
                "", prompt_text, str(prompt_path), self.model.sample_rate, ""
            )
            payload["model_input"] = model_input
            metadata["prompt_duration"] = prompt_duration
        elif condition is Condition.XVECTOR_ONLY:
            embeddings = []
            for item in refs:
                embedding = self.model.frontend._extract_spk_embedding(
                    item.path
                )
                embeddings.append(torch.as_tensor(embedding))
            payload["embedding"] = self.average_embeddings(embeddings)
        elif condition is not Condition.NEITHER:
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
        seed_everything(seed)
        extracted = self.model.frontend._extract_text_token(target_text)
        if isinstance(extracted, tuple):
            text, text_len = extracted
            target_input = {"text": text, "text_len": text_len}
        else:
            target_input = {"text": extracted}

        if prepared.condition in {Condition.FULL, Condition.SEQUENCE_ONLY}:
            model_input = dict(prepared.payload["model_input"])
            model_input.update(target_input)
        elif prepared.condition is Condition.XVECTOR_ONLY:
            embedding = prepared.payload["embedding"]
            model_input = {
                **target_input,
                "llm_embedding": embedding,
                "flow_embedding": embedding,
            }
        elif prepared.condition is Condition.NEITHER:
            model_input = target_input
        else:
            raise ValueError(f"Unknown condition: {prepared.condition}")

        model_input = self.apply_condition(
            model_input,
            prepared.condition,
            getattr(self, "embedding_ablation_mode", "zero"),
        )
        chunks: list[torch.Tensor] = []
        with torch.inference_mode():
            for output in self.model.model.tts(**model_input):
                chunks.append(torch.as_tensor(output["tts_speech"]))
        if not chunks:
            raise RuntimeError("CosyVoice produced no audio chunks")

        waveform = (
            torch.cat(chunks, dim=-1).detach().cpu().to(torch.float32).reshape(-1).numpy()
        )
        return GenerationResult(
            waveform=waveform,
            sample_rate=int(self.model.sample_rate),
            metadata={
                "backend_revision": self.revision,
                "source_revision": getattr(self, "source_revision", ""),
                "model_revision": getattr(self, "model_revision", ""),
                "embedding_ablation_mode": getattr(
                    self, "embedding_ablation_mode", "zero"
                ),
                "condition": prepared.condition.value,
                "seed": seed,
            },
        )
