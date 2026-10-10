from __future__ import annotations

import json
import random
import subprocess
import sys
from copy import copy
from hashlib import sha256
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

from condsites.backends.base import (
    Condition,
    GenerationResult,
    PreparedReference,
)
from condsites.references import RefItem

APPROVED_SOURCE_REVISION = "65b18437192794391a0308a8f705b1e33e633948"
APPROVED_MODEL_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
APPROVED_PERTH_REVISION = "ce86c49d029f42272c1902eccb675556b9ed2330"
APPROVED_RUNTIME_DEPENDENCIES = {
    "numpy": "1.26.4",
    "librosa": "0.11.0",
    "s3tokenizer": "0.3.0",
    "torch": "2.6.0",
    "torchaudio": "2.6.0",
    "transformers": "5.2.0",
    "diffusers": "0.29.0",
    "conformer": "0.3.2",
    "safetensors": "0.5.3",
    "spacy-pkuseg": "1.0.1",
    "pykakasi": "2.3.0",
    "gradio": "6.8.0",
    "pyloudnorm": "0.2.0",
    "omegaconf": "2.3.0",
    "resemble-perth": "1.0.1",
}
EXPECTED_RUNTIME_DEPENDENCY_PROVENANCE = {
    "schema_version": 1,
    "chatterbox_source_revision": APPROVED_SOURCE_REVISION,
    "chatterbox_source_commit_date": "2026-06-10T16:18:35Z",
    "pypi_metadata_base": "https://pypi.org/pypi",
    "packages": APPROVED_RUNTIME_DEPENDENCIES,
    "perth": {
        "repository": "https://github.com/resemble-ai/Perth.git",
        "revision": APPROVED_PERTH_REVISION,
        "metadata": (
            "https://raw.githubusercontent.com/resemble-ai/Perth/"
            f"{APPROVED_PERTH_REVISION}/pyproject.toml"
        ),
    },
}
EMPTY_T3_PROMPT = None
EXPECTED_LICENSE_SHA256 = (
    "4248e910a928849fe5815a0f9236e17fa07768d95b9193212752c464b93d6caa"
)
EXPECTED_MODEL_ARTIFACT_SHA256 = {
    "ve.safetensors": (
        "f0921cab452fa278bc25cd23ffd59d36f816d7dc5181dd1bef9751a7fb61f63c"
    ),
    "t3_cfg.safetensors": (
        "914cb1696f47527fe8852ca8f1fe1fa63cb34f76f9c715e84e067b744dd0da81"
    ),
    "s3gen.safetensors": (
        "2b78103c654207393955e4900aac14a12de8ef25f4b09424f1ef91941f161d4e"
    ),
    "tokenizer.json": (
        "d71e3a44eabb1784df9a68e9f95b251ecbf1a7af6a9f50835856b2ca9d8c14a5"
    ),
    "conds.pt": (
        "6552d70568833628ba019c6b03459e77fe71ca197d5c560cef9411bee9d87f4e"
    ),
}
REQUIRED_MODEL_ARTIFACTS = tuple(EXPECTED_MODEL_ARTIFACT_SHA256)


def verify_runtime_dependencies(provenance_path: Path) -> None:
    try:
        recorded = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Chatterbox runtime dependency provenance is missing or invalid"
        ) from exc
    if recorded != EXPECTED_RUNTIME_DEPENDENCY_PROVENANCE:
        raise RuntimeError("Chatterbox runtime dependency provenance mismatch")

    mismatched = []
    for name, expected in APPROVED_RUNTIME_DEPENDENCIES.items():
        try:
            actual = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            actual = "<missing>"
        if actual != expected:
            mismatched.append(f"{name}: expected {expected}, found {actual}")
    if mismatched:
        raise RuntimeError(
            "Chatterbox runtime dependency version mismatch: "
            + "; ".join(mismatched)
        )

    distribution = importlib_metadata.distribution("resemble-perth")
    direct_url_text = distribution.read_text("direct_url.json")
    try:
        direct_url = json.loads(direct_url_text or "")
        vcs_info = direct_url["vcs_info"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "Chatterbox Perth immutable install provenance is missing"
        ) from exc
    if (
        direct_url.get("url") != "https://github.com/resemble-ai/Perth.git"
        or vcs_info.get("vcs") != "git"
        or vcs_info.get("commit_id") != APPROVED_PERTH_REVISION
        or vcs_info.get("requested_revision") != APPROVED_PERTH_REVISION
    ):
        raise RuntimeError("Chatterbox Perth immutable install provenance mismatch")


def _artifact_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_integrity(model_dir: Path) -> None:
    missing = [
        name
        for name in EXPECTED_MODEL_ARTIFACT_SHA256
        if not (model_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Chatterbox model artifacts missing: " + ", ".join(missing)
        )
    mismatched = [
        name
        for name, expected in EXPECTED_MODEL_ARTIFACT_SHA256.items()
        if _artifact_sha256(model_dir / name) != expected
    ]
    if mismatched:
        raise RuntimeError(
            "Chatterbox artifact integrity mismatch: " + ", ".join(mismatched)
        )


def _git_output(source_dir: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_dir), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Chatterbox source revision validation failed: "
            f"{exc.output.strip()}"
        ) from exc


def validate_chatterbox_runtime(
    source_dir: Path,
    model_dir: Path,
    source_revision: str,
    model_revision: str,
) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Chatterbox source directory missing: {source_dir}"
        )
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Chatterbox model directory missing: {model_dir}")
    actual_revision = _git_output(source_dir, "rev-parse", "HEAD")
    if actual_revision != source_revision:
        raise RuntimeError(
            f"Chatterbox source revision mismatch: expected {source_revision}, "
            f"found {actual_revision}"
        )
    if _git_output(source_dir, "status", "--porcelain"):
        raise RuntimeError("Chatterbox source checkout is dirty")
    actual_model_revision = _git_output(model_dir, "rev-parse", "HEAD")
    if actual_model_revision != model_revision:
        raise RuntimeError(
            "Chatterbox model checkout revision mismatch: "
            f"expected {model_revision}, found {actual_model_revision}"
        )
    license_path = source_dir / "LICENSE"
    if (
        not license_path.is_file()
        or _artifact_sha256(license_path) != EXPECTED_LICENSE_SHA256
    ):
        raise RuntimeError("Chatterbox pinned source license integrity mismatch")

    expected_markers = {
        ".source_revision": source_revision,
        ".model_revision": model_revision,
    }
    for name, expected in expected_markers.items():
        marker = model_dir / name
        if (
            not marker.is_file()
            or marker.read_text(encoding="utf-8").strip() != expected
        ):
            label = name.removeprefix(".").replace("_", " ")
            raise RuntimeError(
                f"Chatterbox {label} marker mismatch: expected {expected}"
            )
    provenance = model_dir / ".license_provenance"
    if not provenance.is_file():
        raise RuntimeError("Chatterbox license provenance marker is missing")
    provenance_text = provenance.read_text(encoding="utf-8")
    required_provenance = (
        f"source_revision={source_revision}",
        f"model_revision={model_revision}",
        "license=MIT",
    )
    if any(item not in provenance_text for item in required_provenance):
        raise RuntimeError("Chatterbox license provenance marker mismatch")
    verify_artifact_integrity(model_dir)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _condition_intervention(condition: Condition) -> dict[str, str]:
    prompt_state = "retained" if condition is Condition.FULL else "none"
    metadata = {
        "condition": condition.value,
        "speaker_embedding": "retained",
        "t3_prompt_speech_tokens": prompt_state,
        "t3_prompt_speech_embedding": prompt_state,
        "s3gen_reference_conditioning": "retained_identically",
    }
    if condition is Condition.XVECTOR_ONLY:
        metadata["pinned_null_semantics"] = (
            "T3Cond.cond_prompt_speech_tokens=None and "
            "T3Cond.cond_prompt_speech_emb=None"
        )
    return metadata


class ChatterboxBackend:
    name = "chatterbox"

    def __init__(
        self,
        model_dir: Path,
        source_dir: Path,
        source_revision: str,
        model_revision: str,
        device: str | None = None,
    ):
        if (
            source_revision != APPROVED_SOURCE_REVISION
            or model_revision != APPROVED_MODEL_REVISION
        ):
            raise ValueError(
                "Chatterbox revisions must match the approved Chatterbox "
                f"source {APPROVED_SOURCE_REVISION} and model "
                f"{APPROVED_MODEL_REVISION}"
            )
        validate_chatterbox_runtime(
            source_dir, model_dir, source_revision, model_revision
        )
        verify_runtime_dependencies(model_dir / ".runtime_dependencies.json")
        self.source_revision = source_revision
        self.model_revision = model_revision
        self.revision = f"source:{source_revision};model:{model_revision}"
        source_package = source_dir / "src"
        if str(source_package) not in sys.path:
            sys.path.insert(0, str(source_package))

        from chatterbox.tts import ChatterboxTTS

        module_path = Path(sys.modules[ChatterboxTTS.__module__].__file__).resolve()
        if not module_path.is_relative_to(source_dir.resolve()):
            raise RuntimeError(
                "Loaded Chatterbox package does not come from pinned source checkout"
            )

        import torch

        runtime_device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = ChatterboxTTS.from_local(model_dir, runtime_device)

    @staticmethod
    def _validate_condition(condition: Condition) -> None:
        if condition not in {Condition.FULL, Condition.XVECTOR_ONLY}:
            raise NotImplementedError(
                f"Chatterbox does not support condition {condition.value}"
            )

    def prepare_reference(
        self,
        refs: tuple[RefItem, ...],
        condition: Condition,
        work_dir: Path,
    ) -> PreparedReference:
        del work_dir
        self._validate_condition(condition)
        if len(refs) != 1:
            raise ValueError("Chatterbox requires exactly one reference")
        ref = refs[0]
        self.model.prepare_conditionals(ref.path)
        full_conditionals = self.model.conds
        if full_conditionals is None:
            raise RuntimeError("Chatterbox did not prepare reference conditionals")

        conditionals = full_conditionals
        if condition is Condition.XVECTOR_ONLY:
            conditionals = copy(full_conditionals)
            conditionals.t3 = copy(full_conditionals.t3)
            conditionals.t3.cond_prompt_speech_tokens = EMPTY_T3_PROMPT
            conditionals.t3.cond_prompt_speech_emb = EMPTY_T3_PROMPT
            conditionals.gen = full_conditionals.gen

        return PreparedReference(
            condition=condition,
            refs=refs,
            prompt_path=Path(ref.path),
            prompt_text=ref.text,
            payload={"conditionals": conditionals},
            metadata={
                "backend_revision": self.revision,
                "source_revision": self.source_revision,
                "model_revision": self.model_revision,
                "prompt_duration": ref.duration,
                "intervention": _condition_intervention(condition),
            },
        )

    def synthesize(
        self,
        prepared: PreparedReference,
        target_text: str,
        seed: int,
    ) -> GenerationResult:
        self._validate_condition(prepared.condition)
        seed_everything(seed)
        conditionals = prepared.payload["conditionals"]
        self.model.conds = conditionals
        generated: Any = self.model.generate(target_text)
        if hasattr(generated, "detach"):
            generated = generated.detach().cpu().numpy()
        waveform = np.asarray(generated, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            raise RuntimeError("Chatterbox produced no audio")
        return GenerationResult(
            waveform=waveform,
            sample_rate=int(self.model.sr),
            metadata={
                "backend_revision": self.revision,
                "source_revision": self.source_revision,
                "model_revision": self.model_revision,
                "condition": prepared.condition.value,
                "seed": seed,
                "intervention": _condition_intervention(prepared.condition),
            },
        )
