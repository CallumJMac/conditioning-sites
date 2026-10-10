from __future__ import annotations

import json
import random
import subprocess
import sys
from hashlib import sha256
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from condsites.backends.base import (
    Condition,
    GenerationResult,
    PreparedReference,
)
from condsites.references import RefItem

APPROVED_SOURCE_REVISION = "dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e"
APPROVED_MODEL_REVISION = "6c2b0d75eae4b7047358e3b6bd9325f857d43f77"
APPROVED_SOURCE_REPOSITORY = "https://github.com/coqui-ai/TTS.git"
APPROVED_MODEL_REPOSITORY = "https://huggingface.co/coqui/XTTS-v2"
APPROVED_LICENSE_NAME = "Coqui Public Model License 1.0.0"
APPROVED_LICENSE_ID = APPROVED_LICENSE_NAME
APPROVED_USAGE_RESTRICTION = "research-only"
TARGET_LANGUAGE = "en"
SAMPLE_RATE = 24000
APPROVED_RUNTIME_DEPENDENCIES = {
    "numpy": "1.22.0",
    "Cython": "0.29.30",
    "scipy": "1.11.2",
    "torch": "2.1.0",
    "torchaudio": "2.1.0",
    "soundfile": "0.12.1",
    "librosa": "0.10.0",
    "scikit-learn": "1.3.0",
    "numba": "0.57.0",
    "inflect": "5.6.0",
    "tqdm": "4.64.1",
    "anyascii": "0.3.0",
    "PyYAML": "6.0",
    "fsspec": "2023.6.0",
    "aiohttp": "3.8.1",
    "packaging": "23.1",
    "mutagen": "1.47.0",
    "Flask": "2.0.1",
    "pysbd": "0.3.4",
    "umap-learn": "0.5.1",
    "pandas": "1.4.0",
    "matplotlib": "3.7.0",
    "trainer": "0.0.36",
    "coqpit": "0.0.16",
    "jieba": "0.42.1",
    "pypinyin": "0.50.0",
    "hangul-romanize": "0.1.0",
    "gruut": "2.2.3",
    "jamo": "0.4.1",
    "nltk": "3.8.1",
    "g2pkk": "0.1.2",
    "bangla": "0.0.2",
    "bnnumerizer": "0.0.2",
    "bnunicodenormalizer": "0.1.6",
    "einops": "0.6.0",
    "transformers": "4.33.0",
    "encodec": "0.1.1",
    "Unidecode": "1.3.2",
    "num2words": "0.5.13",
    "spacy": "3.7.2",
}
EXPECTED_RUNTIME_DEPENDENCY_PROVENANCE = {
    "schema_version": 1,
    "xtts_source_revision": APPROVED_SOURCE_REVISION,
    "xtts_source_commit_date": "2024-02-10T14:20:58Z",
    "pypi_metadata_base": "https://pypi.org/pypi",
    "packages": APPROVED_RUNTIME_DEPENDENCIES,
}
EXPECTED_LICENSE_SHA256 = (
    "190f6d7c19b8984f91b97712b94ce92d2b2e640fc677dacab966e955ece9d043"
)
EXPECTED_MODEL_ARTIFACT_SHA256 = {
    "LICENSE.txt": EXPECTED_LICENSE_SHA256,
    "config.json": (
        "ef262b1454dd2a77e1461b0b2cd53e19b8a7624cc131b837d36df67356bc75e8"
    ),
    "model.pth": (
        "c7ea20001c6a0a841c77e252d8409f6a74fb423e79b3206a0771ba5989776187"
    ),
    "speakers_xtts.pth": (
        "f0f6137c19a4eab0cbbe4c99b5babacf68b1746e50da90807708c10e645b943b"
    ),
    "vocab.json": (
        "928260878a59da8a72a2a5b7687fea29d5106137669d90945430fe17e415304a"
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
        name
        for name in EXPECTED_MODEL_ARTIFACT_SHA256
        if not (model_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError("XTTS model artifacts missing: " + ", ".join(missing))
    mismatched = [
        name
        for name, expected in EXPECTED_MODEL_ARTIFACT_SHA256.items()
        if _artifact_sha256(model_dir / name) != expected
    ]
    if mismatched:
        raise RuntimeError("XTTS artifact integrity mismatch: " + ", ".join(mismatched))


def verify_runtime_dependencies(provenance_path: Path) -> None:
    try:
        recorded = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "XTTS runtime dependency provenance is missing or invalid"
        ) from exc
    if recorded != EXPECTED_RUNTIME_DEPENDENCY_PROVENANCE:
        raise RuntimeError("XTTS runtime dependency provenance mismatch")
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
            "XTTS runtime dependency version mismatch: " + "; ".join(mismatched)
        )


def _git_output(directory: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(directory), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"XTTS revision validation failed: {exc.output.strip()}"
        ) from exc


def validate_xtts_runtime(
    source_dir: Path,
    model_dir: Path,
    source_revision: str,
    model_revision: str,
) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"XTTS source directory missing: {source_dir}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"XTTS model directory missing: {model_dir}")
    actual_source_revision = _git_output(source_dir, "rev-parse", "HEAD")
    if actual_source_revision != source_revision:
        raise RuntimeError(
            f"XTTS source revision mismatch: expected {source_revision}, "
            f"found {actual_source_revision}"
        )
    if _git_output(source_dir, "status", "--porcelain"):
        raise RuntimeError("XTTS source checkout is dirty")
    actual_model_revision = _git_output(model_dir, "rev-parse", "HEAD")
    if actual_model_revision != model_revision:
        raise RuntimeError(
            f"XTTS model checkout revision mismatch: expected {model_revision}, "
            f"found {actual_model_revision}"
        )
    for name, expected in {
        ".source_revision": source_revision,
        ".model_revision": model_revision,
    }.items():
        marker = model_dir / name
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != expected:
            label = name.removeprefix(".").replace("_", " ")
            raise RuntimeError(f"XTTS {label} marker mismatch: expected {expected}")
    provenance = model_dir / ".license_provenance"
    if not provenance.is_file():
        raise RuntimeError("XTTS license provenance marker is missing")
    provenance_lines = provenance.read_text(encoding="utf-8").splitlines()
    try:
        recorded_provenance = dict(line.split("=", 1) for line in provenance_lines)
    except ValueError as exc:
        raise RuntimeError("XTTS license provenance marker mismatch") from exc
    expected_provenance = {
        "source_repository": APPROVED_SOURCE_REPOSITORY,
        "source_revision": source_revision,
        "model_repository": APPROVED_MODEL_REPOSITORY,
        "model_revision": model_revision,
        "license": APPROVED_LICENSE_NAME,
        "license_sha256": EXPECTED_LICENSE_SHA256,
    }
    if (
        len(recorded_provenance) != len(provenance_lines)
        or recorded_provenance != expected_provenance
    ):
        raise RuntimeError("XTTS license provenance marker mismatch")
    verify_artifact_integrity(model_dir)


def verify_import_provenance(source_dir: Path, *modules: ModuleType) -> None:
    source_root = source_dir.resolve()
    for module in modules:
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(source_root):
            raise RuntimeError(
                "Loaded XTTS package does not come from pinned source checkout"
            )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_heldout_mean_gpt_latent(
    model: Any,
    controls: tuple[RefItem, ...],
) -> Any:
    if len(controls) != 2:
        raise ValueError("XTTS held-out mean requires exactly two controls")
    if any(not str(control.path).strip() for control in controls):
        raise ValueError("XTTS control audio paths must be non-empty")
    latents = [
        model.get_conditioning_latents(audio_path=control.path)[0]
        for control in controls
    ]
    first, second = latents
    if (
        first.shape != second.shape
        or first.dtype != second.dtype
        or first.device != second.device
    ):
        raise ValueError(
            "XTTS control GPT latents must have identical shape, dtype, and device"
        )
    mean = (first + second) / 2
    if (
        mean.shape != first.shape
        or mean.dtype != first.dtype
        or mean.device != first.device
    ):
        raise ValueError(
            "XTTS held-out mean changed GPT latent shape, dtype, or device"
        )
    return mean.clone()


def _condition_intervention(
    condition: Condition,
    embedding_ablation_mode: str = "zero_gpt_conditioning_latents",
    control_reference_ids: tuple[str, ...] = (),
    control_speaker_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    intervention: dict[str, Any] = {
        "condition": condition.value,
        "gpt_conditioning_latents": (
            "retained"
            if condition is Condition.FULL
            else (
                "fixed_arithmetic_mean_of_heldout_female_male_controls"
                if embedding_ablation_mode == "gpt_latent_heldout_mean"
                else "zeroed_with_torch.zeros_like"
            )
        ),
        "speaker_embedding": "retained_identically",
    }
    if embedding_ablation_mode == "gpt_latent_heldout_mean":
        intervention["gpt_control_reference_ids"] = list(control_reference_ids)
        intervention["gpt_control_speaker_ids"] = list(control_speaker_ids)
    return intervention


class XTTSBackend:
    name = "xtts"

    def __init__(
        self,
        model_dir: Path,
        source_dir: Path,
        source_revision: str,
        model_revision: str,
        license_id: str,
        usage_restriction: str,
        device: str | None = None,
        embedding_ablation_mode: str = "zero_gpt_conditioning_latents",
        gpt_control_references: tuple[RefItem, ...] = (),
    ):
        if embedding_ablation_mode not in {
            "zero_gpt_conditioning_latents",
            "gpt_latent_heldout_mean",
        }:
            raise ValueError("Unsupported XTTS embedding ablation mode")
        if (
            embedding_ablation_mode == "gpt_latent_heldout_mean"
            and len(gpt_control_references) != 2
        ):
            raise ValueError(
                "XTTS held-out mean requires exactly two control references"
            )
        if (
            embedding_ablation_mode == "zero_gpt_conditioning_latents"
            and gpt_control_references
        ):
            raise ValueError("Zero GPT latent mode does not accept control references")
        if (
            source_revision != APPROVED_SOURCE_REVISION
            or model_revision != APPROVED_MODEL_REVISION
        ):
            raise ValueError(
                "XTTS revisions must match the approved XTTS source "
                f"{APPROVED_SOURCE_REVISION} and model {APPROVED_MODEL_REVISION}"
            )
        if (
            license_id != APPROVED_LICENSE_ID
            or usage_restriction != APPROVED_USAGE_RESTRICTION
        ):
            raise ValueError(
                "XTTS usage must be research-only under the approved "
                "Coqui Public Model License 1.0.0"
            )
        validate_xtts_runtime(
            source_dir, model_dir, source_revision, model_revision
        )
        verify_runtime_dependencies(model_dir / ".runtime_dependencies.json")
        self.source_revision = source_revision
        self.model_revision = model_revision
        self.license_id = license_id
        self.usage_restriction = usage_restriction
        self.embedding_ablation_mode = embedding_ablation_mode
        self.gpt_control_reference_ids = tuple(
            item.id for item in gpt_control_references
        )
        self.gpt_control_speaker_ids = tuple(
            item.speaker_id for item in gpt_control_references
        )
        self.revision = f"source:{source_revision};model:{model_revision}"
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))

        from TTS.tts.configs import xtts_config as xtts_config_module
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models import xtts as xtts_model_module
        from TTS.tts.models.xtts import Xtts

        verify_import_provenance(
            source_dir, xtts_config_module, xtts_model_module
        )
        config = XttsConfig()
        config.load_json(str(model_dir / "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config, checkpoint_dir=str(model_dir), eval=True
        )
        import torch

        runtime_device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(runtime_device)
        self.sample_rate = int(config.audio.output_sample_rate)
        self._heldout_mean_gpt_cond_latent = (
            _compute_heldout_mean_gpt_latent(self.model, gpt_control_references)
            if embedding_ablation_mode == "gpt_latent_heldout_mean"
            else None
        )

    @staticmethod
    def _validate_condition(condition: Condition) -> None:
        if condition not in {Condition.FULL, Condition.XVECTOR_ONLY}:
            raise NotImplementedError(
                f"XTTS does not support condition {condition.value}"
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
            raise ValueError("XTTS requires exactly one reference")
        ref = refs[0]
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=ref.path
        )
        if condition is Condition.XVECTOR_ONLY:
            if self.embedding_ablation_mode == "gpt_latent_heldout_mean":
                control = self._heldout_mean_gpt_cond_latent
                if control is None:
                    raise RuntimeError("XTTS held-out mean control is unavailable")
                if (
                    control.shape != gpt_cond_latent.shape
                    or control.dtype != gpt_cond_latent.dtype
                    or control.device != gpt_cond_latent.device
                ):
                    raise ValueError(
                        "XTTS held-out mean and target GPT latent must have "
                        "identical shape, dtype, and device"
                    )
                gpt_cond_latent = control.clone()
            else:
                import torch

                gpt_cond_latent = torch.zeros_like(gpt_cond_latent)
        return PreparedReference(
            condition=condition,
            refs=refs,
            prompt_path=Path(ref.path),
            prompt_text=ref.text,
            payload={
                "gpt_cond_latent": gpt_cond_latent,
                "speaker_embedding": speaker_embedding,
            },
            metadata={
                "backend_revision": self.revision,
                "source_revision": self.source_revision,
                "model_revision": self.model_revision,
                "prompt_duration": ref.duration,
                "language": TARGET_LANGUAGE,
                "license_id": self.license_id,
                "usage_restriction": self.usage_restriction,
                "intervention": _condition_intervention(
                    condition,
                    self.embedding_ablation_mode,
                    self.gpt_control_reference_ids,
                    self.gpt_control_speaker_ids,
                ),
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
        generated: Any = self.model.inference(
            target_text,
            TARGET_LANGUAGE,
            prepared.payload["gpt_cond_latent"],
            prepared.payload["speaker_embedding"],
        )
        waveform = np.asarray(generated["wav"], dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            raise RuntimeError("XTTS produced no audio")
        return GenerationResult(
            waveform=waveform,
            sample_rate=self.sample_rate,
            metadata={
                "backend_revision": self.revision,
                "source_revision": self.source_revision,
                "model_revision": self.model_revision,
                "condition": prepared.condition.value,
                "language": TARGET_LANGUAGE,
                "seed": seed,
                "license_id": self.license_id,
                "usage_restriction": self.usage_restriction,
                "intervention": _condition_intervention(
                    prepared.condition,
                    self.embedding_ablation_mode,
                    self.gpt_control_reference_ids,
                    self.gpt_control_speaker_ids,
                ),
            },
        )
