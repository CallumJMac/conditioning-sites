from __future__ import annotations

import importlib
import inspect
import json
import random
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import soundfile as sf

try:
    import librosa  # noqa: F401
except ModuleNotFoundError:
    sys.modules["librosa"] = SimpleNamespace()

from condsites.backends.base import Condition, PreparedReference
from condsites.references import RefItem

SOURCE_REVISION = "dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e"
MODEL_REVISION = "6c2b0d75eae4b7047358e3b6bd9325f857d43f77"
LICENSE_ID = "Coqui Public Model License 1.0.0"
USAGE_RESTRICTION = "research-only"


def xtts_backend():
    return importlib.import_module("condsites.backends.xtts")


class FakeTensor:
    def __init__(self, values, *, dtype="float16", device="cuda:0"):
        self.values = tuple(values)
        self.shape = (1, len(self.values), 1)
        self.dtype = dtype
        self.device = device

    def clone(self):
        return FakeTensor(self.values, dtype=self.dtype, device=self.device)

    def __add__(self, other):
        return FakeTensor(
            [left + right for left, right in zip(self.values, other.values, strict=True)],
            dtype=self.dtype,
            device=self.device,
        )

    def __truediv__(self, divisor):
        return FakeTensor(
            [value / divisor for value in self.values],
            dtype=self.dtype,
            device=self.device,
        )


def make_ref(tmp_path: Path, ident: str = "ref") -> RefItem:
    path = tmp_path / f"{ident}.wav"
    sf.write(path, np.full(800, 0.1, dtype=np.float32), 16000)
    return RefItem(ident, "speaker", "reference text", str(path), 16000, 0.05)


def make_backend(
    gpt_cond_latent=None,
    speaker_embedding=None,
    *,
    embedding_ablation_mode="zero_gpt_conditioning_latents",
    heldout_mean=None,
):
    module = xtts_backend()
    backend = module.XTTSBackend.__new__(module.XTTSBackend)
    backend.source_revision = SOURCE_REVISION
    backend.model_revision = MODEL_REVISION
    backend.revision = f"source:{SOURCE_REVISION};model:{MODEL_REVISION}"
    backend.license_id = LICENSE_ID
    backend.usage_restriction = USAGE_RESTRICTION
    backend.embedding_ablation_mode = embedding_ablation_mode
    backend.gpt_control_reference_ids = ()
    backend.gpt_control_speaker_ids = ()
    backend._heldout_mean_gpt_cond_latent = heldout_mean
    backend.model = SimpleNamespace(
        get_conditioning_latents=Mock(
            return_value=(
                gpt_cond_latent or FakeTensor([1.0, 2.0]),
                speaker_embedding or FakeTensor([3.0]),
            )
        ),
        inference=Mock(
            return_value={"wav": np.array([[0.1, 0.2, 0.3]], dtype=np.float64)}
        ),
    )
    backend.sample_rate = 24000
    return backend


def test_full_gets_conditioning_once_and_passes_both_tensors_unchanged(tmp_path):
    module = xtts_backend()
    gpt = FakeTensor([1.0, 2.0])
    speaker = FakeTensor([3.0])
    backend = make_backend(gpt, speaker)
    ref = make_ref(tmp_path)

    prepared = backend.prepare_reference((ref,), Condition.FULL, tmp_path / "work")

    backend.model.get_conditioning_latents.assert_called_once_with(
        audio_path=ref.path
    )
    assert prepared.payload["gpt_cond_latent"] is gpt
    assert prepared.payload["speaker_embedding"] is speaker
    assert prepared.metadata["intervention"] == {
        "condition": "full",
        "gpt_conditioning_latents": "retained",
        "speaker_embedding": "retained_identically",
    }
    assert module.TARGET_LANGUAGE == "en"


def test_xvector_only_zeros_only_gpt_latents_and_preserves_speaker(tmp_path, monkeypatch):
    xtts_backend()
    gpt = FakeTensor([1.0, 2.0], dtype="bfloat16", device="cuda:7")
    speaker = FakeTensor([3.0], dtype="float32", device="cuda:7")
    zero = FakeTensor([0.0, 0.0], dtype=gpt.dtype, device=gpt.device)
    zeros_like = Mock(return_value=zero)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(zeros_like=zeros_like))
    backend = make_backend(gpt, speaker)

    prepared = backend.prepare_reference(
        (make_ref(tmp_path),), Condition.XVECTOR_ONLY, tmp_path / "work"
    )

    zeros_like.assert_called_once_with(gpt)
    assert prepared.payload["gpt_cond_latent"] is zero
    assert zero.shape == gpt.shape
    assert zero.dtype == gpt.dtype
    assert zero.device == gpt.device
    assert prepared.payload["speaker_embedding"] is speaker
    assert speaker.values == (3.0,)
    assert prepared.metadata["intervention"] == {
        "condition": "xvector_only",
        "gpt_conditioning_latents": "zeroed_with_torch.zeros_like",
        "speaker_embedding": "retained_identically",
    }


def test_xvector_only_clones_heldout_mean_and_preserves_target_speaker(tmp_path):
    target_gpt = FakeTensor([9.0, 10.0], dtype="bfloat16", device="cuda:7")
    speaker = FakeTensor([3.0], dtype="float32", device="cuda:7")
    heldout_mean = FakeTensor([2.0, 4.0], dtype="bfloat16", device="cuda:7")
    backend = make_backend(
        target_gpt,
        speaker,
        embedding_ablation_mode="gpt_latent_heldout_mean",
        heldout_mean=heldout_mean,
    )
    backend.gpt_control_reference_ids = ("female-control", "male-control")
    backend.gpt_control_speaker_ids = ("8555", "908")

    prepared = backend.prepare_reference(
        (make_ref(tmp_path),), Condition.XVECTOR_ONLY, tmp_path / "work"
    )

    substituted = prepared.payload["gpt_cond_latent"]
    assert substituted is not heldout_mean
    assert substituted.values == (2.0, 4.0)
    assert substituted.shape == target_gpt.shape
    assert substituted.dtype == target_gpt.dtype
    assert substituted.device == target_gpt.device
    assert prepared.payload["speaker_embedding"] is speaker
    assert prepared.metadata["intervention"] == {
        "condition": "xvector_only",
        "gpt_conditioning_latents": "fixed_arithmetic_mean_of_heldout_female_male_controls",
        "gpt_control_reference_ids": ["female-control", "male-control"],
        "gpt_control_speaker_ids": ["8555", "908"],
        "speaker_embedding": "retained_identically",
    }


def test_heldout_mean_is_computed_once_from_exact_control_paths():
    module = xtts_backend()
    female = RefItem(
        "female-control", "8555", "female", "/exact/female.wav", 24000, 5.96, "F"
    )
    male = RefItem(
        "male-control", "908", "male", "/exact/male.wav", 24000, 5.52, "M"
    )
    female_gpt = FakeTensor([1.0, 3.0], dtype="float32", device="cuda:0")
    male_gpt = FakeTensor([3.0, 5.0], dtype="float32", device="cuda:0")
    model = SimpleNamespace(
        get_conditioning_latents=Mock(
            side_effect=[
                (female_gpt, FakeTensor([10.0])),
                (male_gpt, FakeTensor([20.0])),
            ]
        )
    )

    mean = module._compute_heldout_mean_gpt_latent(model, (female, male))

    assert model.get_conditioning_latents.call_args_list == [
        call(audio_path="/exact/female.wav"),
        call(audio_path="/exact/male.wav"),
    ]
    assert mean.values == (2.0, 4.0)
    assert mean.shape == female_gpt.shape
    assert mean.dtype == female_gpt.dtype
    assert mean.device == female_gpt.device


@pytest.mark.parametrize(
    "controls, message",
    [
        ((), "exactly two"),
        (
            (
                RefItem("a", "1", "", "/a.wav", 24000, 5.5, "F"),
                RefItem("b", "2", "", "", 24000, 5.5, "M"),
            ),
            "non-empty",
        ),
    ],
)
def test_heldout_mean_rejects_invalid_control_inputs(controls, message):
    module = xtts_backend()
    with pytest.raises(ValueError, match=message):
        module._compute_heldout_mean_gpt_latent(
            SimpleNamespace(get_conditioning_latents=Mock()), controls
        )


def test_heldout_mean_rejects_incompatible_latents():
    module = xtts_backend()
    first = FakeTensor([1.0, 2.0], dtype="float32", device="cpu")
    second = FakeTensor([3.0, 4.0], dtype="float16", device="cpu")
    model = SimpleNamespace(
        get_conditioning_latents=Mock(
            side_effect=[(first, FakeTensor([1.0])), (second, FakeTensor([1.0]))]
        )
    )
    controls = (
        RefItem("a", "1", "", "/a.wav", 24000, 5.5, "F"),
        RefItem("b", "2", "", "/b.wav", 24000, 5.5, "M"),
    )

    with pytest.raises(ValueError, match="shape, dtype, and device"):
        module._compute_heldout_mean_gpt_latent(model, controls)


def test_synthesize_uses_pinned_direct_inference_api_and_language(monkeypatch):
    module = xtts_backend()
    backend = make_backend()
    seeded = Mock()
    monkeypatch.setattr(module, "seed_everything", seeded)
    gpt, speaker = FakeTensor([0.0]), FakeTensor([3.0])
    prepared = PreparedReference(
        condition=Condition.XVECTOR_ONLY,
        refs=(),
        prompt_path=None,
        prompt_text="",
        payload={"gpt_cond_latent": gpt, "speaker_embedding": speaker},
    )

    result = backend.synthesize(prepared, "target text", seed=73)

    seeded.assert_called_once_with(73)
    backend.model.inference.assert_called_once_with(
        "target text", "en", gpt, speaker
    )
    assert result.waveform.shape == (3,)
    assert result.waveform.dtype == np.float32
    assert result.sample_rate == 24000
    assert result.metadata["seed"] == 73
    assert result.metadata["language"] == "en"
    assert result.metadata["license_id"] == LICENSE_ID
    assert result.metadata["usage_restriction"] == USAGE_RESTRICTION


@pytest.mark.parametrize("condition", [Condition.SEQUENCE_ONLY, Condition.NEITHER])
def test_rejects_unsupported_conditions(tmp_path, condition):
    backend = make_backend()
    with pytest.raises(NotImplementedError, match=condition.value):
        backend.prepare_reference((make_ref(tmp_path),), condition, tmp_path / "work")


@pytest.mark.parametrize("count", [0, 2])
def test_rejects_reference_counts_other_than_one(tmp_path, count):
    backend = make_backend()
    refs = tuple(make_ref(tmp_path, str(index)) for index in range(count))
    with pytest.raises(ValueError, match="exactly one reference"):
        backend.prepare_reference(refs, Condition.FULL, tmp_path / "work")


def test_seed_everything_reproduces_python_numpy_and_torch_values(monkeypatch):
    module = xtts_backend()
    fake_torch = SimpleNamespace(
        manual_seed=Mock(),
        cuda=SimpleNamespace(is_available=lambda: True, manual_seed_all=Mock()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    module.seed_everything(123)
    first = (random.random(), np.random.random())
    module.seed_everything(123)
    second = (random.random(), np.random.random())
    assert first == second
    assert fake_torch.manual_seed.call_args_list == [call(123), call(123)]
    assert fake_torch.cuda.manual_seed_all.call_args_list == [call(123), call(123)]




def test_backend_loader_uses_only_pinned_local_model():
    module = xtts_backend()
    source = inspect.getsource(module.XTTSBackend.__init__)
    assert "XttsConfig" in source
    assert "Xtts.init_from_config" in source
    assert "load_checkpoint" in source
    assert "checkpoint_dir=str(model_dir)" in source
    assert "download" not in source
    assert "from_pretrained" not in source
    assert source.index("verify_runtime_dependencies") < source.index("load_checkpoint")
    assert source.index("verify_import_provenance") < source.index("load_checkpoint")


@pytest.mark.parametrize(
    ("source_revision", "model_revision"),
    [("wrong-source", MODEL_REVISION), (SOURCE_REVISION, "wrong-model")],
)
def test_backend_rejects_unapproved_revisions_before_runtime_validation(
    tmp_path, monkeypatch, source_revision, model_revision
):
    module = xtts_backend()
    validate = Mock()
    monkeypatch.setattr(module, "validate_xtts_runtime", validate)
    with pytest.raises(ValueError, match="approved XTTS"):
        module.XTTSBackend(
            tmp_path / "model",
            tmp_path / "source",
            source_revision,
            model_revision,
            license_id=LICENSE_ID,
            usage_restriction=USAGE_RESTRICTION,
        )
    validate.assert_not_called()


@pytest.mark.parametrize(
    ("license_id", "usage_restriction"),
    [("other", USAGE_RESTRICTION), (LICENSE_ID, "commercial")],
)
def test_backend_rejects_unapproved_usage_before_runtime_validation(
    tmp_path, monkeypatch, license_id, usage_restriction
):
    module = xtts_backend()
    validate = Mock()
    monkeypatch.setattr(module, "validate_xtts_runtime", validate)

    with pytest.raises(ValueError, match="research-only"):
        module.XTTSBackend(
            tmp_path / "model",
            tmp_path / "source",
            SOURCE_REVISION,
            MODEL_REVISION,
            license_id=license_id,
            usage_restriction=usage_restriction,
        )

    validate.assert_not_called()




def test_runtime_validation_checks_source_model_license_and_artifacts(
    tmp_path, monkeypatch
):
    module = xtts_backend()
    source = tmp_path / "source"
    model = tmp_path / "model"
    source.mkdir()
    model.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "tracked").write_text("clean", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "tracked"], check=True)
    subprocess.run(
        [
            "git", "-C", str(source), "-c", "user.name=Test",
            "-c", "user.email=test@example.com", "commit", "-qm", "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    artifacts = {
        "LICENSE.txt": b"CPML fixture",
        "config.json": b"{}",
        "model.pth": b"model",
        "vocab.json": b"{}",
        "speakers_xtts.pth": b"speakers",
    }
    for name, contents in artifacts.items():
        (model / name).write_bytes(contents)
    subprocess.run(["git", "init", "-q", str(model)], check=True)
    subprocess.run(["git", "-C", str(model), "add", *artifacts], check=True)
    subprocess.run(
        [
            "git", "-C", str(model), "-c", "user.name=Test",
            "-c", "user.email=test@example.com", "commit", "-qm", "model fixture",
        ],
        check=True,
    )
    model_revision = subprocess.check_output(
        ["git", "-C", str(model), "rev-parse", "HEAD"], text=True
    ).strip()
    (model / ".source_revision").write_text(revision + "\n", encoding="utf-8")
    (model / ".model_revision").write_text(model_revision + "\n", encoding="utf-8")
    (model / ".license_provenance").write_text(
        f"source_repository={module.APPROVED_SOURCE_REPOSITORY}\n"
        f"source_revision={revision}\n"
        f"model_repository={module.APPROVED_MODEL_REPOSITORY}\n"
        f"model_revision={model_revision}\n"
        f"license={module.APPROVED_LICENSE_NAME}\n"
        f"license_sha256={module.EXPECTED_LICENSE_SHA256}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "EXPECTED_MODEL_ARTIFACT_SHA256",
        {name: sha256(contents).hexdigest() for name, contents in artifacts.items()},
    )

    module.validate_xtts_runtime(source, model, revision, model_revision)

    (source / "tracked").write_text("dirty", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty"):
        module.validate_xtts_runtime(source, model, revision, model_revision)
    (source / "tracked").write_text("clean", encoding="utf-8")
    (model / "model.pth").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="model.pth"):
        module.validate_xtts_runtime(source, model, revision, model_revision)


@pytest.mark.parametrize(
    "field",
    [
        "source_repository",
        "source_revision",
        "model_repository",
        "model_revision",
        "license",
        "license_sha256",
    ],
)
def test_license_provenance_rejects_every_mutated_marker_field(
    tmp_path, monkeypatch, field
):
    module = xtts_backend()
    source = tmp_path / "source"
    model = tmp_path / "model"
    source.mkdir()
    model.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "tracked").write_text("clean", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "tracked"], check=True)
    subprocess.run(
        [
            "git", "-C", str(source), "-c", "user.name=Test",
            "-c", "user.email=test@example.com", "commit", "-qm", "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    artifacts = {
        "LICENSE.txt": b"CPML fixture",
        "config.json": b"{}",
        "model.pth": b"model",
        "vocab.json": b"{}",
        "speakers_xtts.pth": b"speakers",
    }
    for name, contents in artifacts.items():
        (model / name).write_bytes(contents)
    subprocess.run(["git", "init", "-q", str(model)], check=True)
    subprocess.run(["git", "-C", str(model), "add", *artifacts], check=True)
    subprocess.run(
        [
            "git", "-C", str(model), "-c", "user.name=Test",
            "-c", "user.email=test@example.com", "commit", "-qm", "model",
        ],
        check=True,
    )
    model_revision = subprocess.check_output(
        ["git", "-C", str(model), "rev-parse", "HEAD"], text=True
    ).strip()
    (model / ".source_revision").write_text(revision + "\n", encoding="utf-8")
    (model / ".model_revision").write_text(model_revision + "\n", encoding="utf-8")
    marker = {
        "source_repository": module.APPROVED_SOURCE_REPOSITORY,
        "source_revision": revision,
        "model_repository": module.APPROVED_MODEL_REPOSITORY,
        "model_revision": model_revision,
        "license": module.APPROVED_LICENSE_NAME,
        "license_sha256": module.EXPECTED_LICENSE_SHA256,
    }
    marker[field] = "mutated"
    (model / ".license_provenance").write_text(
        "".join(f"{name}={value}\n" for name, value in marker.items()),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "EXPECTED_MODEL_ARTIFACT_SHA256",
        {name: sha256(contents).hexdigest() for name, contents in artifacts.items()},
    )

    with pytest.raises(RuntimeError, match="license provenance marker mismatch"):
        module.validate_xtts_runtime(source, model, revision, model_revision)


def test_pinned_artifact_and_cpml_hashes_match_immutable_model_metadata():
    module = xtts_backend()
    assert module.EXPECTED_LICENSE_SHA256 == (
        "190f6d7c19b8984f91b97712b94ce92d2b2e640fc677dacab966e955ece9d043"
    )
    assert module.EXPECTED_MODEL_ARTIFACT_SHA256 == {
        "LICENSE.txt": "190f6d7c19b8984f91b97712b94ce92d2b2e640fc677dacab966e955ece9d043",
        "config.json": "ef262b1454dd2a77e1461b0b2cd53e19b8a7624cc131b837d36df67356bc75e8",
        "model.pth": "c7ea20001c6a0a841c77e252d8409f6a74fb423e79b3206a0771ba5989776187",
        "speakers_xtts.pth": "f0f6137c19a4eab0cbbe4c99b5babacf68b1746e50da90807708c10e645b943b",
        "vocab.json": "928260878a59da8a72a2a5b7687fea29d5106137669d90945430fe17e415304a",
    }
























def test_runtime_dependency_provenance_fails_closed(tmp_path, monkeypatch):
    module = xtts_backend()
    marker = tmp_path / ".runtime_dependencies.json"
    marker.write_text(
        json.dumps(module.EXPECTED_RUNTIME_DEPENDENCY_PROVENANCE, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module.importlib_metadata,
        "version",
        lambda name: module.APPROVED_RUNTIME_DEPENDENCIES[name],
    )
    module.verify_runtime_dependencies(marker)
    marker.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dependency provenance"):
        module.verify_runtime_dependencies(marker)
