from __future__ import annotations

import inspect
import json
import random
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import soundfile as sf

try:
    import librosa  # noqa: F401
except ModuleNotFoundError:
    sys.modules["librosa"] = SimpleNamespace()

from condsites.backends import chatterbox as chatterbox_backend
from condsites.backends.base import Condition, PreparedReference
from condsites.backends.chatterbox import (
    EMPTY_T3_PROMPT,
    ChatterboxBackend,
    seed_everything,
    validate_chatterbox_runtime,
    verify_artifact_integrity,
)
from condsites.references import RefItem

SOURCE_REVISION = "65b18437192794391a0308a8f705b1e33e633948"
MODEL_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"


def make_ref(tmp_path: Path, ident: str = "ref") -> RefItem:
    path = tmp_path / f"{ident}.wav"
    sf.write(path, np.full(800, 0.1, dtype=np.float32), 16000)
    return RefItem(ident, "speaker", "reference text", str(path), 16000, 0.05)


def make_conditionals():
    prompt_tokens = object()
    prompt_emb = object()
    speaker_emb = object()
    emotion_adv = object()
    t3 = SimpleNamespace(
        speaker_emb=speaker_emb,
        clap_emb=None,
        cond_prompt_speech_tokens=prompt_tokens,
        cond_prompt_speech_emb=prompt_emb,
        emotion_adv=emotion_adv,
    )
    return SimpleNamespace(t3=t3, gen={"s3_reference": object()})


def make_backend(conditionals):
    backend = ChatterboxBackend.__new__(ChatterboxBackend)
    backend.source_revision = SOURCE_REVISION
    backend.model_revision = MODEL_REVISION
    backend.revision = f"source:{SOURCE_REVISION};model:{MODEL_REVISION}"
    backend.model = SimpleNamespace(
        conds=None,
        sr=24000,
        prepare_conditionals=Mock(side_effect=lambda _path: setattr(backend.model, "conds", conditionals)),
        generate=Mock(return_value=np.array([[0.1, 0.2, 0.3]], dtype=np.float32)),
    )
    return backend


def test_verify_runtime_dependencies_called_before_from_local():
    source = inspect.getsource(ChatterboxBackend.__init__)
    assert "verify_runtime_dependencies" in source
    assert "from_local" in source
    assert source.index("verify_runtime_dependencies") < source.index("from_local")


def test_pinned_t3_null_prompt_is_none_not_an_empty_tensor():
    assert EMPTY_T3_PROMPT is None


def test_full_preserves_all_t3_and_s3gen_reference_conditioning(tmp_path):
    original = make_conditionals()
    backend = make_backend(original)
    ref = make_ref(tmp_path)

    prepared = backend.prepare_reference((ref,), Condition.FULL, tmp_path / "work")

    backend.model.prepare_conditionals.assert_called_once_with(ref.path)
    actual = prepared.payload["conditionals"]
    assert actual.t3.speaker_emb is original.t3.speaker_emb
    assert actual.t3.cond_prompt_speech_tokens is original.t3.cond_prompt_speech_tokens
    assert actual.t3.cond_prompt_speech_emb is original.t3.cond_prompt_speech_emb
    assert actual.gen is original.gen
    assert prepared.prompt_path == Path(ref.path)
    assert prepared.metadata["intervention"]["t3_prompt_speech_tokens"] == "retained"


def test_xvector_only_ablates_only_t3_prompt_and_preserves_speaker_and_s3gen(tmp_path):
    original = make_conditionals()
    backend = make_backend(original)
    ref = make_ref(tmp_path)

    prepared = backend.prepare_reference(
        (ref,), Condition.XVECTOR_ONLY, tmp_path / "work"
    )

    actual = prepared.payload["conditionals"]
    assert actual is not original
    assert actual.t3 is not original.t3
    assert actual.t3.speaker_emb is original.t3.speaker_emb
    assert actual.t3.cond_prompt_speech_tokens is EMPTY_T3_PROMPT
    assert actual.t3.cond_prompt_speech_emb is EMPTY_T3_PROMPT
    assert actual.t3.emotion_adv is original.t3.emotion_adv
    assert actual.gen is original.gen
    assert original.t3.cond_prompt_speech_tokens is not None
    assert prepared.metadata["intervention"] == {
        "condition": "xvector_only",
        "speaker_embedding": "retained",
        "t3_prompt_speech_tokens": "none",
        "t3_prompt_speech_embedding": "none",
        "s3gen_reference_conditioning": "retained_identically",
        "pinned_null_semantics": (
            "T3Cond.cond_prompt_speech_tokens=None and "
            "T3Cond.cond_prompt_speech_emb=None"
        ),
    }


@pytest.mark.parametrize("condition", [Condition.SEQUENCE_ONLY, Condition.NEITHER])
def test_rejects_unsupported_conditions(tmp_path, condition):
    backend = make_backend(make_conditionals())
    with pytest.raises(NotImplementedError, match=condition.value):
        backend.prepare_reference((make_ref(tmp_path),), condition, tmp_path / "work")


@pytest.mark.parametrize("count", [0, 2])
def test_rejects_reference_counts_other_than_one(tmp_path, count):
    backend = make_backend(make_conditionals())
    refs = tuple(make_ref(tmp_path, str(index)) for index in range(count))
    with pytest.raises(ValueError, match="exactly one reference"):
        backend.prepare_reference(refs, Condition.FULL, tmp_path / "work")


def test_synthesize_seeds_and_returns_flat_float32_waveform(monkeypatch):
    conditionals = make_conditionals()
    backend = make_backend(conditionals)
    seeded = Mock()
    monkeypatch.setattr(chatterbox_backend, "seed_everything", seeded)
    prepared = PreparedReference(
        condition=Condition.XVECTOR_ONLY,
        refs=(),
        prompt_path=None,
        prompt_text="",
        payload={"conditionals": conditionals},
    )

    result = backend.synthesize(prepared, "target text", seed=73)

    seeded.assert_called_once_with(73)
    backend.model.generate.assert_called_once_with("target text")
    assert backend.model.conds is conditionals
    assert result.waveform.shape == (3,)
    assert result.waveform.dtype == np.float32
    assert result.sample_rate == 24000
    assert result.metadata["seed"] == 73
    assert result.metadata["source_revision"] == SOURCE_REVISION
    assert result.metadata["model_revision"] == MODEL_REVISION


def test_seed_everything_reproduces_python_numpy_and_torch_values(monkeypatch):
    fake_torch = SimpleNamespace(
        manual_seed=Mock(),
        cuda=SimpleNamespace(is_available=lambda: True, manual_seed_all=Mock()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    seed_everything(123)
    first = (random.random(), np.random.random())
    seed_everything(123)
    second = (random.random(), np.random.random())
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert fake_torch.manual_seed.call_args_list == [
        call(123),
        call(123),
    ]
    assert fake_torch.cuda.manual_seed_all.call_args_list == [
        call(123),
        call(123),
    ]




def test_backend_loader_uses_only_pinned_local_model():
    source = inspect.getsource(ChatterboxBackend.__init__)
    assert "from_local" in source
    assert "from_pretrained" not in source
    assert "validate_chatterbox_runtime" in source


@pytest.mark.parametrize(
    ("source_revision", "model_revision"),
    [
        ("wrong-source", MODEL_REVISION),
        (SOURCE_REVISION, "wrong-model"),
    ],
)
def test_backend_rejects_unapproved_revisions_before_runtime_validation(
    tmp_path, monkeypatch, source_revision, model_revision
):
    validate = Mock()
    monkeypatch.setattr(chatterbox_backend, "validate_chatterbox_runtime", validate)

    with pytest.raises(ValueError, match="approved Chatterbox"):
        ChatterboxBackend(
            model_dir=tmp_path / "model",
            source_dir=tmp_path / "source",
            source_revision=source_revision,
            model_revision=model_revision,
        )

    validate.assert_not_called()


def test_backend_rejects_wrong_import_before_model_loading(tmp_path, monkeypatch):
    load = Mock()
    fake_class = type(
        "ChatterboxTTS",
        (),
        {"__module__": "chatterbox.tts", "from_local": load},
    )
    package = ModuleType("chatterbox")
    package.__path__ = []
    tts_module = ModuleType("chatterbox.tts")
    tts_module.__file__ = str(tmp_path / "unpinned" / "chatterbox" / "tts.py")
    tts_module.ChatterboxTTS = fake_class
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "chatterbox", package)
    monkeypatch.setitem(sys.modules, "chatterbox.tts", tts_module)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        chatterbox_backend, "validate_chatterbox_runtime", lambda *args: None
    )
    monkeypatch.setattr(
        chatterbox_backend, "verify_runtime_dependencies", lambda *args: None
    )

    with pytest.raises(RuntimeError, match="pinned source checkout"):
        ChatterboxBackend(
            model_dir=tmp_path / "model",
            source_dir=tmp_path / "source",
            source_revision=SOURCE_REVISION,
            model_revision=MODEL_REVISION,
        )

    load.assert_not_called()


def test_runtime_validation_checks_source_model_and_license_provenance(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    model = tmp_path / "model"
    source.mkdir()
    model.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "LICENSE").write_text("MIT fixture", encoding="utf-8")
    (source / "tracked").write_text("clean", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "LICENSE", "tracked"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    artifacts = {
        "ve.safetensors": b"ve",
        "t3_cfg.safetensors": b"t3",
        "s3gen.safetensors": b"s3",
        "tokenizer.json": b"tokenizer",
        "conds.pt": b"conds",
    }
    for name, contents in artifacts.items():
        (model / name).write_bytes(contents)
    subprocess.run(["git", "init", "-q", str(model)], check=True)
    subprocess.run(["git", "-C", str(model), "add", *artifacts], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(model),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "model fixture",
        ],
        check=True,
    )
    model_revision = subprocess.check_output(
        ["git", "-C", str(model), "rev-parse", "HEAD"], text=True
    ).strip()
    (model / ".model_revision").write_text(model_revision + "\n", encoding="utf-8")
    (model / ".source_revision").write_text(revision + "\n", encoding="utf-8")
    (model / ".license_provenance").write_text(
        f"source_revision={revision}\nmodel_revision={model_revision}\nlicense=MIT\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        chatterbox_backend,
        "EXPECTED_MODEL_ARTIFACT_SHA256",
        {name: sha256(contents).hexdigest() for name, contents in artifacts.items()},
    )
    monkeypatch.setattr(
        chatterbox_backend,
        "EXPECTED_LICENSE_SHA256",
        sha256(b"MIT fixture").hexdigest(),
    )

    validate_chatterbox_runtime(source, model, revision, model_revision)

    (source / "tracked").write_text("dirty", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty"):
        validate_chatterbox_runtime(source, model, revision, model_revision)
    (source / "tracked").write_text("clean", encoding="utf-8")
    (model / ".model_revision").write_text("wrong\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="model revision"):
        validate_chatterbox_runtime(source, model, revision, model_revision)
    (model / ".model_revision").write_text(model_revision + "\n", encoding="utf-8")
    (model / "new-commit").write_text("changes head only", encoding="utf-8")
    subprocess.run(["git", "-C", str(model), "add", "new-commit"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(model),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "wrong model revision",
        ],
        check=True,
    )
    with pytest.raises(RuntimeError, match="model checkout revision mismatch"):
        validate_chatterbox_runtime(source, model, revision, model_revision)


def test_official_model_hashes_match_immutable_repository_metadata():
    assert chatterbox_backend.EXPECTED_MODEL_ARTIFACT_SHA256 == {
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


def test_artifact_hash_mismatch_fails_without_self_blessing(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    expected = {}
    for name in chatterbox_backend.REQUIRED_MODEL_ARTIFACTS:
        contents = f"fixture-{name}".encode()
        (model / name).write_bytes(contents)
        expected[name] = sha256(contents).hexdigest()
    monkeypatch.setattr(
        chatterbox_backend, "EXPECTED_MODEL_ARTIFACT_SHA256", expected
    )
    verify_artifact_integrity(model)
    (model / "s3gen.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="s3gen.safetensors"):
        verify_artifact_integrity(model)
    assert not any(model.glob("*sha256*"))








def test_runtime_dependency_pins_are_complete_and_immutable():
    assert chatterbox_backend.APPROVED_RUNTIME_DEPENDENCIES == {
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
    assert chatterbox_backend.APPROVED_PERTH_REVISION == (
        "ce86c49d029f42272c1902eccb675556b9ed2330"
    )


def test_runtime_dependency_provenance_fails_closed(tmp_path, monkeypatch):
    marker = tmp_path / ".runtime_dependencies.json"
    marker.write_text(
        json.dumps(
            chatterbox_backend.EXPECTED_RUNTIME_DEPENDENCY_PROVENANCE,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        chatterbox_backend.importlib_metadata,
        "version",
        lambda name: chatterbox_backend.APPROVED_RUNTIME_DEPENDENCIES[name],
    )
    direct_url = json.dumps(
        {
            "url": "https://github.com/resemble-ai/Perth.git",
            "vcs_info": {
                "vcs": "git",
                "commit_id": chatterbox_backend.APPROVED_PERTH_REVISION,
                "requested_revision": chatterbox_backend.APPROVED_PERTH_REVISION,
            },
        }
    )
    monkeypatch.setattr(
        chatterbox_backend.importlib_metadata,
        "distribution",
        lambda _name: SimpleNamespace(read_text=lambda _path: direct_url),
    )

    chatterbox_backend.verify_runtime_dependencies(marker)

    marker.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dependency provenance"):
        chatterbox_backend.verify_runtime_dependencies(marker)
