import inspect
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
import torch

from condsites.backends.base import Condition, PreparedReference
from condsites.backends.cosyvoice import CosyVoiceBackend, seed_everything
from condsites.backends.qwen import QwenBackend
from condsites.references import RefItem


def fake_frontend_input():
    return {
        "text": torch.ones(1, 2, dtype=torch.int32),
        "prompt_text": torch.ones(1, 2, dtype=torch.int32),
        "llm_prompt_speech_token": torch.ones(1, 3, dtype=torch.int32),
        "flow_prompt_speech_token": torch.ones(1, 3, dtype=torch.int32),
        "prompt_speech_feat": torch.ones(1, 6, 80),
        "llm_embedding": torch.ones(1, 192),
        "flow_embedding": torch.ones(1, 192),
    }


def make_ref(tmp_path: Path, ident: str, value: float = 0.1) -> RefItem:
    path = tmp_path / f"{ident}.wav"
    sf.write(path, np.full(800, value, dtype=np.float32), 16000)
    return RefItem(ident, "speaker", f"text {ident}", str(path), 16000, 0.05)


def prepared(condition: Condition, payload=None) -> PreparedReference:
    return PreparedReference(
        condition=condition,
        refs=(),
        prompt_path=None,
        prompt_text="",
        payload=payload or {},
    )


def test_xvector_only_keeps_embeddings_and_removes_sequence_inputs():
    backend = CosyVoiceBackend.__new__(CosyVoiceBackend)
    model_input = backend.apply_condition(fake_frontend_input(), Condition.XVECTOR_ONLY)
    assert set(model_input) == {"text", "llm_embedding", "flow_embedding"}


def test_sequence_only_uses_explicit_same_shape_zero_embeddings():
    backend = CosyVoiceBackend.__new__(CosyVoiceBackend)
    model_input = backend.apply_condition(fake_frontend_input(), Condition.SEQUENCE_ONLY)
    assert set(model_input) >= {"llm_embedding", "flow_embedding"}
    assert torch.equal(model_input["llm_embedding"], torch.zeros(1, 192))
    assert torch.equal(model_input["flow_embedding"], torch.zeros(1, 192))
    assert "llm_prompt_speech_token" in model_input


def test_neither_keeps_target_text_and_explicit_zero_embeddings():
    backend = CosyVoiceBackend.__new__(CosyVoiceBackend)
    model_input = backend.apply_condition(fake_frontend_input(), Condition.NEITHER)
    assert set(model_input) == {"text", "llm_embedding", "flow_embedding"}
    assert model_input["llm_embedding"].shape == (1, 192)
    assert model_input["flow_embedding"].shape == (1, 192)
    assert torch.count_nonzero(model_input["llm_embedding"]) == 0
    assert torch.count_nonzero(model_input["flow_embedding"]) == 0


def test_native_empty_sensitivity_omits_embeddings_only_when_explicit():
    model_input = CosyVoiceBackend.apply_condition(
        fake_frontend_input(),
        Condition.SEQUENCE_ONLY,
        embedding_ablation_mode="native_empty_sensitivity",
    )
    assert "llm_embedding" not in model_input
    assert "flow_embedding" not in model_input


def test_average_embeddings_normalizes_before_and_after_averaging():
    average = CosyVoiceBackend.average_embeddings(
        [torch.tensor([[3.0, 0.0]]), torch.tensor([[0.0, 2.0]])]
    )
    expected = torch.tensor([[2**-0.5, 2**-0.5]])
    assert torch.allclose(average, expected)
    assert torch.allclose(torch.linalg.vector_norm(average, dim=-1), torch.ones(1))


def test_qwen_average_embeddings_preserves_magnitude():
    single = torch.tensor([6.0, 8.0])
    assert torch.allclose(QwenBackend.average_embeddings([single]), single)
    average = QwenBackend.average_embeddings(
        [torch.tensor([6.0, 8.0]), torch.tensor([10.0, 0.0])]
    )
    assert torch.allclose(average, torch.tensor([8.0, 4.0]))


def test_cosyvoice_full_prepares_concatenated_frontend_input(tmp_path):
    refs = (make_ref(tmp_path, "a"), make_ref(tmp_path, "b"))
    frontend = Mock()
    frontend.frontend_zero_shot.return_value = fake_frontend_input()
    backend = CosyVoiceBackend.__new__(CosyVoiceBackend)
    backend.revision = "cosy-rev"
    backend.embedding_ablation_mode = "zero"
    backend.model = SimpleNamespace(frontend=frontend, sample_rate=22050)

    result = backend.prepare_reference(refs, Condition.FULL, tmp_path / "work")

    assert result.prompt_path == tmp_path / "work" / "prompt.wav"
    assert result.prompt_path.exists()
    assert result.prompt_text == "text a text b"
    assert result.payload["model_input"] is frontend.frontend_zero_shot.return_value
    args = frontend.frontend_zero_shot.call_args.args
    assert args[0:2] == ("", "text a text b")
    assert args[2] == str(result.prompt_path)
    assert args[3] == 22050
    assert args[4] == ""


def test_cosyvoice_xvector_prepares_normalized_mean_embedding(tmp_path):
    refs = (make_ref(tmp_path, "a"), make_ref(tmp_path, "b"))
    frontend = Mock()
    frontend._extract_spk_embedding.side_effect = [
        torch.tensor([[3.0, 0.0]]),
        torch.tensor([[0.0, 2.0]]),
    ]
    backend = CosyVoiceBackend.__new__(CosyVoiceBackend)
    backend.revision = "cosy-rev"
    backend.model = SimpleNamespace(frontend=frontend, sample_rate=22050)

    result = backend.prepare_reference(refs, Condition.XVECTOR_ONLY, tmp_path / "work")

    assert frontend._extract_spk_embedding.call_count == 2
    assert [call.args[0] for call in frontend._extract_spk_embedding.call_args_list] == [
        refs[0].path,
        refs[1].path,
    ]
    assert result.prompt_path is None
    assert torch.allclose(
        result.payload["embedding"], torch.tensor([[2**-0.5, 2**-0.5]])
    )


def test_cosyvoice_neither_prepares_no_reference_payload(tmp_path):
    backend = CosyVoiceBackend.__new__(CosyVoiceBackend)
    backend.revision = "cosy-rev"
    backend.model = SimpleNamespace()
    refs = (make_ref(tmp_path, "a"),)

    result = backend.prepare_reference(refs, Condition.NEITHER, tmp_path / "work")

    assert result.prompt_path is None
    assert result.payload == {}


def test_cosyvoice_synthesize_masks_condition_and_concatenates_chunks():
    frontend = Mock()
    frontend._extract_text_token.return_value = (
        torch.tensor([[4, 5]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    )
    tts = Mock(
        return_value=iter(
            [
                {"tts_speech": torch.tensor([[0.1, 0.2]])},
                {"tts_speech": torch.tensor([[0.3]])},
            ]
        )
    )
    backend = CosyVoiceBackend.__new__(CosyVoiceBackend)
    backend.revision = "cosy-rev"
    backend.model = SimpleNamespace(
        frontend=frontend, model=SimpleNamespace(tts=tts), sample_rate=22050
    )
    frontend_input = fake_frontend_input()

    result = backend.synthesize(
        prepared(Condition.SEQUENCE_ONLY, {"model_input": frontend_input}),
        "target",
        seed=7,
    )

    model_input = tts.call_args.kwargs
    assert torch.equal(model_input["text"], torch.tensor([[4, 5]], dtype=torch.int32))
    assert torch.equal(model_input["text_len"], torch.tensor([2], dtype=torch.int32))
    assert torch.equal(model_input["llm_embedding"], torch.zeros(1, 192))
    assert torch.equal(model_input["flow_embedding"], torch.zeros(1, 192))
    assert np.allclose(result.waveform, np.array([0.1, 0.2, 0.3]))
    assert result.sample_rate == 22050


def test_cosyvoice_xvector_synthesis_uses_embedding_for_both_paths():
    frontend = Mock()
    frontend._extract_text_token.return_value = (
        torch.tensor([[4, 5]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    )
    tts = Mock(return_value=iter([{"tts_speech": torch.tensor([[0.1]])}]))
    backend = CosyVoiceBackend.__new__(CosyVoiceBackend)
    backend.revision = "cosy-rev"
    backend.model = SimpleNamespace(
        frontend=frontend, model=SimpleNamespace(tts=tts), sample_rate=22050
    )
    embedding = torch.tensor([[1.0, 0.0]])

    backend.synthesize(
        prepared(Condition.XVECTOR_ONLY, {"embedding": embedding}), "target", seed=7
    )

    model_input = tts.call_args.kwargs
    assert set(model_input) == {"text", "llm_embedding", "flow_embedding"}
    assert model_input["llm_embedding"] is embedding
    assert model_input["flow_embedding"] is embedding


def test_cosyvoice_neither_synthesis_uses_text_and_exact_zero_embeddings():
    frontend = Mock()
    text = torch.tensor([[4, 5]], dtype=torch.int32)
    frontend._extract_text_token.return_value = (
        text,
        torch.tensor([2], dtype=torch.int32),
    )
    tts = Mock(return_value=iter([{"tts_speech": torch.tensor([[0.1]])}]))
    backend = CosyVoiceBackend.__new__(CosyVoiceBackend)
    backend.revision = "cosy-rev"
    backend.embedding_ablation_mode = "zero"
    backend.model = SimpleNamespace(
        frontend=frontend, model=SimpleNamespace(tts=tts), sample_rate=22050
    )

    backend.synthesize(prepared(Condition.NEITHER), "target", seed=7)

    model_input = tts.call_args.kwargs
    assert set(model_input) == {"text", "llm_embedding", "flow_embedding"}
    assert model_input["text"] is text
    assert torch.equal(model_input["llm_embedding"], torch.zeros(1, 192))
    assert torch.equal(model_input["flow_embedding"], torch.zeros(1, 192))


def test_cosyvoice_pinned_api_usage_passes_paths_and_consumes_generator():
    prepare_source = inspect.getsource(CosyVoiceBackend.prepare_reference)
    synthesize_source = inspect.getsource(CosyVoiceBackend.synthesize)

    assert "str(prompt_path)" in prepare_source
    assert "item.path" in prepare_source
    assert "for output in self.model.model.tts(**model_input)" in synthesize_source


def test_seed_everything_reproduces_python_numpy_and_torch_values():
    seed_everything(123)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )

    seed_everything(123)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


class FakePrompt:
    def __init__(
        self,
        ref_code,
        ref_spk_embedding,
        x_vector_only_mode,
        icl_mode,
        ref_text=None,
    ):
        self.ref_code = ref_code
        self.ref_spk_embedding = ref_spk_embedding
        self.x_vector_only_mode = x_vector_only_mode
        self.icl_mode = icl_mode
        self.ref_text = ref_text


def test_qwen_loads_on_gpu_with_bfloat16_by_default(monkeypatch):
    from_pretrained = Mock(return_value=object())
    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        SimpleNamespace(
            Qwen3TTSModel=SimpleNamespace(from_pretrained=from_pretrained)
        ),
    )

    QwenBackend("model-id", revision="revision")

    from_pretrained.assert_called_once_with(
        "model-id",
        revision="revision",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )


def test_qwen_full_uses_public_prompt_api_on_concatenated_reference(tmp_path):
    model = Mock()
    prompt = [object()]
    model.create_voice_clone_prompt.return_value = prompt
    backend = QwenBackend.__new__(QwenBackend)
    backend.model = model
    backend.revision = "qwen-rev"
    backend.language = "English"
    backend.silence_ms = 125
    refs = (make_ref(tmp_path, "a"), make_ref(tmp_path, "b"))

    result = backend.prepare_reference(refs, Condition.FULL, tmp_path / "work")

    model.create_voice_clone_prompt.assert_called_once_with(
        ref_audio=str(tmp_path / "work" / "prompt.wav"),
        ref_text="text a text b",
        x_vector_only_mode=False,
    )
    assert result.payload["voice_clone_prompt"] is prompt
    assert sf.info(result.prompt_path).frames == 3600


def test_qwen_xvector_uses_raw_mean_prompt(tmp_path):
    model = Mock()
    model.create_voice_clone_prompt.side_effect = [
        [FakePrompt(None, torch.tensor([[3.0, 0.0]]), True, False)],
        [FakePrompt(None, torch.tensor([[0.0, 2.0]]), True, False)],
    ]
    backend = QwenBackend.__new__(QwenBackend)
    backend.model = model
    backend.revision = "qwen-rev"
    backend.language = "English"
    refs = (make_ref(tmp_path, "a"), make_ref(tmp_path, "b"))

    result = backend.prepare_reference(
        refs, Condition.XVECTOR_ONLY, tmp_path / "work"
    )

    prompt = result.payload["voice_clone_prompt"]
    assert len(prompt) == 1
    assert isinstance(prompt[0], FakePrompt)
    assert prompt[0].ref_code is None
    assert prompt[0].x_vector_only_mode is True
    assert prompt[0].icl_mode is False
    assert torch.allclose(
        prompt[0].ref_spk_embedding, torch.tensor([[1.5, 1.0]])
    )


@pytest.mark.parametrize("condition", [Condition.SEQUENCE_ONLY, Condition.NEITHER])
def test_qwen_rejects_unsupported_conditions(tmp_path, condition):
    backend = QwenBackend.__new__(QwenBackend)
    backend.model = Mock()
    backend.revision = "qwen-rev"
    backend.language = "English"

    with pytest.raises(NotImplementedError, match=condition.value):
        backend.prepare_reference((make_ref(tmp_path, "a"),), condition, tmp_path)


def test_qwen_synthesize_uses_prepared_prompt():
    model = Mock()
    model.generate_voice_clone.return_value = ([np.array([0.1, 0.2])], 24000)
    backend = QwenBackend.__new__(QwenBackend)
    backend.model = model
    backend.revision = "qwen-rev"
    backend.language = "English"
    prompt = [object()]

    result = backend.synthesize(
        prepared(Condition.FULL, {"voice_clone_prompt": prompt}), "target", seed=9
    )

    model.generate_voice_clone.assert_called_once_with(
        text="target", language="English", voice_clone_prompt=prompt
    )
    assert np.allclose(result.waveform, np.array([0.1, 0.2]))
    assert result.sample_rate == 24000
