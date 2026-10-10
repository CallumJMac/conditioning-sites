from types import SimpleNamespace

import pytest
import torch

from condsites.backends.base import Condition, PreparedReference
from condsites.backends.cosyvoice_sites import CosyVoiceSitesBackend, parse_site


def test_parse_site():
    assert parse_site("site:P1L0F1") == (True, False, True)
    with pytest.raises(ValueError):
        parse_site("full")


@pytest.mark.parametrize("site,lm_zero,flow_zero", [("site:P0L1F1", False, False), ("site:P0L0F1", True, False), ("site:P0L1F0", False, True), ("site:P0L0F0", True, True)])
def test_synthesize_zeroes_only_the_requested_sites(site, lm_zero, flow_zero):
    captured = {}

    def tts(**model_input):
        captured.update(model_input)
        yield {"tts_speech": torch.zeros(1, 10)}

    backend = CosyVoiceSitesBackend.__new__(CosyVoiceSitesBackend)
    backend.model = SimpleNamespace(
        frontend=SimpleNamespace(_extract_text_token=lambda text: (torch.ones(1, 3), torch.tensor([3]))),
        model=SimpleNamespace(tts=tts),
        sample_rate=22050,
    )
    embedding = torch.ones(1, 192)
    prepared = PreparedReference(condition=Condition.XVECTOR_ONLY, refs=(), prompt_path=None, prompt_text="", payload={"embedding": embedding}, metadata={"site": site})
    backend.synthesize(prepared, "hello", 42)
    assert torch.count_nonzero(captured["llm_embedding"]).item() == (0 if lm_zero else 192)
    assert torch.count_nonzero(captured["flow_embedding"]).item() == (0 if flow_zero else 192)
    assert "llm_prompt_speech_token" not in captured
