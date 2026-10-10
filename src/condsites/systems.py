"""The TTS systems studied, their pinned revisions, and backend construction.

Hugging Face revisions are the commits that were current when the paper's audio
was synthesised (2026-10-08); none of these repositories changed between the
runs and the time of writing. Source checkouts for CosyVoice, XTTS-v2 and
Chatterbox are pinned and verified by ``scripts/setup_<system>.sh``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / ".external"

# Held-out speakers whose mean GPT conditioning latents replace XTTS-v2's prompt
# latents in the x-vector-only condition (one female, one male, ~6 s each).
XTTS_CONTROL_REFERENCE_IDS = ("908_31957_000019_000001", "8555_284447_000047_000000")


@dataclass(frozen=True)
class System:
    name: str
    label: str
    model: str
    model_revision: str
    source_revision: str | None
    licence: str
    embedding_site: str
    environment: str


SYSTEMS = {
    "qwen06": System("qwen06", "Qwen3-TTS 0.6B", "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                     "5d83992436eae1d760afd27aff78a71d676296fc", None, "Apache-2.0", "LM input", "qwen"),
    "qwen17": System("qwen17", "Qwen3-TTS 1.7B", "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                     "fd4b254389122332181a7c3db7f27e918eec64e3", None, "Apache-2.0", "LM input", "qwen"),
    "cosyvoice1": System("cosyvoice1", "CosyVoice-300M", "FunAudioLLM/CosyVoice-300M",
                         "24c40509c3c5ea6fe06b5f8790ff99e3714a6bee", "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc",
                         "Apache-2.0", "LM and flow decoder", "cosyvoice"),
    "cosyvoice3": System("cosyvoice3", "CosyVoice 3", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
                         "29e01c4e8d000f4bcd70751be16fa94bf3d85a18", "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc",
                         "Apache-2.0", "flow decoder only", "cosyvoice"),
    "xtts": System("xtts", "XTTS-v2", "coqui/XTTS-v2",
                   "6c2b0d75eae4b7047358e3b6bd9325f857d43f77", "dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e",
                   "CPML-1.0 (non-commercial)", "vocoder decoder only", "xtts"),
    "chatterbox": System("chatterbox", "Chatterbox", "ResembleAI/chatterbox",
                         "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18", "65b18437192794391a0308a8f705b1e33e633948",
                         "MIT", "LM (T3)", "chatterbox"),
    "f5tts": System("f5tts", "F5-TTS v1", "SWivid/F5-TTS",
                    "84e5a410d9cead4de2f847e7c9369a6440bdfaca", None, "CC-BY-NC-4.0", "none (prompt only)", "f5"),
}
# The embedding-site factorial wraps CosyVoice-300M.
SYSTEMS["cosyvoice1sites"] = SYSTEMS["cosyvoice1"]


def _xtts_control_references(manifest: Path):
    from condsites.references import load_manifest

    by_id = {item.id: item for item in load_manifest(manifest) if item.id in XTTS_CONTROL_REFERENCE_IDS}
    return tuple(by_id[reference_id] for reference_id in XTTS_CONTROL_REFERENCE_IDS)


def create_backend(name: str, manifest: Path = ROOT / "data/libritts_r_aligned/manifest_40spk.json"):
    """Instantiate the backend for a system name in ``SYSTEMS``.

    Every backend builds full-condition prompts at 24 kHz.
    """
    system = SYSTEMS[name]
    if name in ("qwen06", "qwen17"):
        from condsites.backends.qwen import QwenBackend

        backend = QwenBackend(model_id=system.model, revision=system.model_revision)
    elif name == "cosyvoice1":
        from condsites.backends.cosyvoice import CosyVoiceBackend

        backend = CosyVoiceBackend(EXTERNAL / "models/CosyVoice-300M", EXTERNAL / "CosyVoice",
                                   system.source_revision, system.model_revision)
    elif name == "cosyvoice1sites":
        from condsites.backends.cosyvoice_sites import CosyVoiceSitesBackend

        backend = CosyVoiceSitesBackend(EXTERNAL / "models/CosyVoice-300M", EXTERNAL / "CosyVoice",
                                        system.source_revision, system.model_revision)
    elif name == "cosyvoice3":
        from condsites.backends.cosyvoice3 import CosyVoice3Backend

        backend = CosyVoice3Backend(model_dir=EXTERNAL / "models/Fun-CosyVoice3-0.5B", source_dir=EXTERNAL / "CosyVoice")
    elif name == "xtts":
        from condsites.backends.xtts import XTTSBackend

        backend = XTTSBackend(EXTERNAL / "models/XTTS-v2", EXTERNAL / "TTS", system.source_revision, system.model_revision,
                              license_id="Coqui Public Model License 1.0.0", usage_restriction="research-only",
                              embedding_ablation_mode="gpt_latent_heldout_mean",
                              gpt_control_references=_xtts_control_references(manifest))
    elif name == "chatterbox":
        from condsites.backends.chatterbox import ChatterboxBackend

        backend = ChatterboxBackend(EXTERNAL / "models/chatterbox", EXTERNAL / "chatterbox",
                                    system.source_revision, system.model_revision)
    elif name == "f5tts":
        from condsites.backends.f5tts import F5TTSBackend

        backend = F5TTSBackend()
    else:
        raise KeyError(name)
    backend.prompt_sample_rate = 24000
    return backend


def systems_table() -> str:
    """Markdown table of systems, revisions and licences (used in the README)."""
    rows = ["| System | Hugging Face repo @ revision | Embedding enters | Licence |", "|---|---|---|---|"]
    for name, system in SYSTEMS.items():
        if name == "cosyvoice1sites":
            continue
        rows.append(f"| {system.label} | `{system.model}` @ `{system.model_revision[:10]}` | {system.embedding_site} | {system.licence} |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(json.dumps({name: system.__dict__ for name, system in SYSTEMS.items()}, indent=2))
