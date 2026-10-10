#!/bin/bash
# Create one virtualenv per synthesis system plus the scoring environment.
# Run inside the Docker image (see Dockerfile). Takes about 25 minutes on a fresh machine.
set -euo pipefail
cd "$(dirname "$0")/.."
PY310=$(uv python find 3.10)
PY311=$(uv python find 3.11)
CUDA_TORCH=(torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124)

uv venv -q .venv-qwen --python "$PY311"
uv pip install -q --python .venv-qwen/bin/python -e . qwen-tts==0.1.1
uv pip install -q --python .venv-qwen/bin/python "${CUDA_TORCH[@]}" --reinstall-package torch --reinstall-package torchaudio

uv venv -q .venv-f5 --python "$PY311"
uv pip install -q --python .venv-f5/bin/python -e . f5-tts==1.1.22
uv pip install -q --python .venv-f5/bin/python "${CUDA_TORCH[@]}" --reinstall-package torch --reinstall-package torchaudio

# CosyVoice-300M and CosyVoice 3 share the pinned CosyVoice source tree and environment.
PYTHON_BIN="$PY310" bash scripts/setup_cosyvoice.sh
.venv-cosyvoice/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', revision='29e01c4e8d000f4bcd70751be16fa94bf3d85a18', local_dir='.external/models/Fun-CosyVoice3-0.5B')"
PYTHON_BIN="$PY310" bash scripts/setup_xtts.sh
PYTHON_BIN="$PY310" bash scripts/setup_chatterbox.sh

uv venv -q .venv-scoring --python "$PY311"
uv pip install -q --python .venv-scoring/bin/python "${CUDA_TORCH[@]}"
uv pip install -q --python .venv-scoring/bin/python -e ".[scoring]"
echo "environments ready"
