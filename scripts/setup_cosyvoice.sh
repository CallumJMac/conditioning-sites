#!/usr/bin/env bash
set -euo pipefail

SOURCE_REVISION="074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
MODEL_REVISION="24c40509c3c5ea6fe06b5f8790ff99e3714a6bee"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSYVOICE_HOME="${COSYVOICE_HOME:-${ROOT}/.external/CosyVoice}"
MODEL_HOME="${COSYVOICE_MODEL_HOME:-${ROOT}/.external/models/CosyVoice-300M}"
VENV="${COSYVOICE_VENV:-${ROOT}/.venv-cosyvoice}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

if [[ "${1:-}" == "--check" ]]; then
  test -x "${VENV}/bin/python" || { echo "Missing CosyVoice venv" >&2; exit 1; }
  test "$(git -C "${COSYVOICE_HOME}" rev-parse HEAD)" = "${SOURCE_REVISION}" ||
    { echo "CosyVoice source revision mismatch" >&2; exit 1; }
  test -z "$(git -C "${COSYVOICE_HOME}" status --porcelain)" ||
    { echo "CosyVoice source checkout is dirty" >&2; exit 1; }
  SUBMODULE_STATUS="$(git -C "${COSYVOICE_HOME}" submodule status --recursive)"
  if printf '%s\n' "${SUBMODULE_STATUS}" |
    awk 'substr($0,1,1) ~ /[-+U]/ {found=1} END {exit !found}'; then
    echo "CosyVoice submodule state is invalid" >&2
    exit 1
  fi
  test -f "${MODEL_HOME}/.model_revision" ||
    { echo "Missing model revision marker" >&2; exit 1; }
  test "$(tr -d '\r\n' < "${MODEL_HOME}/.model_revision")" = "${MODEL_REVISION}" ||
    { echo "CosyVoice model revision marker mismatch" >&2; exit 1; }
  PYTHONPATH="${ROOT}/src" "${VENV}/bin/python" -c \
    'import sys; from pathlib import Path; from condsites.backends.cosyvoice import verify_artifact_integrity; verify_artifact_integrity(Path(sys.argv[1]))' \
    "${MODEL_HOME}"
  PYTHONPATH="${COSYVOICE_HOME}:${COSYVOICE_HOME}/third_party/Matcha-TTS" \
    "${VENV}/bin/python" -c "from cosyvoice.cli.cosyvoice import CosyVoice"
  exit 0
fi

mkdir -p "$(dirname "${COSYVOICE_HOME}")" "$(dirname "${MODEL_HOME}")"
if [[ ! -d "${COSYVOICE_HOME}/.git" ]]; then
  git clone --recurse-submodules https://github.com/FunAudioLLM/CosyVoice.git "${COSYVOICE_HOME}"
fi
git -C "${COSYVOICE_HOME}" fetch origin "${SOURCE_REVISION}"
git -C "${COSYVOICE_HOME}" checkout --detach "${SOURCE_REVISION}"
git -C "${COSYVOICE_HOME}" submodule update --init --recursive

"${PYTHON_BIN}" -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip "setuptools<81" wheel
"${VENV}/bin/pip" install "numpy==1.26.4"
"${VENV}/bin/pip" install "jiwer==3.1.0"
"${VENV}/bin/pip" install --no-build-isolation "openai-whisper==20250625"
"${VENV}/bin/pip" install --no-build-isolation "pyworld==0.3.4"
FILTERED_REQUIREMENTS="$(mktemp)"
trap 'rm -f "${FILTERED_REQUIREMENTS}"' EXIT
awk \
  '!/^openai-whisper==/ && !/^pyworld==/' \
  "${COSYVOICE_HOME}/requirements.txt" > "${FILTERED_REQUIREMENTS}"
"${VENV}/bin/pip" install -r "${FILTERED_REQUIREMENTS}"
# Training-only DeepSpeed probes for a CUDA toolkit at import time. The pinned
# runtime image has CUDA libraries but no nvcc, and this pilot uses inference only.
"${VENV}/bin/pip" uninstall --yes deepspeed

if [[ ! -d "${MODEL_HOME}/.git" ]]; then
  if [[ -e "${MODEL_HOME}" ]]; then
    echo "Model directory exists but is not a git checkout: ${MODEL_HOME}" >&2
    exit 1
  fi
  GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout \
    https://huggingface.co/FunAudioLLM/CosyVoice-300M "${MODEL_HOME}"
fi
git -C "${MODEL_HOME}" fetch origin "${MODEL_REVISION}"
GIT_LFS_SKIP_SMUDGE=1 \
  git -C "${MODEL_HOME}" checkout --detach "${MODEL_REVISION}"
git -C "${MODEL_HOME}" lfs pull
printf '%s\n' "${MODEL_REVISION}" > "${MODEL_HOME}/.model_revision"

bash "$0" --check
