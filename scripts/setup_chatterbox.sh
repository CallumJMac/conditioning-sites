#!/usr/bin/env bash
set -euo pipefail

SOURCE_REVISION="65b18437192794391a0308a8f705b1e33e633948"
MODEL_REVISION="5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
PERTH_REVISION="ce86c49d029f42272c1902eccb675556b9ed2330"
SOURCE_REPOSITORY="https://github.com/resemble-ai/chatterbox.git"
MODEL_REPOSITORY="https://huggingface.co/ResembleAI/chatterbox"
PERTH_REPOSITORY="https://github.com/resemble-ai/Perth.git"
PYPI_METADATA_BASE="https://pypi.org/pypi"
PIP_VERSION="26.1.2"
SETUPTOOLS_VERSION="82.0.1"
WHEEL_VERSION="0.47.0"
UV_VERSION="0.9.18"
UV_BUILD_VERSION="0.9.18"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHATTERBOX_HOME="${CHATTERBOX_HOME:-${ROOT}/.external/chatterbox}"
MODEL_HOME="${CHATTERBOX_MODEL_HOME:-${ROOT}/.external/models/chatterbox}"
VENV="${CHATTERBOX_VENV:-${ROOT}/.venv-chatterbox}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
RUNTIME_REQUIREMENTS=(
  "numpy==1.26.4"
  "librosa==0.11.0"
  "s3tokenizer==0.3.0"
  "torch==2.6.0"
  "torchaudio==2.6.0"
  "transformers==5.2.0"
  "diffusers==0.29.0"
  "conformer==0.3.2"
  "safetensors==0.5.3"
  "spacy-pkuseg==1.0.1"
  "pykakasi==2.3.0"
  "gradio==6.8.0"
  "pyloudnorm==0.2.0"
  "omegaconf==2.3.0"
)
PERTH_REQUIREMENT="resemble-perth @ git+${PERTH_REPOSITORY}@${PERTH_REVISION}"
DEPENDENCY_PROVENANCE="${MODEL_HOME}/.runtime_dependencies.json"

if [[ "${1:-}" == "--check" ]]; then
  test -x "${VENV}/bin/python" ||
    { echo "Missing Chatterbox Python 3.10 environment" >&2; exit 1; }
  "${VENV}/bin/python" -c \
    'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
  test "$(git -C "${CHATTERBOX_HOME}" rev-parse HEAD)" = "${SOURCE_REVISION}" ||
    { echo "Chatterbox source revision mismatch" >&2; exit 1; }
  test -z "$(git -C "${CHATTERBOX_HOME}" status --porcelain)" ||
    { echo "Chatterbox source checkout is dirty" >&2; exit 1; }
  test "$(git -C "${MODEL_HOME}" rev-parse HEAD)" = "${MODEL_REVISION}" ||
    { echo "Chatterbox model checkout revision mismatch" >&2; exit 1; }
  PYTHONPATH="${ROOT}/src:${CHATTERBOX_HOME}/src" "${VENV}/bin/python" -c \
    'import sys; from pathlib import Path; from condsites.backends.chatterbox import validate_chatterbox_runtime, verify_runtime_dependencies; validate_chatterbox_runtime(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]); verify_runtime_dependencies(Path(sys.argv[5]))' \
    "${CHATTERBOX_HOME}" "${MODEL_HOME}" "${SOURCE_REVISION}" "${MODEL_REVISION}" \
    "${DEPENDENCY_PROVENANCE}"
  PYTHONPATH="${CHATTERBOX_HOME}/src" "${VENV}/bin/python" -c \
    'from chatterbox.tts import ChatterboxTTS; assert hasattr(ChatterboxTTS, "from_local")'
  exit 0
fi

mkdir -p "$(dirname "${CHATTERBOX_HOME}")" "$(dirname "${MODEL_HOME}")"
if [[ ! -d "${CHATTERBOX_HOME}/.git" ]]; then
  git clone "${SOURCE_REPOSITORY}" "${CHATTERBOX_HOME}"
fi
git -C "${CHATTERBOX_HOME}" fetch origin "${SOURCE_REVISION}"
git -C "${CHATTERBOX_HOME}" checkout --detach "${SOURCE_REVISION}"

"${PYTHON_BIN}" -m venv "${VENV}"
export PATH="${VENV}/bin:${PATH}"
"${VENV}/bin/python" -m pip install \
  "pip==${PIP_VERSION}" "setuptools==${SETUPTOOLS_VERSION}" \
    "wheel==${WHEEL_VERSION}" "uv==${UV_VERSION}" \
    "uv-build==${UV_BUILD_VERSION}"
"${VENV}/bin/pip" install --no-build-isolation "${RUNTIME_REQUIREMENTS[@]}" "${PERTH_REQUIREMENT}"
"${VENV}/bin/pip" install --no-build-isolation --no-deps "${CHATTERBOX_HOME}"

if [[ ! -d "${MODEL_HOME}/.git" ]]; then
  if [[ -e "${MODEL_HOME}" ]]; then
    echo "Model directory exists but is not a git checkout: ${MODEL_HOME}" >&2
    exit 1
  fi
  GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout \
    "${MODEL_REPOSITORY}" "${MODEL_HOME}"
fi
git -C "${MODEL_HOME}" fetch origin "${MODEL_REVISION}"
GIT_LFS_SKIP_SMUDGE=1 \
  git -C "${MODEL_HOME}" checkout --detach "${MODEL_REVISION}"
git -C "${MODEL_HOME}" lfs pull \
  --include="ve.safetensors,t3_cfg.safetensors,s3gen.safetensors,tokenizer.json,conds.pt"

printf '%s\n' "${SOURCE_REVISION}" > "${MODEL_HOME}/.source_revision"
printf '%s\n' "${MODEL_REVISION}" > "${MODEL_HOME}/.model_revision"
{
  printf 'source_repository=%s\n' "${SOURCE_REPOSITORY}"
  printf 'source_revision=%s\n' "${SOURCE_REVISION}"
  printf 'model_repository=%s\n' "${MODEL_REPOSITORY}"
  printf 'model_revision=%s\n' "${MODEL_REVISION}"
  printf 'license=MIT\n'
  printf 'license_sha256=%s\n' \
    "4248e910a928849fe5815a0f9236e17fa07768d95b9193212752c464b93d6caa"
} > "${MODEL_HOME}/.license_provenance"
cat > "${DEPENDENCY_PROVENANCE}" <<EOF
{
  "chatterbox_source_commit_date": "2026-06-10T16:18:35Z",
  "chatterbox_source_revision": "${SOURCE_REVISION}",
  "packages": {
    "conformer": "0.3.2",
    "diffusers": "0.29.0",
    "gradio": "6.8.0",
    "librosa": "0.11.0",
    "numpy": "1.26.4",
    "omegaconf": "2.3.0",
    "pykakasi": "2.3.0",
    "pyloudnorm": "0.2.0",
    "resemble-perth": "1.0.1",
    "s3tokenizer": "0.3.0",
    "safetensors": "0.5.3",
    "spacy-pkuseg": "1.0.1",
    "torch": "2.6.0",
    "torchaudio": "2.6.0",
    "transformers": "5.2.0"
  },
  "perth": {
    "metadata": "https://raw.githubusercontent.com/resemble-ai/Perth/${PERTH_REVISION}/pyproject.toml",
    "repository": "${PERTH_REPOSITORY}",
    "revision": "${PERTH_REVISION}"
  },
  "pypi_metadata_base": "${PYPI_METADATA_BASE}",
  "schema_version": 1
}
EOF

bash "$0" --check
