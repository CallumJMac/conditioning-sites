#!/usr/bin/env bash
set -euo pipefail

SOURCE_REVISION="dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e"
MODEL_REVISION="6c2b0d75eae4b7047358e3b6bd9325f857d43f77"
SOURCE_REPOSITORY="https://github.com/coqui-ai/TTS.git"
MODEL_REPOSITORY="https://huggingface.co/coqui/XTTS-v2"
PYPI_METADATA_BASE="https://pypi.org/pypi"
PIP_VERSION="24.0"
SETUPTOOLS_VERSION="69.0.3"
WHEEL_VERSION="0.42.0"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XTTS_HOME="${XTTS_HOME:-${ROOT}/.external/TTS}"
MODEL_HOME="${XTTS_MODEL_HOME:-${ROOT}/.external/models/XTTS-v2}"
VENV="${XTTS_VENV:-${ROOT}/.venv-xtts}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
DEPENDENCY_PROVENANCE="${MODEL_HOME}/.runtime_dependencies.json"
RUNTIME_REQUIREMENTS=(
  "numpy==1.22.0"
  "cython==0.29.30"
  "scipy==1.11.2"
  "torch==2.1.0"
  "torchaudio==2.1.0"
  "soundfile==0.12.1"
  "librosa==0.10.0"
  "scikit-learn==1.3.0"
  "numba==0.57.0"
  "inflect==5.6.0"
  "tqdm==4.64.1"
  "anyascii==0.3.0"
  "pyyaml==6.0"
  "fsspec==2023.6.0"
  "aiohttp==3.8.1"
  "packaging==23.1"
  "mutagen==1.47.0"
  "flask==2.0.1"
  "pysbd==0.3.4"
  "umap-learn==0.5.1"
  "pandas==1.4.0"
  "matplotlib==3.7.0"
  "trainer==0.0.36"
  "coqpit==0.0.16"
  "jieba==0.42.1"
  "pypinyin==0.50.0"
  "hangul-romanize==0.1.0"
  "gruut[de,es,fr]==2.2.3"
  "jamo==0.4.1"
  "nltk==3.8.1"
  "g2pkk==0.1.2"
  "bangla==0.0.2"
  "bnnumerizer==0.0.2"
  "bnunicodenormalizer==0.1.6"
  "einops==0.6.0"
  "transformers==4.33.0"
  "encodec==0.1.1"
  "unidecode==1.3.2"
  "num2words==0.5.13"
  "spacy[ja]==3.7.2"
)

if [[ "${1:-}" == "--check" ]]; then
  test -x "${VENV}/bin/python" ||
    { echo "Missing XTTS Python 3.10 environment" >&2; exit 1; }
  "${VENV}/bin/python" -c \
    'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
  test "$(git -C "${XTTS_HOME}" rev-parse HEAD)" = "${SOURCE_REVISION}" ||
    { echo "XTTS source revision mismatch" >&2; exit 1; }
  test -z "$(git -C "${XTTS_HOME}" status --porcelain)" ||
    { echo "XTTS source checkout is dirty" >&2; exit 1; }
  test "$(git -C "${MODEL_HOME}" rev-parse HEAD)" = "${MODEL_REVISION}" ||
    { echo "XTTS model checkout revision mismatch" >&2; exit 1; }
  PYTHONPATH="${ROOT}/src:${XTTS_HOME}" "${VENV}/bin/python" -c \
    'import sys; from pathlib import Path; from condsites.backends.xtts import validate_xtts_runtime, verify_runtime_dependencies; validate_xtts_runtime(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]); verify_runtime_dependencies(Path(sys.argv[5]))' \
    "${XTTS_HOME}" "${MODEL_HOME}" "${SOURCE_REVISION}" "${MODEL_REVISION}" \
    "${DEPENDENCY_PROVENANCE}"
  PYTHONPATH="${XTTS_HOME}" "${VENV}/bin/python" -c \
    'from pathlib import Path; import TTS; from TTS.tts.configs.xtts_config import XttsConfig; from TTS.tts.models.xtts import Xtts; assert Path(TTS.__file__).resolve().is_relative_to(Path("'"${XTTS_HOME}"'").resolve()); assert hasattr(Xtts, "get_conditioning_latents"); assert hasattr(Xtts, "inference"); assert hasattr(Xtts, "load_checkpoint"); assert XttsConfig'
  exit 0
fi

mkdir -p "$(dirname "${XTTS_HOME}")" "$(dirname "${MODEL_HOME}")"
if [[ ! -d "${XTTS_HOME}/.git" ]]; then
  git clone "${SOURCE_REPOSITORY}" "${XTTS_HOME}"
fi
git -C "${XTTS_HOME}" fetch origin "${SOURCE_REVISION}"
git -C "${XTTS_HOME}" checkout --detach "${SOURCE_REVISION}"

"${PYTHON_BIN}" -m venv "${VENV}"
"${VENV}/bin/python" -m pip install \
  "pip==${PIP_VERSION}" "setuptools==${SETUPTOOLS_VERSION}" \
  "wheel==${WHEEL_VERSION}"
"${VENV}/bin/pip" install --no-build-isolation "${RUNTIME_REQUIREMENTS[@]}"
"${VENV}/bin/pip" install --no-build-isolation --no-deps "${XTTS_HOME}"

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
  --include="LICENSE.txt,config.json,model.pth,speakers_xtts.pth,vocab.json"

printf '%s\n' "${SOURCE_REVISION}" > "${MODEL_HOME}/.source_revision"
printf '%s\n' "${MODEL_REVISION}" > "${MODEL_HOME}/.model_revision"
{
  printf 'source_repository=%s\n' "${SOURCE_REPOSITORY}"
  printf 'source_revision=%s\n' "${SOURCE_REVISION}"
  printf 'model_repository=%s\n' "${MODEL_REPOSITORY}"
  printf 'model_revision=%s\n' "${MODEL_REVISION}"
  printf 'license=Coqui Public Model License 1.0.0\n'
  printf 'license_sha256=%s\n' \
    "190f6d7c19b8984f91b97712b94ce92d2b2e640fc677dacab966e955ece9d043"
} > "${MODEL_HOME}/.license_provenance"
cat > "${DEPENDENCY_PROVENANCE}" <<EOF
{
  "packages": {
    "Cython": "0.29.30",
    "Flask": "2.0.1",
    "PyYAML": "6.0",
    "Unidecode": "1.3.2",
    "aiohttp": "3.8.1",
    "anyascii": "0.3.0",
    "bangla": "0.0.2",
    "bnnumerizer": "0.0.2",
    "bnunicodenormalizer": "0.1.6",
    "coqpit": "0.0.16",
    "einops": "0.6.0",
    "encodec": "0.1.1",
    "fsspec": "2023.6.0",
    "g2pkk": "0.1.2",
    "gruut": "2.2.3",
    "hangul-romanize": "0.1.0",
    "inflect": "5.6.0",
    "jamo": "0.4.1",
    "jieba": "0.42.1",
    "librosa": "0.10.0",
    "matplotlib": "3.7.0",
    "mutagen": "1.47.0",
    "nltk": "3.8.1",
    "num2words": "0.5.13",
    "numba": "0.57.0",
    "numpy": "1.22.0",
    "packaging": "23.1",
    "pandas": "1.4.0",
    "pypinyin": "0.50.0",
    "pysbd": "0.3.4",
    "scikit-learn": "1.3.0",
    "scipy": "1.11.2",
    "soundfile": "0.12.1",
    "spacy": "3.7.2",
    "torch": "2.1.0",
    "torchaudio": "2.1.0",
    "tqdm": "4.64.1",
    "trainer": "0.0.36",
    "transformers": "4.33.0",
    "umap-learn": "0.5.1"
  },
  "pypi_metadata_base": "${PYPI_METADATA_BASE}",
  "schema_version": 1,
  "xtts_source_commit_date": "2024-02-10T14:20:58Z",
  "xtts_source_revision": "${SOURCE_REVISION}"
}
EOF

bash "$0" --check
