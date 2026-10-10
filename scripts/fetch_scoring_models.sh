#!/bin/bash
# Fetch the speaker-verification model used for SIM-o / SIM-p, exactly as in Seed-TTS-eval.
#   .external/eval/seed-tts-eval         repository at a pinned commit (UniSpeech ECAPA-TDNN code)
#   .external/eval/wavlm_large_finetune.pth   WavLM-large SV checkpoint (Google Drive, checksum-verified)
# Whisper-large-v3 and UTMOS download themselves on first use.
set -euo pipefail
cd "$(dirname "$0")/.."
SEED_TTS_EVAL_COMMIT=752f4297f090c46bb1a55a1f7439e5944ddefe8d
CHECKPOINT_SHA256=51f07e3b94d9e0262a6a675ef5a087be3dd09e8c62e9d886827f44f82fe7f94b
mkdir -p .external/eval
if [ ! -d .external/eval/seed-tts-eval ]; then
  git clone -q https://github.com/BytedanceSpeech/seed-tts-eval.git .external/eval/seed-tts-eval
fi
git -C .external/eval/seed-tts-eval checkout -q "$SEED_TTS_EVAL_COMMIT"
if [ ! -f .external/eval/wavlm_large_finetune.pth ]; then
  python -m gdown 1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP -O .external/eval/wavlm_large_finetune.pth
fi
echo "${CHECKPOINT_SHA256}  .external/eval/wavlm_large_finetune.pth" | shasum -a 256 -c -
