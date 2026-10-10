# Changelog

## 1.0.0 (2026-10-10)

First public release, accompanying the preprint.

- Pathway study: six systems, LibriTTS-R (34 speakers) and Seed-TTS-eval
  test-en (1,088 items), 25,509 utterances, full vs x-vector-only conditioning.
- CosyVoice-300M embedding-site factorial: prompt x LM embedding x decoder
  embedding, 6,480 utterances.
- Reference-count study (one vs three references) on Qwen3-TTS, CosyVoice 3
  and F5-TTS.
- Per-utterance scores, speaker-level analysis, and `make check`, which
  regenerates every table in the paper from the scores.
- Paired-comparison listening study (not yet run).
