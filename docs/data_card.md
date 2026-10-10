# Data card: released scores and samples

## What is released

| File | Rows | Content |
|---|---|---|
| `jobs/pathway.jsonl` | 25,534 | Design of the pathway and reference-count studies: one line per utterance with system, condition, reference paths and transcripts, target text, seed and held-out recording |
| `jobs/factorial.jsonl` | 6,480 | Design of the CosyVoice-300M embedding-site factorial (condition `site:P?L?F?`) |
| `jobs/heldout_map.json` | 34 | LibriTTS-R held-out recording per speaker, with durations |
| `results/scores/pathway_scores.csv` | 25,509 | Per-utterance SIM-p, SIM-o, WER, UTMOS and Whisper transcript; 9 utterances could not be scored and have `SCORING_ERROR` in the transcript column |
| `results/scores/factorial_scores.csv` | 6,480 | The same for the factorial |
| `results/scores/pathway_status.jsonl` | 25,534 | Synthesis status, duration and runtime for every planned utterance, including the 25 failures |
| `results/analysis/*.csv` | | Speaker-level contrasts, condition means and reference-count effects |
| `docs/samples/` | | MP3 samples for four randomly drawn items (see `scripts/build_samples.py`) |

Join scores to the design on `clip_id` = `job_id`.

## Sources

- LibriTTS-R `test-clean` (CC-BY-4.0): 34 speakers, 17 female and 17 male by
  the LibriTTS speaker metadata. Prompts are a single 6 s recording per speaker.
- Seed-TTS-eval test-en: 1,088 Common Voice prompt/target pairs from 666 prompt
  speakers. The test set is not redistributed here.

## Metrics

| Column | Definition |
|---|---|
| `sim_heldout` (SIM-o) | Cosine similarity of WavLM-large SV embeddings, output vs a held-out recording of the speaker |
| `sim_prompt` (SIM-p) | The same, output vs the prompt recording |
| `wer` | Whisper-large-v3 word error rate after Seed-TTS-eval normalisation; can exceed 1 |
| `utmos22` | UTMOS (`utmos22_strong`) predicted naturalness, 1 to 5 |
| `utmosv2` | Present for 1,111 pathway utterances scored before UTMOSv2 was switched off for speed; not used in the paper |

## Known limitations

- English only; read audiobook speech and crowd-sourced Common Voice speech.
- Automatic metrics only. UTMOS rises when speaker information is removed
  (factorial, Seed-TTS-eval: 4.04 with both embeddings to 4.35 with none), so it
  partly rewards generic voices.
- SIM uses one speaker-verification model; absolute values are not comparable
  with papers that use a different checkpoint.
- Seed-TTS-eval utterances use a single seed; LibriTTS-R utterances use three.

## Intended use

Analysing how conditioning routes and sites carry speaker identity, and
reproducing the paper. Not for building systems that impersonate the corpus
speakers.
