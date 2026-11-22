# Listening Study: Design and Deployment

Paired-comparison test of the paper's headline: each system's output with the
acoustic prompt ("full") vs with the prompt removed and the embedding kept
("x-vector only"), for Qwen3-TTS 1.7B (embedding conditions the language model)
and CosyVoice 3 (embedding in the decoder only), on LibriTTS-R and Seed-TTS-eval
(pathway_v2 audio). The key human test is whether removing the prompt costs
more perceived speaker identity for CosyVoice 3 than for Qwen3-TTS.

## Design

| Item | Choice | Basis |
|---|---|---|
| Naturalness | CMOS, 7-point A/B scale (-3..+3) | ITU-T P.800 Annex E (CCR); paired tests are more sensitive than absolute MOS for small differences |
| Speaker match | CSMOS: Reference clip, then "which of A/B is more like the same speaker" | Reference is a *different* utterance of the speaker, not the target recording |
| Stimuli | 60 pairs: 2 systems x 2 datasets x 15 items, drawn at random (seed 20261009); LibriTTS-R items use distinct speakers | No metric-based selection, to avoid biasing the result |
| Counterbalancing | Four lists from stratified quarters (every fourth item within each system x dataset cell): list A1 rates quarter 1 for naturalness and quarter 3 for speaker match, A2 quarters 2 and 4, B1 and B2 the reverse. Each listener hears each pair once (28 to 32 pairs plus 4 traps); block order, trial order and A/B side randomised per listener | Every list covers every cell on both axes; short sessions allow more listeners, which matters more than more items |
| Loudness | All speech normalised to -23 LUFS (ffmpeg loudnorm) | Prevents loudness driving preference |
| Headphone screen | In-study screening first: loudspeakers or any reported hearing difficulty end the session. Then a volume-setting page, then Huggins pitch, 6 x 3AFC trials, pass >= 5/6 | Milne et al., Behav. Res. Methods, 2021 (doi:10.3758/s13428-020-01514-0). Pilot (2026-11-22): 6 of 11 failed, all at 3/6 or below; all passes were 6/6 |
| Traps (per block) | Natural vs heavily degraded (expect clear preference); identical clips (expect ~0) | Exclude listeners failing more than one of the four |
| Other exclusions | Completion time < 1/3 of median | Pre-registered in `analyze_results.py` |
| Listeners | Two-stage sequential design. Stage 1: 40 listeners after exclusions. If the headline test (CosyVoice 3 vs Qwen3-TTS, speaker-match axis) has p < 0.0294, stop; otherwise recruit 40 more and test once at p < 0.0294 (Pocock boundary, overall alpha 0.05 over two looks). Fixed before any data are collected | Simulation from the dummy run (per-rating SD 1.8): power about 0.95 at an effect of 0.6 with 80 listeners; about 0.6 at 0.3 |
| Analysis | LMM, score ~ 1 + (1 \| listener) + (1 \| pair), per axis; Bonferroni over 2 axes; sign test over pairs. Headline: score ~ system on the speaker-match axis | Crossed random effects; reporting follows Kirkland et al., SSW 2023 (state scale labels and instructions) |
| Length / pay | ~22 min (pilot median 21.9 min, range 16.5 to 24.7); GBP 3.70 (~GBP 10/h). Screened-out participants are sent to a paid Prolific screening path | Prolific recommends at least GBP 9/h (pay-rates page, Jul 2026) |

The power of the design is indicative only. Check it with a simulation once the
dummy test gives a realistic per-rating SD.

## 1. Build stimuli

```bash
cd listening_study
# Synthesised audio for qwen17 and cosyvoice3 in ../audio/<system>/<job_id>.wav
# (regenerate with condsites.synthesize, or download the release archive).
../.venv/bin/python build_stimuli.py \
  --jobs ../jobs/pathway.jsonl --audio-dir ../audio \
  --bundle-root .. --seedtts-root ../data/seedtts_en \
  --output-dir .
```

## 2. Dummy test locally

```bash
python3 -m http.server 8000
```

Then open:

- `http://localhost:8000/?debug=1&list=A`: the full study. `debug=1` lets you answer without playing every clip to the end. Leave it off to experience the real pacing.
- `http://localhost:8000/?list=B`: the other list, with real enforcement.

With `resultsEndpoint` empty, the final page downloads `results_<id>.json`.
Analyse dummy runs with:

```bash
../.venv/bin/python analyze_results.py --results ~/Downloads --config configs/study.json
```

## 3. Before going live

1. **Ethics:** obtain approval or exemption, then fill in the contact and ethics reference in the consent text (`js/cmos.js`, `render_consent`).
2. **Results backend:** create a Google Sheet, paste `google_apps_script.js` into Extensions > Apps Script, deploy as a Web App (execute as you, access anyone), and put the URL in `configs/study.json` as `resultsEndpoint`.
3. **Completion code:** create the study on Prolific and paste its completion URL into `prolificCompletionUrl`. The code is visible in client JS, so verify every submission against the Raw sheet before approving it.
4. **Hosting:** any static host, e.g. a `gh-pages` branch. Audio is about 60 MB.
5. **Prolific settings:**
   - Study URL: `https://<host>/index.html?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}`
   - Filters: first language English, no hearing difficulties, approval rate at least 95%, desktop only.
   - Places: 5 for the pilot, then batches up to 40 analysable listeners (stage 1). Estimated time: 22 min.
6. **Pilot:** run 5 participants, check timings and trap pass rates, then release the rest.
