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
| Counterbalancing | Eight lists from stratified eighths (every eighth item within each system x dataset cell): list k rates eighth k for naturalness and eighth k+4 for speaker match. Each listener hears each pair once (12 to 16 pairs plus 4 traps); block order, trial order and A/B side randomised per listener | Every list covers every cell on both axes; short sessions keep pay per person low |
| Loudness | All speech normalised to -23 LUFS (ffmpeg loudnorm) | Prevents loudness driving preference |
| Headphone screen | In-study screening first: loudspeakers or any reported hearing difficulty end the session. Then a volume-setting page, then Huggins pitch, 6 x 3AFC trials, pass >= 5/6 | Milne et al., Behav. Res. Methods, 2021 (doi:10.3758/s13428-020-01514-0). Pilot (2026-11-22): 6 of 11 failed, all at 3/6 or below; all passes were 6/6 |
| Traps (per block) | Natural vs heavily degraded (expect clear preference); identical clips (expect ~0) | Exclude listeners failing more than one of the four |
| Other exclusions | Completion time < 1/3 of median | Pre-registered in `analyze_results.py` |
| Listeners | 20 listeners after exclusions (recruit until reached), analysed once, no interim looks | Interspeech reviewers asked for a substantial listening test with proper statistics; the rebuttal committed to 20. Simulation from the pilot (per-rating SD 1.8): power about 0.84 for the CosyVoice 3 speaker-match effect (-0.9) at Holm-corrected alpha |
| Analysis | Primary: per system and axis, LMM score ~ 1 + (1 \| listener) + (1 \| pair); Holm over the 4 tests; direction compared with SIM-o and UTMOS. Exploratory: score ~ system on each axis; sign test over pairs | Answers whether the metric differences are perceptible (Interspeech R1 to R3); reporting follows Kirkland et al., SSW 2023 |
| Length / pay | ~12 min (pilot: about 0.6 min per trial plus setup; 16 to 20 trials); GBP 2.00 (~GBP 10/h). Screened-out participants can be sent to a paid Prolific screening path | Prolific recommends at least GBP 9/h (pay-rates page, Jul 2026) |

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
   - Places: start with 20 and add places until 20 listeners pass exclusions. Estimated time: 12 min.
6. **Pilot:** run 5 participants, check timings and trap pass rates, then release the rest.
