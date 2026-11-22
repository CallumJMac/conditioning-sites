# Preregistration (AsPredicted format)

Answers to paste into https://aspredicted.org. Written 2026-11-22, before any
main-study data were collected. A 5-person pilot (2026-11-22) was used only to
check timing and data flow and is excluded from all analyses.

**1. Have any data been collected for this study already?**
No, no data have been collected for this study yet. (A 5-participant pilot was
run to test the procedure; its data will not be analysed.)

**2. What's the main question being asked or hypothesis being tested?**
Zero-shot voice-cloning systems condition on a speaker through an acoustic prompt
and a speaker embedding. Automatic metrics show that removing the prompt (keeping
the embedding) costs more speaker similarity in CosyVoice 3, whose embedding
enters only the acoustic decoder, than in Qwen3-TTS 1.7B, whose embedding also
conditions the language model. We test whether human listeners agree:
H1: removing the prompt reduces perceived speaker match more for CosyVoice 3 than
for Qwen3-TTS 1.7B. Secondary: whether removing the prompt changes perceived
naturalness, overall and by test set.

**3. Describe the key dependent variable(s) specifying how they will be measured.**
Paired comparisons on a 7-point scale (-3 "A much better" to +3 "B much better"),
recoded so positive means the prompt-free version is preferred.
Speaker match (CSMOS): after a reference clip of the speaker (a different
utterance), "Which sample sounds more like the same speaker as the Reference?"
Naturalness (CMOS): "Which sample sounds more natural, like a real person
speaking? Ignore background noise and recording quality."

**4. How many and which conditions will participants be assigned to?**
Within-subject. 60 pairs (each the same sentence synthesised with and without the
prompt): 2 systems (Qwen3-TTS 1.7B, CosyVoice 3) x 2 test sets (LibriTTS-R,
Seed-TTS-eval) x 15 items, drawn at random. Each participant is assigned one of
four counterbalanced lists and rates about 15 pairs per axis plus 2 trap trials
per axis. A/B side, block order and trial order are randomised.

**5. Specify exactly which analyses you will conduct to examine the main question/hypothesis.**
Per axis, a linear mixed model score ~ 1 + (1 | listener) + (1 | pair); the
intercept is the mean preference (Bonferroni over the two axes). H1: on the
speaker-match axis, score ~ system with the same random effects; H1 is supported
if the CosyVoice 3 effect is more negative than Qwen3-TTS's, two-sided test at the
sequential boundary below. Robustness: sign test on per-pair means. Cell means
(system x test set) are reported for both axes.

**6. Describe exactly how outliers will be defined and handled, and your precise rule(s) for excluding observations.**
Participants are excluded, in order, if they: (a) report loudspeakers or any
hearing difficulty, or score below 5/6 on the Huggins-pitch headphone check
(these end the session); (b) fail more than one of four trap trials (a degraded
trap not rated at least 1 point towards the natural clip; an identical-clip trap
rated 2 or more points from 0); (c) finish in under one third of the median
completion time. All exclusions are reported.

**7. How many observations will be collected or what will determine sample size?**
Two-stage sequential design. Stage 1: recruit until 40 participants remain after
exclusions. If H1's test has p < 0.0294, stop. Otherwise recruit 40 more and test
once at p < 0.0294 (Pocock boundary, overall alpha 0.05 over two looks). Both
looks are reported.

**8. Anything else you would like to pre-register?**
Analysis code: listening_study/analyze_results.py in
https://github.com/CallumJMac/conditioning-sites (commit at registration).
Participants: Prolific, first language English, no hearing difficulties,
approval rate at least 95%, desktop only; paid about GBP 10/h.
