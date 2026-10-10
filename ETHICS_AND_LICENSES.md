# Licences and ethics

Checked 2026-10-09 against each artefact's own licence metadata (Hugging Face model/dataset cards, GitHub licence files). Not legal advice.

## Models used for synthesis

| Model | Licence | Research use | Publish generated samples |
|---|---|---|---|
| Qwen3-TTS 0.6B / 1.7B Base | Apache-2.0 | Yes | Yes |
| CosyVoice-300M | Apache-2.0 | Yes | Yes |
| Fun-CosyVoice3-0.5B-2512 | Apache-2.0 | Yes | Yes |
| Chatterbox (Resemble AI) | MIT | Yes | Yes (outputs carry Resemble's imperceptible watermark) |
| XTTS-v2 (Coqui) | Coqui Public Model License 1.0.0, non-commercial for the model *and its output* | Yes: "research, experiment, and testing for the benefit of public knowledge" is listed as non-commercial | Yes, non-commercially, with no direct or indirect payment arising from the output; label them CPML-licensed |
| F5-TTS v1 | CC-BY-NC-4.0 (weights; trained on Emilia) | Yes | Yes, non-commercially with attribution |

## Evaluation models

| Model | Licence |
|---|---|
| Whisper-large-v3 | Apache-2.0 |
| Parakeet-TDT-0.6B-v3 | CC-BY-4.0 |
| UTMOS22 (SpeechMOS) | MIT |
| UTMOSv2 | MIT |
| WavLM-large SV (UniSpeech, Seed-TTS-eval checkpoint) | UniSpeech repository licence (GitHub reports NOASSERTION; see its LICENSE file). Used for scoring only; not redistributed. |

## Data

| Data | Licence | Notes |
|---|---|---|
| LibriTTS-R | CC-BY-4.0 | Derived from LibriVox public-domain audiobooks; speakers are volunteer readers who released their recordings publicly. |
| Seed-TTS-eval test-en | No licence file in the repository | Built from Common Voice (CC0). Use it for evaluation and cite it; do not redistribute the test set itself. |

## Implications for the repository and paper

1. The **code** can be released under any licence you choose (MIT or Apache-2.0 suggested). It contains no model weights.
2. **Audio samples**: release a curated subset. Mark XTTS-v2 and F5-TTS samples as non-commercial (CPML and CC-BY-NC-4.0 respectively), and state that the whole sample set is for research use. The work is non-commercial independent research, which both licences permit.
3. Do **not** redistribute model weights, the WavLM-large SV checkpoint, or the Seed-TTS-eval test set; link to their sources.
4. Cite every model, dataset and evaluator.

## Ethics

**Voice cloning risk.** The paper studies how speaker identity is carried by cloning systems, which is relevant to misuse (impersonation). Mitigations to state in the paper:
- Only public, consented corpora are used (LibriVox readers via LibriTTS-R; Common Voice contributors). No private voices are cloned.
- No new cloning capability is released; we analyse existing open models.
- The finding that identity can survive from an embedding alone has a defensive reading: speaker embeddings should be treated as sensitive, identity-bearing data. Say so explicitly.
- Released samples are of corpus speakers only, and the repository states that they are synthetic.

**Listening study (human participants).** As independent work there is no institutional review board, but:
- Use an independent ethics review if the target venue asks for one (several paid services review online minimal-risk studies), or document the study as minimal risk with informed consent.
- Prolific's own rules apply: informed consent, fair pay (at least GBP 9/h recommended; the study pays about GBP 11/h), the right to withdraw, and no deception.
- Collect only the Prolific ID and responses; store them securely; report only aggregates.
- The consent page (`listening_study/js/cmos.js`) still has placeholders for a contact address and an ethics reference.

**Disclosure.** Keep the paper's Generative AI Use Disclosure. Venue policies: TMLR, ICML and ACL ask for AI-use disclosure and a broader-impact or ethics statement.
