---
license: apache-2.0
language: en
pipeline_tag: text-classification
base_model: google/electra-small-discriminator
tags:
- maintenance
- manufacturing
- work-orders
- triage
- small-language-model
---

# mwo-triage-13M

## 1. Summary

mwo-triage-13M is a "System 1" decision model: an encoder classifier that reads one short maintenance work-order text and instantly predicts two labels: **priority** (`STOP_NOW`, `TODAY`, `PLAN`, `MONITOR`) and **route** (`MECHANICAL`, `ELECTRICAL`, `HYDRAULIC_PNEUMATIC`, `INSTRUMENTATION_CONTROLS`, `LUBRICATION`, `ESCALATE`), with a calibrated confidence score for each. It has 13.5M parameters, is built on `google/electra-small-discriminator` with two classification heads on a shared encoder, and runs offline on a CPU in about 6.5 ms per text.

## 2. Intended use

This model suggests a triage call for a human planner to confirm, review or override. It is **not** a safety system, does not replace a site's own safety procedures or isolation/lockout rules, and must not be used to make an automatic stop/isolate decision without a person in the loop. It is intended as a fast first pass over incoming work-order text, not a final determination.

## 3. Labels

Full definitions, worked examples and tie-break rules are in `docs/labelling-rulebook.md` (v1.3). Short summary:

**Priority** (checked in this order; first match wins):
- `STOP_NOW` — immediate response; safety trigger (fire, leak spraying/on a hot surface/in a people area, a missing or bypassed safety device, brake/steering failure, uncontrolled movement, loose wheel nuts, etc.) or a stated line/critical-machine stoppage or rapidly worsening failure.
- `TODAY` — degraded but running, or down with a workaround; likely to get worse within about a day if not actioned.
- `PLAN` — a real defect with no immediate safety, production or quality effect; can wait for a planned window.
- `MONITOR` — no repair action needed yet; observe or recheck.

**Route** (by the component or system to be repaired, not the machine name or symptom):
- `MECHANICAL` — bearings, gears, shafts, belts, structures, mechanical brakes, buckets, liners, etc.
- `ELECTRICAL` — motors, cables, breakers, contactors, drives, batteries, panels, etc.
- `HYDRAULIC_PNEUMATIC` — hoses, cylinders, valves, hydraulic/pneumatic pumps, air lines, compressors.
- `INSTRUMENTATION_CONTROLS` — sensors, transmitters, PLCs, HMIs, calibration, control-loop and communication faults.
- `LUBRICATION` — greasing, oil top-up/change, auto-lube faults.
- `ESCALATE` — no component or discipline-specific symptom given, several disciplines equally likely with no clue, not a maintenance job, or unreadable text. A priority label is still required.

## 4. Results on 1,500 real test texts

`data/processed/test.jsonl` (1,500 real texts, never seen in training or any val-only tuning step). Source: `docs/evaluation.md`.

### Headline table

| model | priority F1 (bal.) | route F1 (bal.) | STOP_NOW recall (bal. / safety) | STOP_NOW precision (bal. / safety) | size (MB) | CPU latency (ms/text) |
|---|---|---|---|---|---|---|
| **mwo-triage-13M (this model)** | **0.8486** | **0.8219** | **0.7207 / 0.8288** | **0.6897 / 0.5412** | **51.52** | **6.506** |
| ELECTRA-small frozen baseline | 0.3448 | 0.1192 | 0.0000 / — | 0.0000 / — | 51.52 | 6.545 |
| DeBERTa-v3-base teacher (reference) | 0.8675 | 0.8415 | 0.6757 / — | 0.7895 / — | 701.37 | 51.249 |

The frozen baseline uses the same base encoder with a simple classification head and no fine-tuning, confirming that full fine-tuning (this project's locked training choice) is what makes the model usable. The teacher is a much larger reference model (184M+ parameters, 701.37 MB) that this model was distilled from; it is not released.

### Per-class route F1 (balanced mode)

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| MECHANICAL | 0.9003 | 0.9259 | 0.9129 | 702 |
| ELECTRICAL | 0.8826 | 0.8529 | 0.8675 | 238 |
| HYDRAULIC_PNEUMATIC | 0.8565 | 0.8768 | 0.8665 | 211 |
| INSTRUMENTATION_CONTROLS | 0.7714 | 0.7714 | 0.7714 | 105 |
| LUBRICATION | 0.9014 | 0.8421 | 0.8707 | 76 |
| ESCALATE | 0.6667 | 0.6190 | 0.6420 | 168 |

### Two operating modes

The model ships with a calibrated STOP_NOW probability threshold (t=0.1596, fit on validation data for STOP_NOW recall ≥ 0.85), giving a choice of operating point:

| mode | priority macro-F1 | route macro-F1 | STOP_NOW recall | STOP_NOW precision |
|---|---|---|---|---|
| Balanced (argmax, calibrated probs) | 0.8486 | 0.8219 | 0.7207 | 0.6897 |
| Safety (STOP_NOW if P ≥ t=0.1596, else argmax) | 0.8312 | — | 0.8288 | 0.5412 |

Safety mode catches more true STOP_NOW cases at the cost of more false STOP_NOW alarms; it does not change the route head.

### Calibration (ECE)

Per-head temperature scaling was fit on validation data (priority T=1.9388, route T=1.9800).

| head | ECE before | ECE after |
|---|---|---|
| priority | 0.0887 | 0.0242 |
| route | 0.1095 | 0.0367 |

### Macro-F1 by text length (test)

| bucket | n | priority macro-F1 | route macro-F1 |
|---|---|---|---|
| ≤4 words | 406 | 0.8566 | 0.8076 |
| 5-7 words | 1008 | 0.8512 | 0.8205 |
| ≥8 words | 86 | 0.8286 | 0.7935 |

### Macro-F1 by real-text source (test)

| source | n | priority macro-F1 | route macro-F1 |
|---|---|---|---|
| maintie | 343 | 0.8364 | 0.7760 |
| maintnorm | 1157 | 0.8520 | 0.8317 |

## 5. Training data

Real training examples are work orders from **MaintNorm** and **MaintIE** (mining and heavy mobile equipment maintenance text). A set of verified synthetic factory work orders (CNC, presses, robots, conveyors, compressors, paint shops, etc.) was added to the training split only, to broaden coverage beyond mining-fleet language; the test set is real text only.

Labels follow `docs/labelling-rulebook.md` and were produced by two independent AI labelling passes — Claude Sonnet 5.5 (Pass A) and Claude Opus 5.5 (Pass B) — reading the same rulebook. Validation and test disagreements between the two passes were adjudicated by a third pass using the rulebook's tie-break rules; training-pool disagreements take Pass B's label rather than being dropped, after Pass B was found to be the more accurate labeller against adjudicated validation labels. Overall agreement between the two passes was 86.6%. **There was no human domain-expert review** of any label; quality comes from the rulebook, the two-pass agreement process, and the limitations documented here and in the rulebook itself.

## 6. Training procedure

Full fine-tune (no LoRA/QLoRA) of `google/electra-small-discriminator` with two independent classification heads (priority, route) on a shared encoder, distilled from a separately fine-tuned `microsoft/deberta-v3-base` teacher (two-head setup, lr 3e-5, batch size 32, max sequence length 32).

- Encoder parameters: 13,483,008
- Learning rate: 1e-4, AdamW, 10% linear warmup then linear decay
- Batch size 64, max sequence length 32, seed 42, fp16
- Up to 25 epochs, early stopping on validation mean macro-F1 (patience 4); best epoch 11
- Distillation loss per head: 0.5 × hard-label cross-entropy + 0.5 × KL(student ‖ teacher) at temperature 2, scaled by T²
- Validation data only was used for all model selection, calibration and threshold fitting; the test set was loaded exactly once, after those choices were locked.

## 7. Limitations

- **`ESCALATE` is the weakest route** (F1 0.6420 on test, balanced mode) — by construction it covers vague, multi-discipline or non-maintenance text, which is the hardest class to separate.
- **`INSTRUMENTATION_CONTROLS` is the second-weakest route** (F1 0.7714).
- **English only.**
- Real training and test data is **mostly mining heavy-mobile-equipment language** (MaintNorm, MaintIE); performance on other industrial domains relies on the added synthetic factory data and may be weaker than the headline numbers suggest, particularly for text styles not resembling either source.
- **No label was reviewed by a certified maintenance engineer.** The rulebook itself states it reflects general industrial practice, is not a safety standard, and must not replace a site's own safety procedures.
- **Safety mode trades precision for recall**: STOP_NOW precision drops from 0.6897 to 0.5412 when recall is pushed from 0.7207 to 0.8288, i.e. more false STOP_NOW alarms in exchange for fewer missed ones.
- Performance degrades somewhat on longer texts (route macro-F1 0.8205 at 5-7 words vs. 0.7935 at ≥8 words) and the ≥8-word bucket has a small test support (n=86).
- Route performance is weaker on `maintie` texts than `maintnorm` texts (0.7760 vs. 0.8317 macro-F1).
- STOP_NOW itself is the hardest priority class even in balanced mode (F1 0.7048 vs. 0.87-0.92 for the other three priority classes).

## 8. How to use

*Placeholder — the packaging and inference format (e.g. a `transformers`-compatible wrapper, a custom `pipeline`, or raw PyTorch checkpoint loading) has not been decided yet. This section will be filled in before publishing.*

## 9. Credits and citations

Real training and test text is drawn from:

- **MaintNorm** — Bikaun, Hodkiewicz and Liu, WNUT 2024. MIT licence. NLP-TLP Group, University of Western Australia. Repo: https://github.com/nlp-tlp/maintnorm . Paper: https://aclanthology.org/2024.wnut-1.7
- **MaintIE** — Bikaun et al., LREC-COLING 2024. MIT licence. NLP-TLP Group, University of Western Australia. Repo: https://github.com/nlp-tlp/maintie . Paper: https://aclanthology.org/2024.lrec-main.954

Please credit MaintNorm and MaintIE and retain their licence notices if you use this model or the underlying data.

Base encoder: **Google ELECTRA** (`google/electra-small-discriminator`, https://huggingface.co/google/electra-small-discriminator), Apache-2.0 licence.

## 10. Author

karthe05
