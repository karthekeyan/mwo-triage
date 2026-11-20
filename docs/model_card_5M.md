---
license: apache-2.0
language: en
pipeline_tag: text-classification
base_model: google/bert_uncased_L-4_H-128_A-2
# datasets: TODO - fill in the published dataset id(s) once the dataset is pushed to the Hub
tags:
- maintenance
- manufacturing
- work-orders
- triage
- small-language-model
---

# mwo-triage-5M

## Example

*Placeholder — to be filled with real model output during packaging, once the inference wrapper (section 8) exists.*

## 1. Summary

mwo-triage-5M is the lightweight sibling of `mwo-triage-13M`: a "System 1" decision model that reads one short maintenance work-order text and instantly predicts two labels: **priority** (`STOP_NOW`, `TODAY`, `PLAN`, `MONITOR`) and **route** (`MECHANICAL`, `ELECTRICAL`, `HYDRAULIC_PNEUMATIC`, `INSTRUMENTATION_CONTROLS`, `LUBRICATION`, `ESCALATE`), with a calibrated confidence score for each. It has 4.78M parameters, is built on `google/bert_uncased_L-4_H-128_A-2` with two classification heads on a shared encoder, and runs offline on a CPU in about 1.8 ms per text. It is meant for edge devices, phones and other low-cost hardware where even the already-small `mwo-triage-13M` is too big or too slow. On the test set, `mwo-triage-5M` trades a small amount of accuracy for a large speed-up: priority macro-F1 0.8265 vs. 0.8486 and route macro-F1 0.8008 vs. 0.8219 for `mwo-triage-13M`, at 1.799 ms/text vs. 6.506 ms/text (about 3.6x faster).

## 2. Intended use

This model suggests a triage call for a human planner to confirm, review or override. It is **not** a safety system, does not replace a site's own safety procedures or isolation/lockout rules, and must not be used to make an automatic stop/isolate decision without a person in the loop. It is intended as a fast first pass over incoming work-order text, not a final determination.

### Out of scope

- Non-English text.
- Long free-form reports (the model is trained and evaluated on short work-order-style text, mostly under 8 words; see section 4).
- Text describing several faults at once.
- Non-maintenance tickets.
- Any automatic safety action (stopping, isolating or locking out equipment) taken without a person in the loop.

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

| model | priority macro-F1 (bal.) | route macro-F1 (bal.) | STOP_NOW recall (bal. / safety) | STOP_NOW precision (bal. / safety) | size (MB) | CPU latency (ms/text) |
|---|---|---|---|---|---|---|
| **mwo-triage-5M (this model)** | **0.8265** | **0.8008** | **0.6396 / 0.8378** | **0.6339 / 0.4650** | **18.28** | **1.799** |
| BERT L-4/H-128 frozen baseline | 0.4648 | 0.4255 | 0.1081 / — | 0.2667 / — | 18.28 | 1.799 |
| DeBERTa-v3-base teacher (reference) | 0.8675 | 0.8415 | 0.6757 / — | 0.7895 / — | 701.37 | 51.249 |

The frozen baseline uses the same base encoder with a simple classification head and no fine-tuning, confirming that full fine-tuning (this project's locked training choice) is what makes the model usable. The teacher is a much larger reference model (184M+ parameters, 701.37 MB) that `mwo-triage-13M` was distilled from; distillation was tried for this model too but not used (see section 6). The teacher is not released.

### Per-class route F1 (balanced mode)

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| MECHANICAL | 0.8886 | 0.9316 | 0.9096 | 702 |
| ELECTRICAL | 0.8675 | 0.8529 | 0.8602 | 238 |
| HYDRAULIC_PNEUMATIC | 0.8708 | 0.8626 | 0.8667 | 211 |
| INSTRUMENTATION_CONTROLS | 0.7476 | 0.7333 | 0.7404 | 105 |
| LUBRICATION | 0.8841 | 0.8026 | 0.8414 | 76 |
| ESCALATE | 0.6242 | 0.5536 | 0.5868 | 168 |

### Two operating modes

The model ships with a calibrated STOP_NOW probability threshold (t=0.2009, fit on validation data for STOP_NOW recall ≥ 0.85), giving a choice of operating point:

| mode | priority macro-F1 | route macro-F1 | STOP_NOW recall | STOP_NOW precision |
|---|---|---|---|---|
| Balanced (argmax, calibrated probs) | 0.8265 | 0.8008 | 0.6396 | 0.6339 |
| Safety (STOP_NOW if P ≥ t=0.2009, else argmax) | 0.8088 | 0.8008 | 0.8378 | 0.4650 |

Safety mode catches more true STOP_NOW cases at the cost of more false STOP_NOW alarms; it does not change the route head.

### Calibration (ECE)

Per-head temperature scaling was fit on validation data (priority T=1.4289, route T=1.3630).

| head | ECE before | ECE after |
|---|---|---|
| priority | 0.0513 | 0.0101 |
| route | 0.0700 | 0.0202 |

### Macro-F1 by text length (test)

| bucket | n | priority macro-F1 | route macro-F1 |
|---|---|---|---|
| ≤4 words | 406 | 0.7942 | 0.7701 |
| 5-7 words | 1008 | 0.8373 | 0.8042 |
| ≥8 words | 86 | 0.8346 | 0.8261 |

### Macro-F1 by real-text source (test)

| source | n | priority macro-F1 | route macro-F1 |
|---|---|---|---|
| maintie | 343 | 0.8302 | 0.7643 |
| maintnorm | 1157 | 0.8248 | 0.8076 |

## 5. Training data

Real training examples are work orders from **MaintNorm** and **MaintIE** (mining and heavy mobile equipment maintenance text). A set of verified synthetic factory work orders (CNC, presses, robots, conveyors, compressors, paint shops, etc.) was added to the training split only, to broaden coverage beyond mining-fleet language; the test set is real text only.

Labels follow `docs/labelling-rulebook.md` and were produced by two independent AI labelling passes — Claude Sonnet 5.5 (Pass A) and Claude Opus 5.5 (Pass B) — reading the same rulebook, over the complete real-text labelling pool (11,497 texts). Agreement between the two passes was 92.6% on priority (kappa 0.879), 92.9% on route (kappa 0.900), and 86.6% on both heads together (an example counts as agreeing only if priority and route both match).

Where the two passes disagreed:

- **Test and validation pools**: every disagreement was adjudicated by a third pass applying the rulebook's tie-break rules. 11 of these were decided by Claude Fable 5.2 (batch mode); the remaining 300 were decided by Claude Code in-session (`claude-sonnet-5`).
- **Training pool**: disagreements take Pass B's label rather than being dropped, after Pass B was found to be the more accurate labeller against the adjudicated validation labels (val priority macro-F1 0.975 vs. 0.923 for Pass A; see `docs/experiment-log.md` sections 3-4).

After adjudication, a small number of documented rule overrides were applied on top of whatever label survived, regardless of pool:

- **31 brake/steering cases**: any work order naming a brake or steering fault, u/s, not working or failed on mobile equipment was set to `STOP_NOW`/`P1-SAFETY`, excluding warning/fault-light-only wording, brake cooling-hose/circuit faults, steering-wheel comfort/adjustment features, and brake (signal) lights, which keep their original label (`scripts/apply_rulings.py`, `data/interim/rulings_overrides.jsonl`).
- **Retarder ruling**: a retarder reported as u/s or not working on mobile equipment is `STOP_NOW`/`MECHANICAL`/`P1-SAFETY`.
- **Hook ruling**: a bent or in-need-of-straightening tub hook is `STOP_NOW`/`MECHANICAL`/`P1-SAFETY`.

**No human domain-expert review** was done on any label; quality comes from the rulebook, the two-pass agreement process, the documented rule overrides above, and the limitations documented here and in the rulebook itself.

Synthetic data: 2,489 synthetic factory work orders were written in-session by Claude Code (claude-sonnet-5, under a Claude Pro subscription, with no API calls) to a target label and style plan (`configs/synthetic_plan.yaml`); a separate, fresh Claude Code session then relabelled them blind from the text alone, and the 1,814 (72.9%) where the blind labels matched the targets were kept. Synthetic examples are train-only.

Final split sizes: train 11,011 (9,197 real + 1,814 synthetic), val 800, test 1,500 — val and test are real text only.

## 6. Training procedure

Full fine-tune (no LoRA/QLoRA) of `google/bert_uncased_L-4_H-128_A-2` with two independent classification heads (priority, route) on a shared encoder, on the passB-default training data (train-pool Pass A/Pass B disagreements take Pass B's label rather than being dropped; see section 5).

- Encoder parameters: 4,782,464
- Learning rate: 3e-4, AdamW, 10% linear warmup then linear decay
- Batch size 64, max sequence length 32, seed 42, fp16
- Up to 25 epochs, early stopping on validation mean macro-F1 (patience 4); best epoch 7 (run `L4_a_earlystop_passb`)
- Validation data only was used for all model selection, calibration and threshold fitting; the test set was loaded exactly once, after those choices were locked.

**Distillation from the DeBERTa-v3 teacher was tried and not used for this model.** With the same distillation setup used for `mwo-triage-13M` (0.5 × hard-label cross-entropy + 0.5 × KL(teacher ‖ student) at temperature 2, scaled by T²), the distilled run scored slightly worse on validation than the plain passB-default run: val mean macro-F1 0.8293 vs. 0.8320, and STOP_NOW recall 0.6618 vs. 0.7059 (`docs/experiment-log.md` section 5). The passB-default (non-distilled) checkpoint was carried forward as the final pick for this model (`docs/experiment-log.md` section 6).

## 7. Limitations

- **`ESCALATE` is the weakest route** (F1 0.5868 on test, balanced mode) — by construction it covers vague, multi-discipline or non-maintenance text, which is the hardest class to separate.
- **`INSTRUMENTATION_CONTROLS` is the second-weakest route** (F1 0.7404).
- **English only.**
- Real training and test data is **mostly mining heavy-mobile-equipment language** (MaintNorm, MaintIE); performance on other industrial domains relies on the added synthetic factory data and may be weaker than the headline numbers suggest, particularly for text styles not resembling either source.
- **No label was reviewed by a certified maintenance engineer.** The rulebook itself states it reflects general industrial practice, is not a safety standard, and must not replace a site's own safety procedures.
- **Safety mode trades precision for recall**: STOP_NOW precision drops from 0.6339 to 0.4650 when recall is pushed from 0.6396 to 0.8378, i.e. more false STOP_NOW alarms in exchange for fewer missed ones.
- **Weakest on the shortest texts**: both priority and route macro-F1 are lowest on the ≤4-word bucket (0.7942 / 0.7701) and improve on longer texts (0.8373-0.8346 priority, 0.8042-0.8261 route for 5-7 and ≥8 words) — the opposite pattern from `mwo-triage-13M`, whose longest-text bucket is its weakest. The ≥8-word bucket has a small test support (n=86).
- Route performance is weaker on `maintie` texts than `maintnorm` texts (0.7643 vs. 0.8076 macro-F1).
- STOP_NOW itself is the hardest priority class even in balanced mode (F1 0.6368 vs. 0.84-0.92 for the other three priority classes).
- Smaller than `mwo-triage-13M` on every headline metric (priority macro-F1 0.8265 vs. 0.8486, route macro-F1 0.8008 vs. 0.8219); choose this model when latency or footprint, not peak accuracy, is the constraint.

## 8. How to use

*Placeholder — the packaging and inference format (e.g. a `transformers`-compatible wrapper, a custom `pipeline`, or raw PyTorch checkpoint loading) has not been decided yet. This section will be filled in before publishing.*

## 9. Credits and citations

Real training and test text is drawn from:

- **MaintNorm** — Bikaun, Hodkiewicz and Liu, WNUT 2024. MIT licence. NLP-TLP Group, University of Western Australia. Repo: https://github.com/nlp-tlp/maintnorm . Paper: https://aclanthology.org/2024.wnut-1.7
- **MaintIE** — Bikaun et al., LREC-COLING 2024. MIT licence. NLP-TLP Group, University of Western Australia. Repo: https://github.com/nlp-tlp/maintie . Paper: https://aclanthology.org/2024.lrec-main.954

Please credit MaintNorm and MaintIE and retain their licence notices if you use this model or the underlying data.

Base encoder: one of **Google's BERT miniatures** (`google/bert_uncased_L-4_H-128_A-2`, https://huggingface.co/google/bert_uncased_L-4_H-128_A-2), Apache-2.0 licence. Turc, Chang, Lee and Toutanova, "Well-Read Students Learn Better: On the Importance of Pre-training Compact Models" (2019).

## 10. Author

karthe05
