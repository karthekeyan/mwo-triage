---
license: apache-2.0
language: en
pipeline_tag: text-classification
base_model: google/electra-small-discriminator
datasets:
- karthe05/mwo-triage-data
tags:
- maintenance
- manufacturing
- work-orders
- triage
- small-language-model
---

# mwo-triage-13M

## Why this model exists

**The problem.** In many plants, planners still read every maintenance request by hand to decide its urgency and which team should handle it, even when the plant already has SAP PM or Maximo.

**Why it hurts.** Manual sorting is slow and inconsistent. Some jobs get marked urgent when they are not, while genuinely urgent problems can wait in the queue. Planners can spend a large part of their day just sorting requests instead of fixing anything.

**Why it matters.** Delays on genuinely urgent jobs mean downtime, safety risk and lost production. When sorting is inconsistent, technicians may end up working on less important jobs first while urgent ones wait.

**How it is often done today.** The operator picks a priority from a dropdown when logging the request. A planner or supervisor then corrects it, often in the morning meeting. Genuine emergencies get reported by phone instead of waiting for this process.

**What this model does.** It reads one request and suggests a priority and a team, each with a confidence score. The planner still reviews and approves the suggestion. Vague requests are routed to `ESCALATE` rather than guessed at, and low-confidence suggestions can be flagged for manual review instead of being acted on automatically. It runs inside the plant on an ordinary computer, at about 6.5 ms per request on CPU, with no GPU, no cloud dependency and no per-request cost.

**How well it works.** On 1,500 real work orders it had never seen, it scored macro-F1 0.85 for priority and 0.82 for team. It caught 72% of real emergencies in balanced mode and 83% in safety mode.

## Example

This is real output from the published model in balanced mode (see section 8):

| text | priority | priority_confidence | route | route_confidence |
|---|---|---|---|---|
| "Hyd hose burst on press 3, oil spraying across walkway" | STOP_NOW | 0.96 | HYDRAULIC_PNEUMATIC | 0.95 |
| "c/o position 3 tyre" | PLAN | 0.96 | MECHANICAL | 0.97 |
| "Gearbox oil level low, top up" | PLAN | 0.96 | LUBRICATION | 0.95 |

### Borderline case

"Hydraulic press 3 leaking oil near main cylinder, pressure dropping during cycle" → **STOP_NOW** (priority_confidence 0.54), **HYDRAULIC_PNEUMATIC** (route_confidence 0.99).

According to the rulebook, the correct priority for this example is `TODAY`. The leak is not spraying, is not on a hot surface or in an area where people are present, and the machine is still running with reduced performance.

The model predicts `STOP_NOW`, so its priority prediction is wrong. However, its confidence is only 0.54. This is just above 50%, which means the model is almost equally unsure between two answers. A human planner should review this result instead of acting on it automatically.

This example shows why confidence scores matter. A wrong prediction with high confidence could be more dangerous. Use `priority_confidence` as a guide to how much to trust a prediction, not just as an extra number beside the predicted label.

## 1. Summary

mwo-triage-13M is a fast decision model for short maintenance work-order text. It reads one work order and predicts two things:

- **Priority:** `STOP_NOW`, `TODAY`, `PLAN` or `MONITOR`.
- **Route:** `MECHANICAL`, `ELECTRICAL`, `HYDRAULIC_PNEUMATIC`, `INSTRUMENTATION_CONTROLS`, `LUBRICATION` or `ESCALATE`.

The model also gives a calibrated confidence score for each prediction.

The model has 13.5M parameters. It uses `google/electra-small-discriminator` as its base model and has two classification heads that share the same encoder. It can run offline on a CPU and takes about 6.5 ms to process one text.

## 2. Intended use

The model gives a first suggestion about the priority and route of a maintenance work order. A human planner must review the suggestion and can confirm or change it.

This model is **not a safety system**. It does not replace a site's safety procedures or its isolation and lockout rules. Never use it to stop or isolate equipment automatically without a person making the decision.

Use it as a quick first check of incoming work orders, not as the final decision-maker.

### Out of scope

Do not use this model for:

- Text written in languages other than English.
- Long, free-form reports. The model was trained and tested on short work-order-style text, mostly under 8 words (see section 4).
- Text that describes several faults at the same time.
- Tickets that are not related to maintenance.
- Automatic safety actions, such as stopping, isolating or locking out equipment, without a person in the decision process.

## 3. Labels

The full label definitions, examples and rules for deciding between labels are in [`docs/labelling-rulebook.md`](https://github.com/karthekeyan/mwo-triage/blob/main/docs/labelling-rulebook.md) (v1.3).

Here is a short summary.

**Priority** (check the labels in this order. Use the first one that matches):

- `STOP_NOW` — The issue needs an immediate response. Examples include fire; a leak that is spraying, on a hot surface or in an area where people are present; a missing or bypassed safety device; brake or steering failure; uncontrolled movement; loose wheel nuts; or a stated stoppage of a production line or critical machine. It also applies when a failure is getting worse quickly.
- `TODAY` — The machine is still running with reduced performance, or it is stopped but has a workaround. The problem is likely to get worse within about a day if nobody acts.
- `PLAN` — There is a real defect, but it has no immediate safety, production or quality impact. It can wait until a planned maintenance window.
- `MONITOR` — No repair is needed yet. The issue only needs to be watched or checked again.

**Route** (choose the route based on the component or system that needs repair, not the machine name or the symptom):

- `MECHANICAL` — Bearings, gears, shafts, belts, structures, mechanical brakes, buckets, liners and similar parts.
- `ELECTRICAL` — Motors, cables, breakers, contactors, drives, batteries, panels and similar parts.
- `HYDRAULIC_PNEUMATIC` — Hoses, cylinders, valves, hydraulic or pneumatic pumps, air lines and compressors.
- `INSTRUMENTATION_CONTROLS` — Sensors, transmitters, PLCs, HMIs, calibration, control-loop problems and communication faults.
- `LUBRICATION` — Greasing, oil top-ups or changes, and automatic lubrication faults.
- `ESCALATE` — Use this when the text does not name a component or give a clear discipline-specific symptom; when several routes seem equally likely and there is no clue; when the ticket is not a maintenance job; or when the text cannot be read. A priority label is still required.

## 4. Results on 1,500 real test texts

The results below use the [test split](https://huggingface.co/datasets/karthe05/mwo-triage-data/viewer/default/test) of [`karthe05/mwo-triage-data`](https://huggingface.co/datasets/karthe05/mwo-triage-data). It contains 1,500 real texts that were not used during training or validation-only tuning. See [`docs/evaluation.md`](https://github.com/karthekeyan/mwo-triage/blob/main/docs/evaluation.md).

### Results at a glance

All five models were tested once on the same 1,500 real work orders that none of them had seen. Scores run from 0 to 1, higher is better.

| model | quality score, priority (0-1) | quality score, team (0-1) | emergencies caught |
|---|---|---|---|
| **mwo-triage-13M (ours)** | **0.85** | **0.82** | **72%, or 83% in safety mode** |
| mwo-triage-5M (ours) | 0.83 | 0.80 | 64%, or 84% in safety mode |
| Google ELECTRA-small, as released | 0.34 | 0.12 | 0% |
| Google BERT tiny, as released | 0.46 | 0.43 | 11% |
| Microsoft DeBERTa-v3-base, trained by us as a coach (about 13x bigger) | 0.87 | 0.84 | 68% |

The quality score is macro-F1, which weights every label equally, so a model can't score well by only getting the common labels right.

### Headline table

| model | priority macro-F1 (bal.) | route macro-F1 (bal.) | STOP_NOW recall (bal. / safety) | STOP_NOW precision (bal. / safety) | size (MB) | CPU latency (ms/text) |
|---|---|---|---|---|---|---|
| **mwo-triage-13M (this model)** | **0.8486** | **0.8219** | **0.7207 / 0.8288** | **0.6897 / 0.5412** | **51.52** | **6.506** |
| ELECTRA-small frozen baseline | 0.3448 | 0.1192 | 0.0000 / — | 0.0000 / — | 51.52 | 6.545 |
| DeBERTa-v3-base teacher (reference) | 0.8675 | 0.8415 | 0.6757 / — | 0.7895 / — | 701.37 | 51.249 |

The frozen baseline uses the same base encoder, but only has a simple classification head. It was not fine-tuned. Its results show that full fine-tuning, which was the training approach chosen for this project, is important for making the model useful.

The teacher is a much larger reference model (184M+ parameters, 701.37 MB). This model was distilled from that teacher. The teacher model is not released.

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

The model includes a calibrated threshold for `STOP_NOW`. The threshold is t=0.1596. It was selected using validation data to target STOP_NOW recall ≥ 0.85.

You can choose between two modes:

| mode | priority macro-F1 | route macro-F1 | STOP_NOW recall | STOP_NOW precision |
|---|---|---|---|---|
| Balanced (argmax, calibrated probs) | 0.8486 | 0.8219 | 0.7207 | 0.6897 |
| Safety (STOP_NOW if P ≥ t=0.1596, else argmax) | 0.8312 | 0.8219 | 0.8288 | 0.5412 |

Safety mode finds more of the real `STOP_NOW` cases. However, it also produces more false `STOP_NOW` alarms. It does not change the route prediction.

### Calibration (ECE)

Temperature scaling was applied separately to the priority and route outputs. The values were fitted using validation data (priority T=1.9388, route T=1.9800).

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

The real training examples come from **MaintNorm** and **MaintIE**. These datasets contain maintenance text from mining and heavy mobile equipment.

We also added verified synthetic factory work orders to the training split. These examples cover areas such as CNC machines, presses, robots, conveyors, compressors and paint shops. They help the model cover more than mining and fleet maintenance language. The test set contains real text only.

The labels follow [`docs/labelling-rulebook.md`](https://github.com/karthekeyan/mwo-triage/blob/main/docs/labelling-rulebook.md). Two separate AI labelling passes labelled the same set of real texts. The models used were Claude Sonnet 5.5 (Pass A) and Claude Opus 5.5 (Pass B). Both followed the same rulebook and labelled the full pool of 11,497 real texts.

The two passes agreed on:

- 92.6% of priority labels (kappa 0.879).
- 92.9% of route labels (kappa 0.900).
- 86.6% of examples across both outputs together (kappa not reported for this combined figure). An example counts as an agreement only when both its priority and route labels match.

Where the two passes disagreed:

- **Test and validation pools:** A third pass reviewed every disagreement and applied the rulebook's tie-break rules. Claude Fable 5.2 (batch mode) decided 11 cases. Claude Code in-session (`claude-sonnet-5`) decided the remaining 300.
- **Training pool:** We kept Pass B's label for disagreements instead of removing those examples. Pass B had performed better than Pass A when compared with the adjudicated validation labels (val priority macro-F1 0.975 vs. 0.923 for Pass A; see [`docs/experiment-log.md`](https://github.com/karthekeyan/mwo-triage/blob/main/docs/experiment-log.md) sections 3-4).

We also applied a small number of documented rule overrides:

- **31 brake/steering cases** ([`scripts/apply_rulings.py`](https://github.com/karthekeyan/mwo-triage/blob/main/scripts/apply_rulings.py), `data/interim/rulings_overrides.jsonl`): These changes were made to records where Pass A and Pass B already agreed, across all pools. On mobile equipment, a work order that names a brake or steering fault as u/s, not working or failed was labelled `STOP_NOW`/`P1-SAFETY`. This rule does not apply to wording about warning or fault lights only, brake cooling-hose or circuit faults, steering-wheel comfort or adjustment features, or brake (signal) lights. Those examples keep their original labels.
- **Retarder and hook rulings:** These were applied separately during the in-session review of test and validation disagreements described above. A retarder reported as u/s or not working on mobile equipment, or a bent tub hook or one that needs straightening, is labelled `STOP_NOW`/`MECHANICAL`/`P1-SAFETY`.

**No human domain expert reviewed any of the labels.** Label quality depends on the rulebook, the agreement between the two AI labelling passes, the documented rule overrides above, and the limitations described here and in the rulebook.

### Synthetic data

Claude Code (`claude-sonnet-5`) created 2,489 synthetic factory work orders in-session. It followed a target label and style plan (`configs/synthetic_plan.yaml`).

A separate, fresh Claude Code session then labelled the synthetic texts without seeing their target labels. We kept the 1,814 examples (72.9%) where the new labels matched the original targets. Synthetic examples were used only for training.

### Final split sizes

- Train: 11,011 (9,197 real + 1,814 synthetic).
- Val: 800.
- Test: 1,500.

The validation and test sets contain real text only.

## 6. Training procedure

We fully fine-tuned `google/electra-small-discriminator`. We did not use LoRA or QLoRA. The model has two separate classification heads, one for priority and one for route, using a shared encoder.

We distilled the model from a separately fine-tuned `microsoft/deberta-v3-base` teacher. The teacher used two classification heads, a learning rate of 3e-5, a batch size of 32 and a maximum sequence length of 32.

Training settings:

- Encoder parameters: 13,483,008
- Learning rate: 1e-4, AdamW, 10% linear warmup followed by linear decay
- Batch size 64, maximum sequence length 32, seed 42, fp16
- Up to 25 epochs, with early stopping based on validation mean macro-F1 (patience 4); the best epoch was 11
- Distillation loss for each head: 0.5 × hard-label cross-entropy + 0.5 × KL(teacher ‖ student) at temperature 2, scaled by T²
- Validation data was used for model selection, calibration and threshold fitting. The test set was used exactly once, after all these choices were fixed.

## 7. Limitations

Please keep these limitations in mind when using the model:

- **`ESCALATE` is the weakest route.** Its test F1 is 0.6420 in balanced mode. This class includes vague text, text that could belong to several maintenance areas, and non-maintenance text. These cases are difficult to separate.
- **`INSTRUMENTATION_CONTROLS` is the second-weakest route.** Its F1 is 0.7714.
- **English only.**
- The real training and test data mostly contains language about mining and heavy mobile equipment (MaintNorm and MaintIE). Synthetic factory data was added to improve coverage of other industrial areas. Results in other industries may still be weaker than the headline scores, especially when the wording is different from both data sources.
- **A certified maintenance engineer did not review any labels.** The rulebook says it reflects general industrial practice. It is not a safety standard and must not replace a site's own safety procedures.
- **Safety mode gives up some precision to improve recall.** STOP_NOW precision falls from 0.6897 to 0.5412 when recall increases from 0.7207 to 0.8288. This means safety mode produces more false STOP_NOW alarms in exchange for missing fewer real STOP_NOW cases.
- Results are somewhat weaker for longer texts. Route macro-F1 is 0.8205 for 5-7 words and 0.7935 for ≥8 words. The ≥8-word group has a small test support (n=86).
- Route results are weaker on `maintie` texts than on `maintnorm` texts (0.7760 vs. 0.8317 macro-F1).
- `STOP_NOW` is the most difficult priority class, even in balanced mode. Its F1 is 0.7048, compared with 0.87-0.92 for the other three priority classes.

## 8. How to use

Install the required packages:

```bash
pip install torch transformers safetensors huggingface_hub
```

Download the inference file and load the model:

```python
from huggingface_hub import hf_hub_download
import importlib.util

inference_path = hf_hub_download(repo_id="karthe05/mwo-triage-13M", filename="inference.py")
spec = importlib.util.spec_from_file_location("inference", inference_path)
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)
TriageModel = inference.TriageModel

model = TriageModel.from_pretrained("karthe05/mwo-triage-13M")
texts = ["Hyd hose burst on press 3, oil spraying across walkway"]
balanced = model.predict(texts, mode="balanced")
safety = model.predict(texts, mode="safety")
```

The `predict` method returns one dictionary for each input text. Each dictionary contains:

- `priority` and `route`: the predicted labels.
- `priority_confidence` and `route_confidence`: the calibrated probability of the top prediction from each classification head.
- `mode`: the mode you requested, either `"balanced"` or `"safety"`.

In `"safety"` mode, the model returns `STOP_NOW` whenever its calibrated probability is at or above this checkpoint's fitted threshold (t=0.1596). This applies even if another priority class has a higher raw probability. The route prediction and its confidence do not change between modes.

## 9. Credits and citations

The real training and test text comes from these datasets:

- **MaintNorm** — Bikaun, Hodkiewicz and Liu, WNUT 2024. MIT licence. NLP-TLP Group, University of Western Australia. Repo: https://github.com/nlp-tlp/maintnorm . Paper: https://aclanthology.org/2024.wnut-1.7
- **MaintIE** — Bikaun et al., LREC-COLING 2024. MIT licence. NLP-TLP Group, University of Western Australia. Repo: https://github.com/nlp-tlp/maintie . Paper: https://aclanthology.org/2024.lrec-main.954

Please give credit to MaintNorm and MaintIE, and keep their licence notices if you use this model or the original data.

Base encoder: **Google ELECTRA** (`google/electra-small-discriminator`, https://huggingface.co/google/electra-small-discriminator), Apache-2.0 licence.

## 10. Author

CobuildX AI
