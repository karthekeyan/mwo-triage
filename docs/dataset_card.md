---
license: mit
language: en
task_categories:
- text-classification
size_categories:
- 10K<n<100K
tags:
- maintenance
- manufacturing
- work-orders
- triage
- small-language-model
---

# mwo-triage-data

## 1. Summary

`mwo-triage-data` is the training, validation and test data behind the `mwo-triage` models (`mwo-triage-13M`, `mwo-triage-5M`): short maintenance work-order texts, each labelled with a **priority** (`STOP_NOW`, `TODAY`, `PLAN`, `MONITOR`) and a **route** (`MECHANICAL`, `ELECTRICAL`, `HYDRAULIC_PNEUMATIC`, `INSTRUMENTATION_CONTROLS`, `LUBRICATION`, `ESCALATE`). It combines real technician-written work orders from two MIT-licensed academic datasets (**MaintNorm** and **MaintIE**) with synthetic factory work orders written to broaden coverage beyond mining-fleet language. Every label was produced by an AI labelling process following a fixed rulebook; **no label was reviewed by a certified maintenance engineer**.

## 2. Dataset structure

Each split is a JSON-Lines file (`train.jsonl`, `val.jsonl`, `test.jsonl`), one work order per line, with these fields:

| field | type | description |
|---|---|---|
| `uid` | string | unique id, e.g. `real-02234` or `syn-00415` |
| `text` | string | the work-order text the model sees |
| `priority` | string | one of `STOP_NOW`, `TODAY`, `PLAN`, `MONITOR` |
| `route` | string | one of `MECHANICAL`, `ELECTRICAL`, `HYDRAULIC_PNEUMATIC`, `INSTRUMENTATION_CONTROLS`, `LUBRICATION`, `ESCALATE` |
| `priority_id` | int | integer id for `priority` (see mapping below) |
| `route_id` | int | integer id for `route` (see mapping below) |
| `source` | string | `real` or `synthetic` |
| `label_source` | string | how the label was arrived at (see section 3): `agree`, `passB-default`, `adjudicated:<adjudicator>`, any of those `+override`, or `blind-verified` for synthetic data |

Label id mapping ([`configs/labels.yaml`](https://github.com/karthekeyan/mwo-triage/blob/main/configs/labels.yaml), v1, fixed — do not reorder or renumber against a trained checkpoint):

| priority | id | | route | id |
|---|---|---|---|---|
| STOP_NOW | 0 | | MECHANICAL | 0 |
| TODAY | 1 | | ELECTRICAL | 1 |
| PLAN | 2 | | HYDRAULIC_PNEUMATIC | 2 |
| MONITOR | 3 | | INSTRUMENTATION_CONTROLS | 3 |
| | | | LUBRICATION | 4 |
| | | | ESCALATE | 5 |

### Splits and label distributions

From `data/processed/stats.json`.

| split | n | real | synthetic |
|---|---|---|---|
| train | 11,011 | 9,197 | 1,814 |
| val | 800 | 800 | 0 |
| test | 1,500 | 1,500 | 0 |

Only `train` contains synthetic text; `val` and `test` are real text that was never seen in training or any val-only tuning step.

Priority distribution:

| split | STOP_NOW | TODAY | PLAN | MONITOR |
|---|---|---|---|---|
| train | 1,314 | 4,200 | 4,486 | 1,011 |
| val | 68 | 307 | 370 | 55 |
| test | 111 | 582 | 705 | 102 |

Route distribution:

| split | MECHANICAL | ELECTRICAL | HYDRAULIC_PNEUMATIC | INSTRUMENTATION_CONTROLS | LUBRICATION | ESCALATE |
|---|---|---|---|---|---|---|
| train | 4,724 | 1,672 | 1,425 | 990 | 875 | 1,325 |
| val | 390 | 116 | 108 | 56 | 43 | 87 |
| test | 702 | 238 | 211 | 105 | 76 | 168 |

Mean word count: train 5.66, val 5.41, test 5.34.

## 3. How it was built

1. **Extraction** ([`scripts/extract_real_texts.py`](https://github.com/karthekeyan/mwo-triage/blob/main/scripts/extract_real_texts.py)): real work-order texts were read from **MaintNorm** and **MaintIE** (`data/raw/`). Asset IDs in the source text were replaced by made-up codes of the same letter/digit shape, and any `<sensitive>` token was masked to `[name]`; real IDs are never restored or guessed.
2. **De-duplication, cleaning and group-aware split** ([`scripts/clean_and_split.py`](https://github.com/karthekeyan/mwo-triage/blob/main/scripts/clean_and_split.py)): exact duplicates were removed at extraction. Remaining texts were dropped if too long (>40 words) or too sparse in alphabetic content (fewer than 2 non-numeric alphabetic words). Near-duplicates (token-set Jaccard ≥ 0.85, ignoring numeric tokens) were grouped, and a group is never split across pools — the whole group goes to exactly one of the test, val or train real-text pools (test 1,500, val 800, the rest to train), so no near-duplicate leaks across splits.
3. **Labelling rulebook v1.3** ([`docs/labelling-rulebook.md`](https://github.com/karthekeyan/mwo-triage/blob/main/docs/labelling-rulebook.md)): the only source of label definitions; priority checked in order `STOP_NOW → TODAY → PLAN → MONITOR` (first match wins), route by component/system rather than machine name or symptom.
4. **Two independent AI labelling passes** over the complete real-text pool (11,497 texts): Claude Sonnet 5.5 (Pass A) and Claude Opus 5.5 (Pass B), both reading the same rulebook. Agreement was 92.6% on priority (kappa 0.879), 92.9% on route (kappa 0.900), and 86.6% on both heads together.
5. **Adjudication of disagreements**:
   - **Test and validation pools**: every disagreement was adjudicated by a third pass applying the rulebook's tie-break rules. 11 were decided by Claude Fable 5.2 (batch mode); the remaining 300 by Claude Code in-session (`claude-sonnet-5`).
   - **Training pool**: disagreements (1,225 of them) take Pass B's label rather than being dropped, after Pass B was found to be the more accurate labeller against the adjudicated validation labels (val priority macro-F1 0.975 vs. 0.923 for Pass A).
6. **Documented rule overrides**:
   - **31 brake/steering cases** ([`scripts/apply_rulings.py`](https://github.com/karthekeyan/mwo-triage/blob/main/scripts/apply_rulings.py), `data/interim/rulings_overrides.jsonl`): applied to records where Pass A and Pass B already agreed, across all pools — a brake or steering fault, u/s, not working or failed on mobile equipment is `STOP_NOW`/`P1-SAFETY`, excluding warning/fault-light-only wording, brake cooling-hose/circuit faults, steering-wheel comfort/adjustment features, and brake (signal) lights.
   - **Retarder and hook rulings**: applied separately, during the in-session adjudication of test/val disagreements (step 5), as part of its consistency pass — a retarder reported as u/s or not working on mobile equipment, or a bent or in-need-of-straightening tub hook, is `STOP_NOW`/`MECHANICAL`/`P1-SAFETY`.
7. **Synthetic generation**: 2,489 synthetic factory work orders (CNC, presses, robots, conveyors, compressors, paint shops, etc.) were written in-session by Claude Code (`claude-sonnet-5`) to a target label and style plan ([`configs/synthetic_plan.yaml`](https://github.com/karthekeyan/mwo-triage/blob/main/configs/synthetic_plan.yaml)). A separate, fresh Claude Code session then relabelled them blind from the text alone, and the 1,814 (72.9%) where the blind labels matched the targets were kept. Synthetic examples are train-only.
8. **Leakage checks** ([`scripts/build_dataset.py`](https://github.com/karthekeyan/mwo-triage/blob/main/scripts/build_dataset.py)): no uid appears in more than one split, and no kept synthetic train text is an exact or near-duplicate (token Jaccard ≥ 0.85) of any val or test text — 0 synthetic texts were dropped for this reason, i.e. the check passed cleanly with nothing to remove.

## 4. Intended uses and limitations

Intended for training or evaluating short-text maintenance triage classifiers (priority and route), for research and for the `mwo-triage` model family specifically. Not a safety standard and not a substitute for a site's own safety procedures.

- **Mostly mining heavy-mobile-equipment language.** The real text (MaintNorm, MaintIE) skews toward that domain; the synthetic factory text broadens but does not fully correct this.
- **Labels were produced entirely by AI** following a fixed rulebook (two independent passes, agreement scoring, adjudication, and the documented rule overrides above). **No label was reviewed by a certified maintenance engineer.**
- **Synthetic text is train-only** and was itself only kept where a second, independent AI labelling pass agreed blind with the generation target (72.9% of what was generated).
- **English only.**

## 5. Licence

The real text in this dataset is derived from **MaintNorm** and **MaintIE**, both released under the MIT licence by the NLP-TLP Group, University of Western Australia. This dataset — including the derived/anonymised real text, the rulebook-based labels, and the synthetic text and labels — is released under the **MIT licence**. If you use this dataset, you must credit MaintNorm and MaintIE and retain their licence notices (reproduced in full below), in addition to this dataset's own licence.

## 6. Citations

- **MaintNorm** — Bikaun, Hodkiewicz and Liu, WNUT 2024. MIT licence. NLP-TLP Group, University of Western Australia. Repo: https://github.com/nlp-tlp/maintnorm . Paper: https://aclanthology.org/2024.wnut-1.7
- **MaintIE** — Bikaun et al., LREC-COLING 2024. MIT licence. NLP-TLP Group, University of Western Australia. Repo: https://github.com/nlp-tlp/maintie . Paper: https://aclanthology.org/2024.lrec-main.954

## 7. Source licence notices (reproduced in full)

### MaintNorm — `LICENSE.md`

```
MIT License

Copyright (c) [2024] [NLP-TLP Group]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MaintIE — `LICENSE.md`

```
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The MaintIE licence file leaves its copyright holder as a template placeholder; following the project README and paper, we attribute it to the NLP-TLP Group, University of Western Australia.

## 8. Author

karthe05
