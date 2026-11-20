# Final evaluation on the held-out test set

data/processed/test.jsonl (1,500 real texts, never seen in training or any val-only tuning step) was loaded exactly once, by `scripts/evaluate.py`, after step (1) retrained the frozen-encoder baselines and step (2) fit per-head temperature scaling and the STOP_NOW safety threshold -- both on val only. See `docs/experiment-log.md` section 6 for the final model picks.

## Headline table

| model | priority F1 (bal.) | route F1 (bal.) | STOP_NOW recall (bal. / safety) | STOP_NOW precision (bal. / safety) | size (MB) | CPU latency (ms/text) |
|---|---|---|---|---|---|---|
| mwo-triage-13M (Small, final) | 0.8486 | 0.8219 | 0.7207 / 0.8288 | 0.6897 / 0.5412 | 51.52 | 6.506 |
| mwo-triage-5M (Tiny, final) | 0.8265 | 0.8008 | 0.6396 / 0.8378 | 0.6339 / 0.4650 | 18.28 | 1.799 |
| ELECTRA-small frozen baseline | 0.3448 | 0.1192 | 0.0000 / -- | 0.0000 / -- | 51.52 | 6.545 |
| BERT L-4/H-128 frozen baseline | 0.4648 | 0.4255 | 0.1081 / -- | 0.2667 / -- | 18.28 | 1.799 |
| DeBERTa-v3-base teacher (reference) | 0.8675 | 0.8415 | 0.6757 / -- | 0.7895 / -- | 701.37 | 51.249 |

## mwo-triage-13M (Small, final) (`electra_distilled`)

- model id: `google/electra-small-discriminator`
- test set size: 1500
- model size on disk: 51.52 MB
- mean CPU latency (batch size 1, torch CPU, 200 texts): 6.506 ms/text
- calibration: priority T=1.9388, route T=1.9800, STOP_NOW threshold t=0.1596 (fit on val for recall >= 0.85)
- ECE priority before/after calibration: 0.0887 / 0.0242
- ECE route before/after calibration: 0.1095 / 0.0367

### Balanced mode (balanced (argmax, calibrated probs))

- priority macro-F1: 0.8486, accuracy: 0.8860
- route macro-F1: 0.8219, accuracy: 0.8580
- STOP_NOW recall: 0.7207, precision: 0.6897

Priority per-class:

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| STOP_NOW | 0.6897 | 0.7207 | 0.7048 | 111 |
| TODAY | 0.8769 | 0.8814 | 0.8792 | 582 |
| PLAN | 0.9230 | 0.9177 | 0.9203 | 705 |
| MONITOR | 0.9082 | 0.8725 | 0.8900 | 102 |

Priority confusion matrix (rows=actual, cols=predicted, order=STOP_NOW, TODAY, PLAN, MONITOR):
```
STOP_NOW   [80, 22, 9, 0]
TODAY      [28, 513, 38, 3]
PLAN       [7, 45, 647, 6]
MONITOR    [1, 5, 7, 89]
```

Route per-class:

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| MECHANICAL | 0.9003 | 0.9259 | 0.9129 | 702 |
| ELECTRICAL | 0.8826 | 0.8529 | 0.8675 | 238 |
| HYDRAULIC_PNEUMATIC | 0.8565 | 0.8768 | 0.8665 | 211 |
| INSTRUMENTATION_CONTROLS | 0.7714 | 0.7714 | 0.7714 | 105 |
| LUBRICATION | 0.9014 | 0.8421 | 0.8707 | 76 |
| ESCALATE | 0.6667 | 0.6190 | 0.6420 | 168 |

Route confusion matrix (rows=actual, cols=predicted, order=MECHANICAL, ELECTRICAL, HYDRAULIC_PNEUMATIC, INSTRUMENTATION_CONTROLS, LUBRICATION, ESCALATE):
```
MECHANICAL                [650, 5, 18, 4, 2, 23]
ELECTRICAL                [8, 203, 1, 12, 1, 13]
HYDRAULIC_PNEUMATIC       [16, 0, 185, 3, 1, 6]
INSTRUMENTATION_CONTROLS  [2, 11, 2, 81, 0, 9]
LUBRICATION               [6, 1, 4, 0, 64, 1]
ESCALATE                  [40, 10, 6, 5, 3, 104]
```

### Safety mode (safety (STOP_NOW if P>=t=0.1596, else argmax))

- priority macro-F1: 0.8312, accuracy: 0.8693
- STOP_NOW recall: 0.8288, precision: 0.5412

Priority per-class:

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| STOP_NOW | 0.5412 | 0.8288 | 0.6548 | 111 |
| TODAY | 0.8872 | 0.8247 | 0.8549 | 582 |
| PLAN | 0.9292 | 0.9121 | 0.9205 | 705 |
| MONITOR | 0.9175 | 0.8725 | 0.8945 | 102 |

Priority confusion matrix (rows=actual, cols=predicted, order=STOP_NOW, TODAY, PLAN, MONITOR):
```
STOP_NOW   [92, 13, 6, 0]
TODAY      [64, 480, 36, 2]
PLAN       [13, 43, 643, 6]
MONITOR    [1, 5, 7, 89]
```

Macro-F1 by text length:

| bucket | n | priority macro-F1 | route macro-F1 |
|---|---|---|---|
| <=4 words | 406 | 0.8566 | 0.8076 |
| 5-7 words | 1008 | 0.8512 | 0.8205 |
| >=8 words | 86 | 0.8286 | 0.7935 |

Macro-F1 by real-text source:

| source | n | priority macro-F1 | route macro-F1 |
|---|---|---|---|
| maintie | 343 | 0.8364 | 0.7760 |
| maintnorm | 1157 | 0.8520 | 0.8317 |

## mwo-triage-5M (Tiny, final) (`L4_a_earlystop_passb`)

- model id: `google/bert_uncased_L-4_H-128_A-2`
- test set size: 1500
- model size on disk: 18.28 MB
- mean CPU latency (batch size 1, torch CPU, 200 texts): 1.799 ms/text
- calibration: priority T=1.4289, route T=1.3630, STOP_NOW threshold t=0.2009 (fit on val for recall >= 0.85)
- ECE priority before/after calibration: 0.0513 / 0.0101
- ECE route before/after calibration: 0.0700 / 0.0202

### Balanced mode (balanced (argmax, calibrated probs))

- priority macro-F1: 0.8265, accuracy: 0.8753
- route macro-F1: 0.8008, accuracy: 0.8467
- STOP_NOW recall: 0.6396, precision: 0.6339

Priority per-class:

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| STOP_NOW | 0.6339 | 0.6396 | 0.6368 | 111 |
| TODAY | 0.8690 | 0.8660 | 0.8675 | 582 |
| PLAN | 0.9190 | 0.9177 | 0.9184 | 705 |
| MONITOR | 0.8750 | 0.8922 | 0.8835 | 102 |

Priority confusion matrix (rows=actual, cols=predicted, order=STOP_NOW, TODAY, PLAN, MONITOR):
```
STOP_NOW   [71, 30, 9, 1]
TODAY      [33, 504, 41, 4]
PLAN       [8, 42, 647, 8]
MONITOR    [0, 4, 7, 91]
```

Route per-class:

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| MECHANICAL | 0.8886 | 0.9316 | 0.9096 | 702 |
| ELECTRICAL | 0.8675 | 0.8529 | 0.8602 | 238 |
| HYDRAULIC_PNEUMATIC | 0.8708 | 0.8626 | 0.8667 | 211 |
| INSTRUMENTATION_CONTROLS | 0.7476 | 0.7333 | 0.7404 | 105 |
| LUBRICATION | 0.8841 | 0.8026 | 0.8414 | 76 |
| ESCALATE | 0.6242 | 0.5536 | 0.5868 | 168 |

Route confusion matrix (rows=actual, cols=predicted, order=MECHANICAL, ELECTRICAL, HYDRAULIC_PNEUMATIC, INSTRUMENTATION_CONTROLS, LUBRICATION, ESCALATE):
```
MECHANICAL                [654, 5, 12, 3, 2, 26]
ELECTRICAL                [5, 203, 2, 12, 1, 15]
HYDRAULIC_PNEUMATIC       [17, 2, 182, 4, 2, 4]
INSTRUMENTATION_CONTROLS  [10, 8, 0, 77, 0, 10]
LUBRICATION               [10, 0, 4, 0, 61, 1]
ESCALATE                  [40, 16, 9, 7, 3, 93]
```

### Safety mode (safety (STOP_NOW if P>=t=0.2009, else argmax))

- priority macro-F1: 0.8088, accuracy: 0.8520
- STOP_NOW recall: 0.8378, precision: 0.4650

Priority per-class:

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| STOP_NOW | 0.4650 | 0.8378 | 0.5981 | 111 |
| TODAY | 0.8946 | 0.7732 | 0.8295 | 582 |
| PLAN | 0.9267 | 0.9149 | 0.9208 | 705 |
| MONITOR | 0.8911 | 0.8824 | 0.8867 | 102 |

Priority confusion matrix (rows=actual, cols=predicted, order=STOP_NOW, TODAY, PLAN, MONITOR):
```
STOP_NOW   [93, 13, 4, 1]
TODAY      [88, 450, 41, 3]
PLAN       [17, 36, 645, 7]
MONITOR    [2, 4, 6, 90]
```

Macro-F1 by text length:

| bucket | n | priority macro-F1 | route macro-F1 |
|---|---|---|---|
| <=4 words | 406 | 0.7942 | 0.7701 |
| 5-7 words | 1008 | 0.8373 | 0.8042 |
| >=8 words | 86 | 0.8346 | 0.8261 |

Macro-F1 by real-text source:

| source | n | priority macro-F1 | route macro-F1 |
|---|---|---|---|
| maintie | 343 | 0.8302 | 0.7643 |
| maintnorm | 1157 | 0.8248 | 0.8076 |

## ELECTRA-small frozen baseline (`electra_frozen_final`)

- model id: `google/electra-small-discriminator`
- test set size: 1500
- model size on disk: 51.52 MB
- mean CPU latency (batch size 1, torch CPU, 200 texts): 6.545 ms/text
- calibration: not fit for this model (frozen baseline / teacher reference)
- ECE priority (no calibration fit): 0.1067
- ECE route (no calibration fit): 0.0421

### Balanced mode (balanced (argmax, raw softmax))

- priority macro-F1: 0.3448, accuracy: 0.6427
- route macro-F1: 0.1192, accuracy: 0.4700
- STOP_NOW recall: 0.0000, precision: 0.0000

Priority per-class:

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| STOP_NOW | 0.0000 | 0.0000 | 0.0000 | 111 |
| TODAY | 0.6195 | 0.7216 | 0.6667 | 582 |
| PLAN | 0.6618 | 0.7716 | 0.7125 | 705 |
| MONITOR | 0.0000 | 0.0000 | 0.0000 | 102 |

Priority confusion matrix (rows=actual, cols=predicted, order=STOP_NOW, TODAY, PLAN, MONITOR):
```
STOP_NOW   [0, 77, 34, 0]
TODAY      [0, 420, 162, 0]
PLAN       [0, 161, 544, 0]
MONITOR    [0, 20, 82, 0]
```

Route per-class:

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| MECHANICAL | 0.4709 | 0.9929 | 0.6389 | 702 |
| ELECTRICAL | 0.2000 | 0.0042 | 0.0082 | 238 |
| HYDRAULIC_PNEUMATIC | 0.5000 | 0.0237 | 0.0452 | 211 |
| INSTRUMENTATION_CONTROLS | 0.0000 | 0.0000 | 0.0000 | 105 |
| LUBRICATION | 0.0000 | 0.0000 | 0.0000 | 76 |
| ESCALATE | 0.4000 | 0.0119 | 0.0231 | 168 |

Route confusion matrix (rows=actual, cols=predicted, order=MECHANICAL, ELECTRICAL, HYDRAULIC_PNEUMATIC, INSTRUMENTATION_CONTROLS, LUBRICATION, ESCALATE):
```
MECHANICAL                [697, 0, 3, 0, 0, 2]
ELECTRICAL                [235, 1, 1, 0, 0, 1]
HYDRAULIC_PNEUMATIC       [205, 1, 5, 0, 0, 0]
INSTRUMENTATION_CONTROLS  [103, 1, 1, 0, 0, 0]
LUBRICATION               [75, 1, 0, 0, 0, 0]
ESCALATE                  [165, 1, 0, 0, 0, 2]
```

Macro-F1 by text length:

| bucket | n | priority macro-F1 | route macro-F1 |
|---|---|---|---|
| <=4 words | 406 | 0.3520 | 0.1031 |
| 5-7 words | 1008 | 0.3443 | 0.1238 |
| >=8 words | 86 | 0.3109 | 0.1210 |

Macro-F1 by real-text source:

| source | n | priority macro-F1 | route macro-F1 |
|---|---|---|---|
| maintie | 343 | 0.3562 | 0.1209 |
| maintnorm | 1157 | 0.3413 | 0.1169 |

## BERT L-4/H-128 frozen baseline (`L4_frozen_final`)

- model id: `google/bert_uncased_L-4_H-128_A-2`
- test set size: 1500
- model size on disk: 18.28 MB
- mean CPU latency (batch size 1, torch CPU, 200 texts): 1.799 ms/text
- calibration: not fit for this model (frozen baseline / teacher reference)
- ECE priority (no calibration fit): 0.0927
- ECE route (no calibration fit): 0.0407

### Balanced mode (balanced (argmax, raw softmax))

- priority macro-F1: 0.4648, accuracy: 0.6940
- route macro-F1: 0.4255, accuracy: 0.5907
- STOP_NOW recall: 0.1081, precision: 0.2667

Priority per-class:

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| STOP_NOW | 0.2667 | 0.1081 | 0.1538 | 111 |
| TODAY | 0.7041 | 0.7113 | 0.7077 | 582 |
| PLAN | 0.7229 | 0.8511 | 0.7818 | 705 |
| MONITOR | 0.4054 | 0.1471 | 0.2158 | 102 |

Priority confusion matrix (rows=actual, cols=predicted, order=STOP_NOW, TODAY, PLAN, MONITOR):
```
STOP_NOW   [12, 70, 28, 1]
TODAY      [24, 414, 133, 11]
PLAN       [9, 86, 600, 10]
MONITOR    [0, 18, 69, 15]
```

Route per-class:

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| MECHANICAL | 0.6006 | 0.9060 | 0.7223 | 702 |
| ELECTRICAL | 0.6907 | 0.2815 | 0.4000 | 238 |
| HYDRAULIC_PNEUMATIC | 0.5930 | 0.4834 | 0.5326 | 211 |
| INSTRUMENTATION_CONTROLS | 0.3944 | 0.2667 | 0.3182 | 105 |
| LUBRICATION | 0.4333 | 0.1711 | 0.2453 | 76 |
| ESCALATE | 0.5634 | 0.2381 | 0.3347 | 168 |

Route confusion matrix (rows=actual, cols=predicted, order=MECHANICAL, ELECTRICAL, HYDRAULIC_PNEUMATIC, INSTRUMENTATION_CONTROLS, LUBRICATION, ESCALATE):
```
MECHANICAL                [636, 10, 30, 11, 8, 7]
ELECTRICAL                [129, 67, 9, 18, 1, 14]
HYDRAULIC_PNEUMATIC       [99, 2, 102, 5, 1, 2]
INSTRUMENTATION_CONTROLS  [49, 10, 9, 28, 1, 8]
LUBRICATION               [51, 2, 10, 0, 13, 0]
ESCALATE                  [95, 6, 12, 9, 6, 40]
```

Macro-F1 by text length:

| bucket | n | priority macro-F1 | route macro-F1 |
|---|---|---|---|
| <=4 words | 406 | 0.4831 | 0.3837 |
| 5-7 words | 1008 | 0.4782 | 0.4244 |
| >=8 words | 86 | 0.3424 | 0.6293 |

Macro-F1 by real-text source:

| source | n | priority macro-F1 | route macro-F1 |
|---|---|---|---|
| maintie | 343 | 0.5002 | 0.4549 |
| maintnorm | 1157 | 0.4536 | 0.4178 |

## DeBERTa-v3-base teacher (reference) (`teacher_deberta`)

- model id: `microsoft/deberta-v3-base`
- test set size: 1500
- model size on disk: 701.37 MB
- mean CPU latency (batch size 1, torch CPU, 200 texts): 51.249 ms/text
- calibration: not fit for this model (frozen baseline / teacher reference)
- ECE priority (no calibration fit): 0.0684
- ECE route (no calibration fit): 0.0893

### Balanced mode (balanced (argmax, raw softmax))

- priority macro-F1: 0.8675, accuracy: 0.9033
- route macro-F1: 0.8415, accuracy: 0.8747
- STOP_NOW recall: 0.6757, precision: 0.7895

Priority per-class:

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| STOP_NOW | 0.7895 | 0.6757 | 0.7282 | 111 |
| TODAY | 0.8795 | 0.9158 | 0.8973 | 582 |
| PLAN | 0.9383 | 0.9277 | 0.9330 | 705 |
| MONITOR | 0.9118 | 0.9118 | 0.9118 | 102 |

Priority confusion matrix (rows=actual, cols=predicted, order=STOP_NOW, TODAY, PLAN, MONITOR):
```
STOP_NOW   [75, 28, 7, 1]
TODAY      [17, 533, 30, 2]
PLAN       [3, 42, 654, 6]
MONITOR    [0, 3, 6, 93]
```

Route per-class:

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| MECHANICAL | 0.9144 | 0.9430 | 0.9285 | 702 |
| ELECTRICAL | 0.8987 | 0.8571 | 0.8774 | 238 |
| HYDRAULIC_PNEUMATIC | 0.8889 | 0.8720 | 0.8804 | 211 |
| INSTRUMENTATION_CONTROLS | 0.8000 | 0.7619 | 0.7805 | 105 |
| LUBRICATION | 0.9296 | 0.8684 | 0.8980 | 76 |
| ESCALATE | 0.6784 | 0.6905 | 0.6844 | 168 |

Route confusion matrix (rows=actual, cols=predicted, order=MECHANICAL, ELECTRICAL, HYDRAULIC_PNEUMATIC, INSTRUMENTATION_CONTROLS, LUBRICATION, ESCALATE):
```
MECHANICAL                [662, 3, 13, 5, 1, 18]
ELECTRICAL                [6, 204, 2, 8, 1, 17]
HYDRAULIC_PNEUMATIC       [17, 1, 184, 1, 1, 7]
INSTRUMENTATION_CONTROLS  [2, 10, 1, 80, 0, 12]
LUBRICATION               [6, 1, 2, 0, 66, 1]
ESCALATE                  [31, 8, 5, 6, 2, 116]
```

Macro-F1 by text length:

| bucket | n | priority macro-F1 | route macro-F1 |
|---|---|---|---|
| <=4 words | 406 | 0.8750 | 0.8623 |
| 5-7 words | 1008 | 0.8731 | 0.8318 |
| >=8 words | 86 | 0.8003 | 0.7957 |

Macro-F1 by real-text source:

| source | n | priority macro-F1 | route macro-F1 |
|---|---|---|---|
| maintie | 343 | 0.8646 | 0.8500 |
| maintnorm | 1157 | 0.8692 | 0.8382 |

