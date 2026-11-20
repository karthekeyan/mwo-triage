# mwo-triage experiment log

## 1. Base-model bake-off (2026-11-20)

Six Apache-2.0 candidate encoders (`scripts/survey_base_models.py` survey:
see commit history) were each fine-tuned at two learning rates plus a
frozen-encoder (heads-only) baseline, on `data/processed/train.jsonl`
(real train pool + blind-verified synthetic), evaluated each epoch on
`data/processed/val.jsonl`. Up to 10 epochs, AdamW with 10% linear
warmup then linear decay, batch size 64, max length 32, seed 42, fp16 on
the Titan RTX. `data/processed/test.jsonl` was never touched.
`scripts/bakeoff.sh` ran all 18 runs; full log in
`outputs/bakeoff/bakeoff.log` (not committed; see CLAUDE.md data rules).

The two H-128 ("Tiny" candidates) models were tried at lr 1e-4 and 3e-4;
the three H-256 ("Small" candidates) models at lr 5e-5 and 1e-4. The
frozen baseline used lr 1e-3 (heads only) for all six.

18 runs, 933s total.

| model_id | encoder params | mode | lr | best epoch | val priority F1 | val route F1 | val mean F1 | STOP_NOW recall | train time (s) |
|---|---|---|---|---|---|---|---|---|---|
| google/bert_uncased_L-2_H-128_A-2 | 4,385,920 | frozen | 1e-3 | 10 | 0.4007 | 0.3610 | 0.3808 | 0.0147 | 26.3 |
| google/bert_uncased_L-2_H-128_A-2 | 4,385,920 | finetune | 1e-4 | 9 | 0.7638 | 0.7850 | 0.7744 | 0.2941 | 33.9 |
| google/bert_uncased_L-2_H-128_A-2 | 4,385,920 | finetune | 3e-4 | 10 | 0.8162 | 0.8340 | 0.8251 | 0.5735 | 33.8 |
| google/bert_uncased_L-4_H-128_A-2 | 4,782,464 | frozen | 1e-3 | 10 | 0.3956 | 0.4304 | 0.4130 | 0.0294 | 30.2 |
| google/bert_uncased_L-4_H-128_A-2 | 4,782,464 | finetune | 1e-4 | 9 | 0.7903 | 0.7963 | 0.7933 | 0.4265 | 40.2 |
| **google/bert_uncased_L-4_H-128_A-2** | **4,782,464** | **finetune** | **3e-4** | **7** | **0.8218** | **0.8425** | **0.8321** | **0.6029** | **40.3** |
| google/bert_uncased_L-6_H-128_A-2 | 5,179,008 | frozen | 1e-3 | 10 | 0.3871 | 0.3232 | 0.3552 | 0.0147 | 32.2 |
| google/bert_uncased_L-6_H-128_A-2 | 5,179,008 | finetune | 1e-4 | 9 | 0.7801 | 0.7838 | 0.7819 | 0.3676 | 46.2 |
| google/bert_uncased_L-6_H-128_A-2 | 5,179,008 | finetune | 3e-4 | 8 | 0.8100 | 0.8425 | 0.8263 | 0.4706 | 45.9 |
| google/electra-small-discriminator | 13,483,008 | frozen | 1e-3 | 7 | 0.3376 | 0.1351 | 0.2363 | 0.0000 | 41.2 |
| **google/electra-small-discriminator** | **13,483,008** | **finetune** | **1e-4** | **9** | **0.8525** | **0.8441** | **0.8483** | **0.6471** | **67.6** |
| google/electra-small-discriminator | 13,483,008 | finetune | 5e-5 | 9 | 0.8158 | 0.8164 | 0.8161 | 0.4412 | 67.5 |
| google/bert_uncased_L-8_H-256_A-4 | 14,329,600 | frozen | 1e-3 | 7 | 0.5016 | 0.4407 | 0.4711 | 0.0441 | 35.3 |
| google/bert_uncased_L-8_H-256_A-4 | 14,329,600 | finetune | 1e-4 | 5 | 0.8344 | 0.8182 | 0.8263 | 0.6765 | 61.0 |
| google/bert_uncased_L-8_H-256_A-4 | 14,329,600 | finetune | 5e-5 | 9 | 0.8086 | 0.8281 | 0.8183 | 0.4853 | 61.1 |
| google/bert_uncased_L-12_H-256_A-4 | 17,488,640 | frozen | 1e-3 | 7 | 0.4733 | 0.4489 | 0.4611 | 0.0441 | 41.4 |
| google/bert_uncased_L-12_H-256_A-4 | 17,488,640 | finetune | 1e-4 | 9 | 0.8289 | 0.8457 | 0.8373 | 0.5000 | 74.4 |
| google/bert_uncased_L-12_H-256_A-4 | 17,488,640 | finetune | 5e-5 | 8 | 0.8362 | 0.8389 | 0.8375 | 0.6029 | 73.2 |

**Winners:**
- **Small**: `google/electra-small-discriminator`, lr 1e-4 (val mean macro-F1 0.8483, STOP_NOW recall 0.6471)
- **Tiny**: `google/bert_uncased_L-4_H-128_A-2`, lr 3e-4 (val mean macro-F1 0.8321, STOP_NOW recall 0.6029)

Fine-tuning beats frozen-encoder heads-only training by a wide margin on
every model (mean macro-F1 roughly doubles), confirming the locked
decision to full fine-tune rather than freeze-and-probe.

## 2. Improvement round on the two winners (2026-11-20)

Val only; `data/processed/test.jsonl` was never loaded. `scripts/improve.sh`
ran, in order: (a) same model/lr as the winning bake-off run but up to 25
epochs with early stopping (patience 4 on val mean macro-F1); (b) the
same as (a) plus class-weighted cross-entropy on the priority head
(weight proportional to 1/sqrt(train class frequency), normalised to mean 1; route loss
unweighted); then (c) `scripts/stop_now_threshold.py` picked whichever of
(a)/(b) had the higher val mean macro-F1 per model and searched val for
the highest (most selective) STOP_NOW probability threshold that still
reaches STOP_NOW recall >= 0.85, overriding the priority decision to
STOP_NOW whenever P(STOP_NOW) is at or above that threshold. 4 training
runs, 326s total; full log in `outputs/bakeoff/improve.log` (not
committed). The threshold and post-hoc metrics are saved under
`post_hoc_stop_now` in the chosen run's `metrics.json`.

### Small: google/electra-small-discriminator, lr 1e-4

| variant | priority F1 | route F1 | mean F1 | STOP_NOW recall | STOP_NOW precision |
|---|---|---|---|---|---|
| baseline (`electra_lr1e-4`) | 0.8525 | 0.8441 | 0.8483 | 0.6471 | 0.7097 |
| (a) early-stop (`electra_a_earlystop`) | 0.8415 | 0.8696 | 0.8556 | 0.6029 | 0.6721 |
| (b) class-weighted (`electra_b_classweighted`) | 0.8348 | 0.8542 | 0.8445 | 0.6912 | 0.6351 |
| (c) (a) + threshold (t=0.0126) | 0.8312 | 0.8696 | 0.8504 | **0.8529** | 0.5043 |

Priority confusion matrix at (c) (rows=actual, cols=predicted, order
STOP_NOW/TODAY/PLAN/MONITOR):

```
STOP_NOW   [58,  6,  4,  0]
TODAY      [43, 254,  9,  1]
PLAN       [13, 12, 340,  5]
MONITOR    [ 1,  0,  6, 48]
```

### Tiny: google/bert_uncased_L-4_H-128_A-2, lr 3e-4

| variant | priority F1 | route F1 | mean F1 | STOP_NOW recall | STOP_NOW precision |
|---|---|---|---|---|---|
| baseline (`L4_lr3e-4`) | 0.8218 | 0.8425 | 0.8321 | 0.6029 | 0.6721 |
| (a) early-stop (`L4_a_earlystop`) | 0.8316 | 0.8329 | 0.8322 | 0.5882 | 0.7143 |
| (b) class-weighted (`L4_b_classweighted`) | 0.8185 | 0.8351 | 0.8268 | 0.6176 | 0.6885 |
| (c) (a) + threshold (t=0.0432) | 0.8219 | 0.8329 | 0.8274 | **0.8529** | 0.5088 |

Priority confusion matrix at (c):

```
STOP_NOW   [58,  5,  5,  0]
TODAY      [46, 245, 14,  2]
PLAN       [ 9, 13, 343,  5]
MONITOR    [ 1,  0,  7, 47]
```

**Takeaways**: (a) early-stopping and (b) class-weighting are both small,
inconsistent nudges over the baseline (no clear winner between the two
for either model). The post-hoc STOP_NOW threshold is the real lever:
pushing STOP_NOW recall from ~0.60 to the 0.85 floor costs roughly
15-20 points of STOP_NOW precision and a small priority macro-F1 dip --
the expected safety/false-alarm tradeoff of a "never miss STOP_NOW"
decision rule, and the one worth carrying into the final model choice
if minimising missed STOP_NOW cases matters more than alarm fatigue.

## 3. Pass A / Pass B vs. the models, on val (2026-11-20)

Quick analysis, no training, val only. For each uid in `data/processed/val.jsonl`,
looked up that uid's raw (pre-adjudication) label in `data/interim/full_pass_a.jsonl`
(Pass A, Claude Sonnet 5.5) and `data/interim/full_pass_b.jsonl` (Pass B, Claude
Opus 5.5), and scored each pass's own priority label against the final (adjudicated)
val label used as the reference. All 800 val uids matched in both pass files.

| labeller | n | priority macro-F1 | STOP_NOW recall | STOP_NOW precision |
|---|---|---|---|---|
| Pass A (Sonnet 5.5) | 800 | 0.9228 | 0.6912 | 1.0000 |
| Pass B (Opus 5.5) | 800 | 0.9750 | 0.9265 | 0.9545 |
| ELECTRA-small (a) | 800 | 0.8415 | 0.6029 | 0.6721 |
| BERT L-4/H-128 (a) | 800 | 0.8316 | 0.5882 | 0.7143 |

Note on reading this: val's reference label is itself adjudicated between
Pass A and Pass B when they disagreed, so each pass's STOP_NOW precision
against val is inflated relative to a true external ground truth (val
is not fully independent of either pass). Even so, the gap is informative:
both human-rulebook passes score well above either trained classifier on
priority macro-F1 (0.92-0.98 vs 0.83-0.84) and especially on STOP_NOW
recall (0.69-0.93 vs 0.58-0.60), which is the headline number the
post-hoc threshold step (section 2) exists to close.

### 10 val STOP_NOW texts ELECTRA (a) missed (argmax prediction, no threshold)

27 of the 68 actual STOP_NOW val texts were missed by ELECTRA (a)'s plain
argmax decision (recall 0.6029 -> 41/68 caught). First 10:

| uid | text | predicted priority | STOP_NOW probability |
|---|---|---|---|
| real-00060 | JN872 INV/REP EXHAUST LEAK | TODAY | 0.0269 |
| real-00600 | OC168 LOCATE AND FIT POS4 WHEEL GAURD | PLAN | 0.0024 |
| real-00896 | AY686 - TEL/REM CAMERA FAULT | TODAY | 0.0900 |
| real-00990 | DI743 BUCKET CREEPS UP | TODAY | 0.0003 |
| real-01387 | AR931 REPLACE BLOWN CROWD CYL HOSE | TODAY | 0.0625 |
| real-01903 | DK579 REPLACE BLOWN IMPLIMENT HOSE | TODAY | 0.1388 |
| real-02226 | RD227 PARK BRAKE APPLYING WHEN TRAMMING | TODAY | 0.0126 |
| real-02487 | C/O cracked hoist trunnion links LH&RH | TODAY | 0.2131 |
| real-02518 | Repair Cracked Fuel Module | TODAY | 0.0105 |
| real-02540 | L/H Front ramp cylinder mount cracked | TODAY | 0.3011 |

These are almost all short, code-heavy, abbreviation-dense texts (hose/cylinder/
guard/brake faults) where the rulebook's P1 safety triggers (brake failure,
guard missing/damaged, hazardous leak) require reading past abbreviations
like "C/O", "INV/REP", "LH&RH" that the model, unlike a rulebook-following
human pass, doesn't reliably key off.
