"""Final-evaluation step (3): the one and only test-set evaluation for
this project (data/processed/test.jsonl is loaded exactly once, by this
script, after frozen baselines were retrained and calibration was fit
-- both on val only, in scripts/train_frozen_final.sh and
scripts/calibrate.py).

Evaluates five models:
  - mwo-triage-13M (Small, final): outputs/bakeoff/electra_distilled
  - mwo-triage-5M  (Tiny, final):  outputs/bakeoff/L4_a_earlystop_passb
  - electra_frozen_final: frozen-encoder baseline for the Small base model
  - L4_frozen_final:      frozen-encoder baseline for the Tiny base model
  - teacher_deberta:      the distillation teacher (microsoft/deberta-v3-base), reference only

For the two final models (which have a calibration.json from
scripts/calibrate.py), two decision modes are reported:
  - balanced: argmax over temperature-scaled probabilities
  - safety:   STOP_NOW whenever the temperature-scaled STOP_NOW
              probability is at or above the val-chosen threshold,
              else argmax over the remaining classes
The frozen baselines and the teacher have no calibration fit (per the
task), so only a single (raw-softmax argmax) mode is reported for them,
and their "after calibration" fields are null.

For every model, reports: priority/route macro-F1 and accuracy;
per-class precision/recall/F1; confusion matrices; STOP_NOW
recall/precision in both modes (where applicable); ECE (top-1
confidence, 10 bins) before and after temperature scaling; macro-F1 by
text length bucket (<=4 words, 5-7, >=8) and by real-text source
(maintnorm, maintie, from data/interim/real_labelled.jsonl -- the test
set is real-text only per CLAUDE.md); model size on disk (best_model.pt,
MB); and mean CPU latency per text (batch size 1, torch CPU only, first
200 test texts, seed 42 already baked into the test split order).

Writes docs/evaluation.md (human-readable) and
outputs/final_eval/metrics.json (full machine-readable results).

No API calls.

Usage:
  python -u scripts/evaluate.py
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mwo_triage.classifier import MultiHeadClassifier, MwoDataset  # noqa: E402

LABELS_PATH = Path("configs/labels.yaml")
TEST_PATH = Path("data/processed/test.jsonl")
REAL_LABELLED_PATH = Path("data/interim/real_labelled.jsonl")
OUTPUT_ROOT = Path("outputs/bakeoff")
FINAL_EVAL_DIR = Path("outputs/final_eval")
EVAL_DOC_PATH = Path("docs/evaluation.md")
N_LATENCY_TEXTS = 200

MODELS = [
    {
        "key": "mwo-triage-13M",
        "label": "mwo-triage-13M (Small, final)",
        "model_id": "google/electra-small-discriminator",
        "cache_dir": str(Path.home() / "models" / "google__electra-small-discriminator"),
        "run_name": "electra_distilled",
        "freeze_encoder": False,
        "has_calibration": True,
    },
    {
        "key": "mwo-triage-5M",
        "label": "mwo-triage-5M (Tiny, final)",
        "model_id": "google/bert_uncased_L-4_H-128_A-2",
        "cache_dir": str(Path.home() / "models" / "google__bert_uncased_L-4_H-128_A-2"),
        "run_name": "L4_a_earlystop_passb",
        "freeze_encoder": False,
        "has_calibration": True,
    },
    {
        "key": "electra_frozen_final",
        "label": "ELECTRA-small frozen baseline",
        "model_id": "google/electra-small-discriminator",
        "cache_dir": str(Path.home() / "models" / "google__electra-small-discriminator"),
        "run_name": "electra_frozen_final",
        "freeze_encoder": True,
        "has_calibration": False,
    },
    {
        "key": "L4_frozen_final",
        "label": "BERT L-4/H-128 frozen baseline",
        "model_id": "google/bert_uncased_L-4_H-128_A-2",
        "cache_dir": str(Path.home() / "models" / "google__bert_uncased_L-4_H-128_A-2"),
        "run_name": "L4_frozen_final",
        "freeze_encoder": True,
        "has_calibration": False,
    },
    {
        "key": "teacher_deberta",
        "label": "DeBERTa-v3-base teacher (reference)",
        "model_id": "microsoft/deberta-v3-base",
        "cache_dir": str(Path.home() / "models" / "microsoft__deberta-v3-base"),
        "run_name": "teacher_deberta",
        "freeze_encoder": False,
        "has_calibration": False,
    },
]


def load_label_order():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    priority_ids = cfg["priority"]
    route_ids = cfg["route"]
    priority_order = sorted(priority_ids, key=lambda n: priority_ids[n])
    route_order = sorted(route_ids, key=lambda n: route_ids[n])
    return priority_ids["STOP_NOW"], priority_order, route_order


def load_test_records():
    """The single read of data/processed/test.jsonl for this whole evaluation."""
    records = []
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_real_source_by_uid():
    """uid -> 'maintnorm' / 'maintie', from data/interim/real_labelled.jsonl's
    'source' field (e.g. 'maintnorm/train', 'maintie/engineering')."""
    by_uid = {}
    with open(REAL_LABELLED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                by_uid[row["uid"]] = row["source"].split("/")[0]
    return by_uid


def length_bucket(text):
    n_words = len(text.split())
    if n_words <= 4:
        return "<=4 words"
    if n_words <= 7:
        return "5-7 words"
    return ">=8 words"


def collect_logits(model, loader, device):
    priority_true, priority_logits = [], []
    route_true, route_logits = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            p_logits, r_logits = model(input_ids, attention_mask, token_type_ids)

            priority_true.extend(batch["priority_id"].tolist())
            priority_logits.append(p_logits.cpu())
            route_true.extend(batch["route_id"].tolist())
            route_logits.append(r_logits.cpu())

    return (
        torch.tensor(priority_true),
        torch.cat(priority_logits, dim=0),
        torch.tensor(route_true),
        torch.cat(route_logits, dim=0),
    )


def ece(probs, true_labels, n_bins=10):
    confidences, preds = probs.max(dim=-1)
    correct = (preds == true_labels).float().numpy()
    confidences = confidences.numpy()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidences)
    err = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        err += (mask.sum() / total) * abs(confidences[mask].mean() - correct[mask].mean())
    return float(err)


def apply_safety_threshold(priority_probs, stop_now_id, threshold):
    stop_now_probs = priority_probs[:, stop_now_id]
    non_stop_now_ids = [i for i in range(priority_probs.shape[1]) if i != stop_now_id]
    fallback_pred = np.array(non_stop_now_ids)[priority_probs[:, non_stop_now_ids].argmax(axis=-1)]
    return np.where(stop_now_probs >= threshold, stop_now_id, fallback_pred)


def stop_now_recall_precision(true, pred, stop_now_id):
    tp = int(((pred == stop_now_id) & (true == stop_now_id)).sum())
    actual_pos = int((true == stop_now_id).sum())
    predicted_pos = int((pred == stop_now_id).sum())
    recall = tp / actual_pos if actual_pos else 0.0
    precision = tp / predicted_pos if predicted_pos else 0.0
    return recall, precision


def per_class_report(true, pred, label_order):
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=list(range(len(label_order))), zero_division=0
    )
    return {
        label_order[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(label_order))
    }


def macro_f1_by_group(records, group_fn, priority_true, priority_pred, route_true, route_pred):
    groups = {}
    for i, r in enumerate(records):
        groups.setdefault(group_fn(r, i), []).append(i)

    out = {}
    for name, idxs in groups.items():
        idxs = np.array(idxs)
        out[name] = {
            "n": int(len(idxs)),
            "priority_macro_f1": float(
                f1_score(priority_true[idxs], priority_pred[idxs], average="macro", zero_division=0)
            ),
            "route_macro_f1": float(
                f1_score(route_true[idxs], route_pred[idxs], average="macro", zero_division=0)
            ),
        }
    return out


def measure_cpu_latency(model_id, cache_dir, run_name, freeze_encoder, texts, n_texts=N_LATENCY_TEXTS):
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
    model = MultiHeadClassifier(model_id, freeze_encoder=freeze_encoder, cache_dir=cache_dir, local_files_only=True)
    state_dict = torch.load(OUTPUT_ROOT / run_name / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    sample = texts[:n_texts]
    encoded = [
        tokenizer(t, max_length=32, padding="max_length", truncation=True, return_tensors="pt")
        for t in sample
    ]

    # Warm-up, not timed.
    with torch.no_grad():
        for enc in encoded[: min(5, len(encoded))]:
            model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))

    start = time.perf_counter()
    with torch.no_grad():
        for enc in encoded:
            model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
    elapsed = time.perf_counter() - start

    return (elapsed / len(encoded)) * 1000.0  # ms/text


def evaluate_model(cfg, test_records, priority_order, route_order, stop_now_id, source_by_uid):
    run_dir = OUTPUT_ROOT / cfg["run_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], cache_dir=cfg["cache_dir"], local_files_only=True)
    test_dataset = MwoDataset(TEST_PATH, tokenizer, max_length=32)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MultiHeadClassifier(
        cfg["model_id"], freeze_encoder=cfg["freeze_encoder"], cache_dir=cfg["cache_dir"], local_files_only=True
    )
    state_dict = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    priority_true, priority_logits, route_true, route_logits = collect_logits(model, test_loader, device)
    priority_true_np, route_true_np = priority_true.numpy(), route_true.numpy()

    raw_priority_probs = F.softmax(priority_logits, dim=-1)
    raw_route_probs = F.softmax(route_logits, dim=-1)
    ece_priority_before = ece(raw_priority_probs, priority_true)
    ece_route_before = ece(raw_route_probs, route_true)

    calibration = None
    if cfg["has_calibration"]:
        with open(run_dir / "calibration.json", "r", encoding="utf-8") as f:
            calibration = json.load(f)
        pt, rt = calibration["priority_temperature"], calibration["route_temperature"]
        priority_probs = F.softmax(priority_logits / pt, dim=-1)
        route_probs = F.softmax(route_logits / rt, dim=-1)
        ece_priority_after = ece(priority_probs, priority_true)
        ece_route_after = ece(route_probs, route_true)
    else:
        priority_probs = raw_priority_probs
        route_probs = raw_route_probs
        ece_priority_after = None
        ece_route_after = None

    priority_probs_np = priority_probs.numpy()
    route_pred_np = route_probs.numpy().argmax(axis=-1)

    balanced_priority_pred = priority_probs_np.argmax(axis=-1)
    balanced = {
        "mode": "balanced (argmax, calibrated probs)" if calibration else "balanced (argmax, raw softmax)",
        "priority_macro_f1": float(f1_score(priority_true_np, balanced_priority_pred, average="macro", zero_division=0)),
        "priority_accuracy": float(accuracy_score(priority_true_np, balanced_priority_pred)),
        "route_macro_f1": float(f1_score(route_true_np, route_pred_np, average="macro", zero_division=0)),
        "route_accuracy": float(accuracy_score(route_true_np, route_pred_np)),
        "priority_per_class": per_class_report(priority_true_np, balanced_priority_pred, priority_order),
        "route_per_class": per_class_report(route_true_np, route_pred_np, route_order),
        "priority_confusion_matrix": {
            "label_order": priority_order,
            "matrix": confusion_matrix(
                priority_true_np, balanced_priority_pred, labels=list(range(len(priority_order)))
            ).tolist(),
        },
        "route_confusion_matrix": {
            "label_order": route_order,
            "matrix": confusion_matrix(
                route_true_np, route_pred_np, labels=list(range(len(route_order)))
            ).tolist(),
        },
    }
    balanced["stop_now_recall"], balanced["stop_now_precision"] = stop_now_recall_precision(
        priority_true_np, balanced_priority_pred, stop_now_id
    )

    safety = None
    if calibration:
        safety_priority_pred = apply_safety_threshold(priority_probs_np, stop_now_id, calibration["stop_now_threshold"])
        safety = {
            "mode": f"safety (STOP_NOW if P>=t={calibration['stop_now_threshold']:.4f}, else argmax)",
            "priority_macro_f1": float(f1_score(priority_true_np, safety_priority_pred, average="macro", zero_division=0)),
            "priority_accuracy": float(accuracy_score(priority_true_np, safety_priority_pred)),
            "priority_per_class": per_class_report(priority_true_np, safety_priority_pred, priority_order),
            "priority_confusion_matrix": {
                "label_order": priority_order,
                "matrix": confusion_matrix(
                    priority_true_np, safety_priority_pred, labels=list(range(len(priority_order)))
                ).tolist(),
            },
        }
        safety["stop_now_recall"], safety["stop_now_precision"] = stop_now_recall_precision(
            priority_true_np, safety_priority_pred, stop_now_id
        )

    by_length = macro_f1_by_group(
        test_records, lambda r, i: length_bucket(r["text"]), priority_true, balanced_priority_pred, route_true, route_pred_np
    )
    by_source = macro_f1_by_group(
        test_records, lambda r, i: source_by_uid.get(r["uid"], "unknown"), priority_true, balanced_priority_pred, route_true, route_pred_np
    )

    checkpoint_path = run_dir / "best_model.pt"
    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    latency_texts = [r["text"] for r in test_records]
    latency_ms = measure_cpu_latency(
        cfg["model_id"], cfg["cache_dir"], cfg["run_name"], cfg["freeze_encoder"], latency_texts
    )

    return {
        "label": cfg["label"],
        "model_id": cfg["model_id"],
        "run_name": cfg["run_name"],
        "n_test": len(test_records),
        "model_size_mb": size_mb,
        "cpu_latency_ms_per_text": latency_ms,
        "ece_priority_before": ece_priority_before,
        "ece_priority_after": ece_priority_after,
        "ece_route_before": ece_route_before,
        "ece_route_after": ece_route_after,
        "calibration": calibration,
        "balanced": balanced,
        "safety": safety,
        "macro_f1_by_length": by_length,
        "macro_f1_by_source": by_source,
    }


def render_markdown(results, priority_order, route_order):
    lines = []
    lines.append("# Final evaluation on the held-out test set\n")
    lines.append(
        "data/processed/test.jsonl (1,500 real texts, never seen in training or any "
        "val-only tuning step) was loaded exactly once, by `scripts/evaluate.py`, "
        "after step (1) retrained the frozen-encoder baselines and step (2) fit "
        "per-head temperature scaling and the STOP_NOW safety threshold -- both on "
        "val only. See `docs/experiment-log.md` section 6 for the final model picks.\n"
    )
    lines.append("## Headline table\n")
    lines.append(
        "| model | priority F1 (bal.) | route F1 (bal.) | STOP_NOW recall (bal. / safety) | "
        "STOP_NOW precision (bal. / safety) | size (MB) | CPU latency (ms/text) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        bal = r["balanced"]
        safe = r["safety"]
        recall_str = f"{bal['stop_now_recall']:.4f} / {safe['stop_now_recall']:.4f}" if safe else f"{bal['stop_now_recall']:.4f} / --"
        prec_str = f"{bal['stop_now_precision']:.4f} / {safe['stop_now_precision']:.4f}" if safe else f"{bal['stop_now_precision']:.4f} / --"
        lines.append(
            f"| {r['label']} | {bal['priority_macro_f1']:.4f} | {bal['route_macro_f1']:.4f} | "
            f"{recall_str} | {prec_str} | {r['model_size_mb']:.2f} | {r['cpu_latency_ms_per_text']:.3f} |"
        )
    lines.append("")

    for r in results:
        lines.append(f"## {r['label']} (`{r['run_name']}`)\n")
        lines.append(f"- model id: `{r['model_id']}`")
        lines.append(f"- test set size: {r['n_test']}")
        lines.append(f"- model size on disk: {r['model_size_mb']:.2f} MB")
        lines.append(f"- mean CPU latency (batch size 1, torch CPU, {N_LATENCY_TEXTS} texts): {r['cpu_latency_ms_per_text']:.3f} ms/text")
        if r["calibration"]:
            c = r["calibration"]
            lines.append(
                f"- calibration: priority T={c['priority_temperature']:.4f}, route T={c['route_temperature']:.4f}, "
                f"STOP_NOW threshold t={c['stop_now_threshold']:.4f} (fit on val for recall >= {c['target_stop_now_recall']})"
            )
        else:
            lines.append("- calibration: not fit for this model (frozen baseline / teacher reference)")
        lines.append(
            f"- ECE priority before/after calibration: {r['ece_priority_before']:.4f} / "
            f"{r['ece_priority_after']:.4f}" if r["ece_priority_after"] is not None
            else f"- ECE priority (no calibration fit): {r['ece_priority_before']:.4f}"
        )
        lines.append(
            f"- ECE route before/after calibration: {r['ece_route_before']:.4f} / "
            f"{r['ece_route_after']:.4f}" if r["ece_route_after"] is not None
            else f"- ECE route (no calibration fit): {r['ece_route_before']:.4f}"
        )
        lines.append("")

        for mode_name, mode in (("Balanced mode", r["balanced"]), ("Safety mode", r["safety"])):
            if mode is None:
                continue
            lines.append(f"### {mode_name} ({mode['mode']})\n")
            lines.append(f"- priority macro-F1: {mode['priority_macro_f1']:.4f}, accuracy: {mode['priority_accuracy']:.4f}")
            if "route_macro_f1" in mode:
                lines.append(f"- route macro-F1: {mode['route_macro_f1']:.4f}, accuracy: {mode['route_accuracy']:.4f}")
            lines.append(f"- STOP_NOW recall: {mode['stop_now_recall']:.4f}, precision: {mode['stop_now_precision']:.4f}")
            lines.append("")
            lines.append("Priority per-class:\n")
            lines.append("| class | precision | recall | f1 | support |")
            lines.append("|---|---|---|---|---|")
            for cls in priority_order:
                m = mode["priority_per_class"][cls]
                lines.append(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")
            lines.append("")
            lines.append(
                "Priority confusion matrix (rows=actual, cols=predicted, order="
                + ", ".join(priority_order) + "):\n```"
            )
            for label, row in zip(priority_order, mode["priority_confusion_matrix"]["matrix"]):
                lines.append(f"{label:10s} {row}")
            lines.append("```\n")

            if "route_per_class" in mode:
                lines.append("Route per-class:\n")
                lines.append("| class | precision | recall | f1 | support |")
                lines.append("|---|---|---|---|---|")
                for cls in route_order:
                    m = mode["route_per_class"][cls]
                    lines.append(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")
                lines.append("")
                lines.append(
                    "Route confusion matrix (rows=actual, cols=predicted, order="
                    + ", ".join(route_order) + "):\n```"
                )
                for label, row in zip(route_order, mode["route_confusion_matrix"]["matrix"]):
                    lines.append(f"{label:25s} {row}")
                lines.append("```\n")

        lines.append("Macro-F1 by text length:\n")
        lines.append("| bucket | n | priority macro-F1 | route macro-F1 |")
        lines.append("|---|---|---|---|")
        for bucket in ("<=4 words", "5-7 words", ">=8 words"):
            m = r["macro_f1_by_length"].get(bucket)
            if m:
                lines.append(f"| {bucket} | {m['n']} | {m['priority_macro_f1']:.4f} | {m['route_macro_f1']:.4f} |")
        lines.append("")

        lines.append("Macro-F1 by real-text source:\n")
        lines.append("| source | n | priority macro-F1 | route macro-F1 |")
        lines.append("|---|---|---|---|")
        for source, m in sorted(r["macro_f1_by_source"].items()):
            lines.append(f"| {source} | {m['n']} | {m['priority_macro_f1']:.4f} | {m['route_macro_f1']:.4f} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def main():
    stop_now_id, priority_order, route_order = load_label_order()
    test_records = load_test_records()
    source_by_uid = load_real_source_by_uid()

    print(f"Loaded {len(test_records)} test records (single load for this whole evaluation).", flush=True)

    results = []
    for cfg in MODELS:
        print(f"\n=== Evaluating {cfg['label']} ({cfg['run_name']}) on test ===", flush=True)
        result = evaluate_model(cfg, test_records, priority_order, route_order, stop_now_id, source_by_uid)
        results.append(result)
        bal = result["balanced"]
        print(
            f"  priority_f1={bal['priority_macro_f1']:.4f} route_f1={bal['route_macro_f1']:.4f} "
            f"stop_now_recall={bal['stop_now_recall']:.4f} stop_now_precision={bal['stop_now_precision']:.4f} "
            f"size={result['model_size_mb']:.2f}MB latency={result['cpu_latency_ms_per_text']:.3f}ms/text",
            flush=True,
        )

    FINAL_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(FINAL_EVAL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_test": len(test_records),
                "priority_label_order": priority_order,
                "route_label_order": route_order,
                "models": results,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {FINAL_EVAL_DIR / 'metrics.json'}", flush=True)

    markdown = render_markdown(results, priority_order, route_order)
    with open(EVAL_DOC_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Wrote {EVAL_DOC_PATH}", flush=True)


if __name__ == "__main__":
    main()
