"""Verify a packaged release/<name>/ model's predictions on
data/processed/val.jsonl exactly match the original checkpoint's
calibrated predictions (same temperature-scaled argmax for both heads),
and print predictions for a handful of example texts in both modes.

Meant to run in a clean env (e.g. mwo-pkg-test) with only torch,
transformers, safetensors and huggingface_hub installed, against
scripts/inference.py's TriageModel. Needs data/processed/val.jsonl and
the original outputs/bakeoff/<run>/best_model.pt + calibration.json for
the "original" reference predictions -- run from the repo root.

Usage:
  python scripts/verify_release.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import TriageModel  # noqa: E402

VAL_PATH = Path("data/processed/val.jsonl")
RELEASE_ROOT = Path("release")

RUNS = [
    {"published_name": "mwo-triage-13M", "run_name": "electra_distilled"},
    {"published_name": "mwo-triage-5M", "run_name": "L4_a_earlystop_passb"},
]

EXAMPLES = [
    "Hydraulic press 3 leaking oil near main cylinder, pressure dropping during cycle",
    "Hyd hose burst on press 3, oil spraying across walkway",
    "c/o position 3 tyre",
    "Gearbox oil level low, top up",
    "Machine 7 not working",
]


def load_val_texts():
    records = []
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def original_predictions(run_name, texts):
    """Reproduce scripts/calibrate.py's calibrated balanced-mode argmax
    directly from outputs/bakeoff/<run_name>, as the ground truth the
    packaged model must match. Only used against val, from the full
    training env (this import is not part of scripts/inference.py's
    minimal dependency set)."""
    import torch
    import torch.nn.functional as F
    import yaml
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from mwo_triage.classifier import MultiHeadClassifier

    with open("configs/labels.yaml", "r", encoding="utf-8") as f:
        labels_cfg = yaml.safe_load(f)
    priority_names = [n for n, _ in sorted(labels_cfg["priority"].items(), key=lambda kv: kv[1])]
    route_names = [n for n, _ in sorted(labels_cfg["route"].items(), key=lambda kv: kv[1])]

    run_dir = Path("outputs/bakeoff") / run_name
    with open(run_dir / "calibration.json", "r", encoding="utf-8") as f:
        calibration = json.load(f)

    model_id = calibration["model_id"]
    cache_dir = str(Path.home() / "models" / ("google__" + model_id.split("/", 1)[1]))
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
    model = MultiHeadClassifier(model_id, freeze_encoder=False, cache_dir=cache_dir, local_files_only=True)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
    model.eval()

    encoded = tokenizer(texts, max_length=32, padding="max_length", truncation=True, return_tensors="pt")
    kwargs = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
    if "token_type_ids" in encoded:
        kwargs["token_type_ids"] = encoded["token_type_ids"]
    with torch.no_grad():
        priority_logits, route_logits = model(**kwargs)
    priority_probs = F.softmax(priority_logits / calibration["priority_temperature"], dim=-1)
    route_probs = F.softmax(route_logits / calibration["route_temperature"], dim=-1)
    priority_pred = [priority_names[i] for i in priority_probs.argmax(dim=-1).tolist()]
    route_pred = [route_names[i] for i in route_probs.argmax(dim=-1).tolist()]
    return priority_pred, route_pred


def main():
    val_records = load_val_texts()
    val_texts = [r["text"] for r in val_records]

    for run in RUNS:
        release_dir = RELEASE_ROOT / run["published_name"]
        print(f"\n=== {run['published_name']} ===")

        model = TriageModel.from_pretrained(str(release_dir))
        packaged = model.predict(val_texts, mode="balanced")
        packaged_priority = [r["priority"] for r in packaged]
        packaged_route = [r["route"] for r in packaged]

        original_priority, original_route = original_predictions(run["run_name"], val_texts)

        n = len(val_texts)
        priority_matches = sum(a == b for a, b in zip(packaged_priority, original_priority))
        route_matches = sum(a == b for a, b in zip(packaged_route, original_route))
        both_matches = sum(
            a == b and c == d
            for a, b, c, d in zip(packaged_priority, original_priority, packaged_route, original_route)
        )
        print(f"  priority match: {priority_matches}/{n} ({priority_matches / n:.4%})")
        print(f"  route match:    {route_matches}/{n} ({route_matches / n:.4%})")
        print(f"  both match:     {both_matches}/{n} ({both_matches / n:.4%})")

        print("\n  Example outputs:")
        for mode in ("balanced", "safety"):
            print(f"  -- mode={mode} --")
            for text, result in zip(EXAMPLES, model.predict(EXAMPLES, mode=mode)):
                print(
                    f"    {text!r}\n"
                    f"      -> priority={result['priority']} (p={result['priority_confidence']:.4f}), "
                    f"route={result['route']} (p={result['route_confidence']:.4f})"
                )


if __name__ == "__main__":
    main()
