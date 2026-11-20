"""Download the six candidate base models into ~/models/ and report, for
each: licence (from the model card), encoder parameter count, total
parameter count with two linear classification heads added (4 priority
classes + 6 route classes on the pooled output), vocab size, layer count,
hidden size, and the mean / 95th-percentile token length of our train
texts under that model's tokenizer.

No Anthropic API calls. Downloads from Hugging Face only.

Usage:
  python -u scripts/survey_base_models.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODELS_DIR = Path.home() / "models"
TRAIN_PATH = Path("data/processed/train.jsonl")

CANDIDATES = [
    "google/bert_uncased_L-2_H-128_A-2",
    "google/bert_uncased_L-4_H-128_A-2",
    "google/bert_uncased_L-6_H-128_A-2",
    "google/electra-small-discriminator",
    "google/bert_uncased_L-8_H-256_A-4",
    "google/bert_uncased_L-12_H-256_A-4",
]

NUM_PRIORITY_CLASSES = 4
NUM_ROUTE_CLASSES = 6


def load_train_texts():
    texts = []
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return texts


def get_license(api, model_id):
    info = api.model_info(model_id)
    card = info.cardData or {}
    if "license" in card:
        return card["license"]
    if info.tags:
        for tag in info.tags:
            if tag.startswith("license:"):
                return tag.split(":", 1)[1]
    return "unknown"


def head_param_count(hidden_size):
    # One linear head per task on the pooled [hidden_size] output:
    # weight [classes, hidden_size] + bias [classes].
    priority_head = hidden_size * NUM_PRIORITY_CLASSES + NUM_PRIORITY_CLASSES
    route_head = hidden_size * NUM_ROUTE_CLASSES + NUM_ROUTE_CLASSES
    return priority_head + route_head


def token_length_stats(tokenizer, texts):
    lengths = [len(tokenizer(t)["input_ids"]) for t in texts]
    arr = np.array(lengths)
    return float(arr.mean()), float(np.percentile(arr, 95))


def survey_one(api, model_id, texts, local_dir):
    print(f"\n--- {model_id} ---", flush=True)
    license_str = get_license(api, model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=local_dir)
    config = AutoConfig.from_pretrained(model_id, cache_dir=local_dir)
    model = AutoModel.from_pretrained(model_id, cache_dir=local_dir)

    encoder_params = sum(p.numel() for p in model.parameters())
    hidden_size = config.hidden_size
    total_params = encoder_params + head_param_count(hidden_size)
    num_layers = getattr(config, "num_hidden_layers", None)
    vocab_size = config.vocab_size

    mean_len, p95_len = token_length_stats(tokenizer, texts)

    row = {
        "model_id": model_id,
        "license": license_str,
        "encoder_params": encoder_params,
        "total_params_with_heads": total_params,
        "vocab_size": vocab_size,
        "layers": num_layers,
        "hidden_size": hidden_size,
        "mean_token_len": round(mean_len, 2),
        "p95_token_len": round(p95_len, 2),
    }
    print(json.dumps(row, indent=2), flush=True)
    return row


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    texts = load_train_texts()
    print(f"Loaded {len(texts)} train texts for tokenizer length stats", flush=True)

    api = HfApi()
    rows = []
    for model_id in CANDIDATES:
        local_dir = MODELS_DIR / model_id.replace("/", "__")
        try:
            row = survey_one(api, model_id, texts, local_dir)
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {model_id}: {e}", flush=True)

    print("\n=== Summary table ===", flush=True)
    header = ["model_id", "license", "encoder_params", "total_params_with_heads",
              "vocab_size", "layers", "hidden_size", "mean_token_len", "p95_token_len"]
    print(" | ".join(header), flush=True)
    for row in rows:
        print(" | ".join(str(row[h]) for h in header), flush=True)

    out_path = MODELS_DIR / "survey_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
