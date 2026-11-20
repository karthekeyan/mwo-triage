"""Export the teacher's temperature-softened probabilities for both
heads on every data/processed/train.jsonl text, for knowledge
distillation (scripts/train.py --distill-teacher-probs).

For each train text, computes softmax(teacher_logits / T) for the
priority head and the route head (T = --temperature, default 2, the
same temperature train.py's distillation loss uses), and writes
{uid, priority_soft_probs, route_soft_probs} per line.

No API calls. data/processed/test.jsonl is never loaded.

Usage:
  python -u scripts/export_teacher_probs.py \
      --model-id microsoft/deberta-v3-base \
      --cache-dir ~/models/microsoft__deberta-v3-base \
      --checkpoint outputs/bakeoff/teacher_deberta/best_model.pt \
      --output data/interim/teacher_train_probs.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mwo_triage.classifier import MultiHeadClassifier, MwoDataset  # noqa: E402

TRAIN_PATH = Path("data/processed/train.jsonl")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--checkpoint", required=True, help="Path to the teacher's best_model.pt")
    p.add_argument("--output", required=True)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--max-length", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir, local_files_only=True)
    train_dataset = MwoDataset(TRAIN_PATH, tokenizer, max_length=args.max_length)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    model = MultiHeadClassifier(args.model_id, freeze_encoder=False, cache_dir=args.cache_dir, local_files_only=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        idx = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            priority_logits, route_logits = model(input_ids, attention_mask, token_type_ids)
            priority_soft = torch.softmax(priority_logits / args.temperature, dim=-1).cpu().tolist()
            route_soft = torch.softmax(route_logits / args.temperature, dim=-1).cpu().tolist()

            batch_size = len(priority_soft)
            for j in range(batch_size):
                rows.append({
                    "uid": train_dataset.records[idx + j]["uid"],
                    "priority_soft_probs": priority_soft[j],
                    "route_soft_probs": route_soft[j],
                })
            idx += batch_size

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} teacher soft-probability rows (T={args.temperature}) to {args.output}", flush=True)


if __name__ == "__main__":
    main()
