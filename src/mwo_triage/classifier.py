"""Shared model and data-loading code for the mwo-triage encoder classifier:
one encoder with two linear heads (priority, route) on the [CLS] token of
the last hidden state, used by scripts/train.py for every bake-off run.
"""

import json

import torch
from torch.utils.data import Dataset
from transformers import AutoModel

NUM_PRIORITY_CLASSES = 4
NUM_ROUTE_CLASSES = 6


class MultiHeadClassifier(torch.nn.Module):
    """A pretrained encoder with two linear classification heads (priority,
    route) on the [CLS] token of the last hidden state. Works the same way
    for encoders with or without a pooler (e.g. ELECTRA has none), since
    both heads read the token-0 hidden state directly rather than relying
    on an architecture-specific pooler.
    """

    def __init__(self, model_name_or_path, freeze_encoder=False, cache_dir=None, local_files_only=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, local_files_only=local_files_only
        )
        hidden_size = self.encoder.config.hidden_size
        self.priority_head = torch.nn.Linear(hidden_size, NUM_PRIORITY_CLASSES)
        self.route_head = torch.nn.Linear(hidden_size, NUM_ROUTE_CLASSES)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        cls_hidden = outputs.last_hidden_state[:, 0]
        return self.priority_head(cls_hidden), self.route_head(cls_hidden)


class MwoDataset(Dataset):
    """A tokenized data/processed/{split}.jsonl file. Each item carries
    input_ids, attention_mask, (token_type_ids if the tokenizer makes
    them), priority_id and route_id.
    """

    def __init__(self, path, tokenizer, max_length=32):
        self.records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        encoded = tokenizer(
            [r["text"] for r in self.records],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.token_type_ids = encoded.get("token_type_ids")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "priority_id": self.records[idx]["priority_id"],
            "route_id": self.records[idx]["route_id"],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        return item
