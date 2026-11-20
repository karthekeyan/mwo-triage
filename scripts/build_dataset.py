"""Build the final train/val/test files from the real and synthetic label
sources.

Inputs:
  configs/labels.yaml            fixed priority/route id mappings
  data/interim/real_labelled.jsonl   real texts, pool in {train, val, test}
  data/interim/synthetic_verified.jsonl  synthetic texts kept by the blind
                                      check (train pool only -- the test and
                                      val sets must be real text never seen
                                      in training, per CLAUDE.md section 2)

Steps:
  1. Load configs/labels.yaml for the id mappings (written by hand, not by
     this script -- this script only reads it, to keep the ids as the
     single source of truth).
  2. Split: train = real pool "train" + synthetic_verified; val = real pool
     "val"; test = real pool "test".
  3. Leakage checks:
       - no uid appears in more than one split.
       - no synthetic train text is an exact or near-duplicate (token
         Jaccard >= 0.85, the same rule as scripts/synthetic_tools.py
         cmd_dedup) of any val or test text. Any synthetic text that is
         gets dropped from train, and the drop count is reported.
  4. Write data/processed/{train,val,test}.jsonl, each record with uid,
     text, priority, route, priority_id, route_id, source ("real" or
     "synthetic"), label_source.
  5. Write data/processed/stats.json and print a summary: counts per
     split, priority/route distribution per split, the share of synthetic
     records in train per priority, and mean word count per split.

No API calls.

Usage:
  python scripts/build_dataset.py
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_tools import cleaned_tokens, jaccard, norm_key  # noqa: E402

LABELS_PATH = Path("configs/labels.yaml")
REAL_LABELLED_PATH = Path("data/interim/real_labelled.jsonl")
SYNTHETIC_VERIFIED_PATH = Path("data/interim/synthetic_verified.jsonl")
OUT_DIR = Path("data/processed")
STATS_PATH = OUT_DIR / "stats.json"

TRAIN_SHUFFLE_SEED = 42
JACCARD_THRESHOLD = 0.85


def load_jsonl(path):
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["priority"], cfg["route"]


# --------------------------------------------------------------- splits

def split_real(real_records):
    by_pool = defaultdict(list)
    for r in real_records:
        by_pool[r["pool"]].append(r)
    return by_pool["train"], by_pool["val"], by_pool["test"]


def real_record_to_row(r, priority_ids, route_ids):
    return {
        "uid": r["uid"],
        "text": r["text"],
        "priority": r["priority"],
        "route": r["route"],
        "priority_id": priority_ids[r["priority"]],
        "route_id": route_ids[r["route"]],
        "source": "real",
        "label_source": r["label_source"],
    }


def synthetic_record_to_row(r, priority_ids, route_ids):
    return {
        "uid": r["uid"],
        "text": r["text"],
        "priority": r["target_priority"],
        "route": r["target_route"],
        "priority_id": priority_ids[r["target_priority"]],
        "route_id": route_ids[r["target_route"]],
        "source": "synthetic",
        "label_source": "blind-verified",
    }


# ------------------------------------------------------------ leakage

def check_no_cross_split_uids(train, val, test):
    seen = {}
    dupes = []
    for split_name, rows in (("train", train), ("val", val), ("test", test)):
        for r in rows:
            if r["uid"] in seen and seen[r["uid"]] != split_name:
                dupes.append((r["uid"], seen[r["uid"]], split_name))
            seen[r["uid"]] = split_name
    if dupes:
        raise ValueError(f"uids appearing in more than one split: {dupes[:10]} ...")


def drop_synthetic_near_duplicates_of_heldout(synthetic_rows, val_rows, test_rows):
    """Drop any synthetic row whose text is an exact or near-duplicate
    (Jaccard >= 0.85 over cleaned tokens, as in synthetic_tools.cmd_dedup)
    of any val or test text. Returns (kept_rows, dropped_rows)."""
    heldout_texts = [r["text"] for r in val_rows] + [r["text"] for r in test_rows]
    heldout_token_sets = [cleaned_tokens(norm_key(t)) for t in heldout_texts]

    # Inverted index: token -> list of heldout indices, so we only compare
    # a synthetic text against heldout texts that share at least one
    # cleaned token (a near-duplicate pair always shares one).
    inverted = defaultdict(list)
    for i, toks in enumerate(heldout_token_sets):
        for t in toks:
            inverted[t].append(i)

    kept, dropped = [], []
    for row in synthetic_rows:
        toks = cleaned_tokens(norm_key(row["text"]))
        candidates = set()
        for t in toks:
            candidates.update(inverted.get(t, ()))
        is_dup = any(jaccard(toks, heldout_token_sets[i]) >= JACCARD_THRESHOLD for i in candidates)
        (dropped if is_dup else kept).append(row)
    return kept, dropped


# ---------------------------------------------------------------- stats

def word_count(text):
    return len(text.split())


def distribution(rows, field):
    c = Counter(r[field] for r in rows)
    return dict(sorted(c.items()))


def mean_word_count(rows):
    if not rows:
        return 0.0
    return sum(word_count(r["text"]) for r in rows) / len(rows)


def synthetic_share_per_priority(train_rows):
    by_priority = defaultdict(lambda: [0, 0])  # priority -> [synthetic, total]
    for r in train_rows:
        by_priority[r["priority"]][1] += 1
        if r["source"] == "synthetic":
            by_priority[r["priority"]][0] += 1
    return {
        p: (synthetic / total if total else 0.0)
        for p, (synthetic, total) in sorted(by_priority.items())
    }


def build_stats(train, val, test):
    splits = {"train": train, "val": val, "test": test}
    return {
        "counts": {name: len(rows) for name, rows in splits.items()},
        "priority_distribution": {name: distribution(rows, "priority") for name, rows in splits.items()},
        "route_distribution": {name: distribution(rows, "route") for name, rows in splits.items()},
        "synthetic_share_in_train_by_priority": synthetic_share_per_priority(train),
        "mean_word_count": {name: mean_word_count(rows) for name, rows in splits.items()},
    }


def print_stats(stats):
    print("\n=== Split counts ===")
    for name, n in stats["counts"].items():
        print(f"  {name}: {n}")

    print("\n=== Priority distribution per split ===")
    for name, dist in stats["priority_distribution"].items():
        print(f"  {name}: {dist}")

    print("\n=== Route distribution per split ===")
    for name, dist in stats["route_distribution"].items():
        print(f"  {name}: {dist}")

    print("\n=== Share of synthetic in train, per priority ===")
    for priority, share in stats["synthetic_share_in_train_by_priority"].items():
        print(f"  {priority}: {share:.1%}")

    print("\n=== Mean word count per split ===")
    for name, mean in stats["mean_word_count"].items():
        print(f"  {name}: {mean:.2f}")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    priority_ids, route_ids = load_labels()

    real = load_jsonl(REAL_LABELLED_PATH)
    synthetic_verified = load_jsonl(SYNTHETIC_VERIFIED_PATH)

    real_train, real_val, real_test = split_real(real)

    val_rows = [real_record_to_row(r, priority_ids, route_ids) for r in real_val]
    test_rows = [real_record_to_row(r, priority_ids, route_ids) for r in real_test]
    real_train_rows = [real_record_to_row(r, priority_ids, route_ids) for r in real_train]
    synthetic_rows = [synthetic_record_to_row(r, priority_ids, route_ids) for r in synthetic_verified]

    kept_synthetic, dropped_synthetic = drop_synthetic_near_duplicates_of_heldout(
        synthetic_rows, val_rows, test_rows
    )
    print(f"Synthetic texts dropped as a duplicate (Jaccard >= {JACCARD_THRESHOLD}) of a val/test text: "
          f"{len(dropped_synthetic)}")

    train_rows = real_train_rows + kept_synthetic
    random.Random(TRAIN_SHUFFLE_SEED).shuffle(train_rows)

    check_no_cross_split_uids(train_rows, val_rows, test_rows)

    write_jsonl(OUT_DIR / "train.jsonl", train_rows)
    write_jsonl(OUT_DIR / "val.jsonl", val_rows)
    write_jsonl(OUT_DIR / "test.jsonl", test_rows)
    print(f"Wrote {len(train_rows)} train, {len(val_rows)} val, {len(test_rows)} test records to {OUT_DIR}/")

    stats = build_stats(train_rows, val_rows, test_rows)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {STATS_PATH}")

    print_stats(stats)


if __name__ == "__main__":
    main()
