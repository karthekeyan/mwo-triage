"""Clean real_texts.jsonl: drop junk, group near-duplicates, split into
test/val/train pools stratified by source family. Groups are never split
across pools.
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
IN_PATH = Path("data/interim/real_texts.jsonl")
OUT_TEST = Path("data/interim/real_test_pool.jsonl")
OUT_VAL = Path("data/interim/real_val_pool.jsonl")
OUT_TRAIN = Path("data/interim/real_train_pool.jsonl")

TEST_SIZE = 1500
VAL_SIZE = 800

JACCARD_THRESHOLD = 0.85

NUM_RE = re.compile(r"\d")
ALPHA_RE = re.compile(r"^[A-Za-z]+$")


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def is_junk(text):
    words = text.split()
    if len(words) > 40:
        return "too_long"
    no_id_no_num = [w for w in words if not NUM_RE.search(w)]
    alpha_words = [w for w in no_id_no_num if ALPHA_RE.match(w)]
    if len(alpha_words) < 2:
        return "too_few_alpha_words"
    return None


def cleaned_tokens(text_norm):
    words = text_norm.split()
    tokens = [w for w in words if not NUM_RE.search(w)]
    return frozenset(tokens)


def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    inter = a & b
    return len(inter) / len(union)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def group_near_duplicates(records, token_sets):
    # Block by each record's rarest token (lowest document frequency) so we
    # only pairwise-compare records that are plausible near-duplicates.
    # This keeps ~12k records fast while still catching Jaccard>=0.85 pairs,
    # since two near-duplicate texts almost always share their rarest token.
    df = Counter()
    for toks in token_sets:
        for t in toks:
            df[t] += 1

    buckets = defaultdict(list)
    for i, toks in enumerate(token_sets):
        if not toks:
            buckets[("__EMPTY__",)].append(i)
            continue
        rarest = min(toks, key=lambda t: (df[t], t))
        buckets[rarest].append(i)

    uf = UnionFind(len(records))
    for key, idxs in buckets.items():
        if key == ("__EMPTY__",):
            continue
        n = len(idxs)
        for a in range(n):
            for b in range(a + 1, n):
                i, j = idxs[a], idxs[b]
                if jaccard(token_sets[i], token_sets[j]) >= JACCARD_THRESHOLD:
                    uf.union(i, j)

    comp_to_group_id = {}
    group_ids = [None] * len(records)
    next_id = 0
    for i in range(len(records)):
        root = uf.find(i)
        if root not in comp_to_group_id:
            comp_to_group_id[root] = f"grp-{next_id:05d}"
            next_id += 1
        group_ids[i] = comp_to_group_id[root]
    return group_ids, next_id


def source_family(source):
    return source.split("/")[0]


def group_family(records_in_group):
    families = [source_family(r["source"]) for r in records_in_group]
    return Counter(families).most_common(1)[0][0]


def stratified_group_split(groups_by_family, rng):
    """groups_by_family: family -> list of (group_id, size). Returns dict
    group_id -> pool name ('test', 'val', 'train')."""
    total_texts = sum(size for groups in groups_by_family.values() for _, size in groups)
    assignment = {}
    for family, groups in groups_by_family.items():
        family_total = sum(size for _, size in groups)
        family_frac = family_total / total_texts
        target_test = round(TEST_SIZE * family_frac)
        target_val = round(VAL_SIZE * family_frac)

        shuffled = groups[:]
        rng.shuffle(shuffled)

        cum = 0
        pool = "test"
        target = target_test
        for gid, size in shuffled:
            if pool == "test" and cum >= target:
                pool = "val"
                target = target_val
                cum = 0
            if pool == "val" and cum >= target:
                pool = "train"
            assignment[gid] = pool
            cum += size
    return assignment


def main():
    rng = random.Random(SEED)
    records = load_records(IN_PATH)

    drop_counts = Counter()
    kept = []
    for r in records:
        reason = is_junk(r["text"])
        if reason:
            drop_counts[reason] += 1
        else:
            kept.append(r)

    token_sets = [cleaned_tokens(r["text_norm"]) for r in kept]
    group_ids, n_groups = group_near_duplicates(kept, token_sets)
    for r, gid in zip(kept, group_ids):
        r["group_id"] = gid

    groups = defaultdict(list)
    for r in kept:
        groups[r["group_id"]].append(r)

    groups_by_family = defaultdict(list)
    for gid, members in groups.items():
        fam = group_family(members)
        groups_by_family[fam].append((gid, len(members)))

    assignment = stratified_group_split(groups_by_family, rng)

    pools = {"test": [], "val": [], "train": []}
    for r in kept:
        pools[assignment[r["group_id"]]].append(r)

    with open(OUT_TEST, "w", encoding="utf-8") as f:
        for r in pools["test"]:
            f.write(json.dumps(r) + "\n")
    with open(OUT_VAL, "w", encoding="utf-8") as f:
        for r in pools["val"]:
            f.write(json.dumps(r) + "\n")
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for r in pools["train"]:
            f.write(json.dumps(r) + "\n")

    print("=== Dropped (junk) ===")
    for reason, count in drop_counts.items():
        print(f"  {reason}: {count}")
    print(f"  total dropped: {sum(drop_counts.values())}")
    print(f"  total kept: {len(kept)}")

    print(f"\n=== Groups ===")
    print(f"  number of groups: {n_groups}")

    print(f"\n=== Pool sizes by source family ===")
    for pool_name in ["test", "val", "train"]:
        fam_counts = Counter(source_family(r["source"]) for r in pools[pool_name])
        total = len(pools[pool_name])
        print(f"  {pool_name}_pool: total={total}  {dict(fam_counts)}")

    print(f"\n=== 5 random examples per pool ===")
    for pool_name in ["test", "val", "train"]:
        print(f"  --- {pool_name}_pool ---")
        sample = rng.sample(pools[pool_name], min(5, len(pools[pool_name])))
        for r in sample:
            print(f"    [{r['group_id']}] ({r['source']}) {r['text']}")

    test_groups = {r["group_id"] for r in pools["test"]}
    val_groups = {r["group_id"] for r in pools["val"]}
    train_groups = {r["group_id"] for r in pools["train"]}
    overlap = (test_groups & val_groups) | (test_groups & train_groups) | (val_groups & train_groups)
    print(f"\n=== Group leakage check ===")
    if overlap:
        print(f"  FAILED: {len(overlap)} group_id(s) appear in more than one pool!")
    else:
        print("  OK: no group_id appears in more than one pool")


if __name__ == "__main__":
    main()
