"""Tools for the synthetic train-pool work-order texts described in
configs/synthetic_plan.yaml.

Commands:
  next    Print the next chunk of 50 slots still to fill (cell, block,
          theme, equipment), in plan order.
  check   Validate data/interim/synthetic_raw.jsonl: required fields,
          valid labels, no duplicate uids, and progress per cell/block/
          theme versus the plan.
  dedup   Remove exact and near-duplicate (token-Jaccard >= 0.85, as in
          scripts/clean_and_split.py) texts from
          data/interim/synthetic_raw.jsonl, both within the synthetic set
          and against every real text in data/interim/real_labelled.jsonl.

No API calls. All texts are written by hand and appended to
data/interim/synthetic_raw.jsonl as they're written; this file only plans,
checks and cleans that file.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mwo_triage.rulebook_labels import PRIORITIES, ROUTES  # noqa: E402

PLAN_PATH = Path("configs/synthetic_plan.yaml")
SYNTHETIC_PATH = Path("data/interim/synthetic_raw.jsonl")
REAL_LABELLED_PATH = Path("data/interim/real_labelled.jsonl")

CHUNK_SIZE = 50
JACCARD_THRESHOLD = 0.85
NUM_RE = re.compile(r"\d")
NONALNUM_RE = re.compile(r"[^a-z0-9 ]+")

REQUIRED_FIELDS = ["uid", "text", "target_priority", "target_route", "block", "theme", "equipment"]
UID_RE = re.compile(r"^syn-\d{5}$")
VALID_BLOCKS = {"topup", "factory", "hard_case"}


def load_plan():
    with open(PLAN_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


# ---------------------------------------------------------------- slots

def build_slot_groups(plan):
    """Ordered list of slot groups: topup cells, then factory cells, then
    hard_case themes. Each group has a key used to match generated records
    against it."""
    groups = []
    for cell in plan["topup"]["cells"]:
        groups.append({
            "block": "topup",
            "priority": cell["priority"],
            "route": cell["route"],
            "theme": None,
            "target": cell["to_generate"],
        })
    for cell in plan["factory"]["cells"]:
        groups.append({
            "block": "factory",
            "priority": cell["priority"],
            "route": cell["route"],
            "theme": None,
            "target": cell["to_generate"],
        })
    for theme in plan["hard_case"]["themes"]:
        groups.append({
            "block": "hard_case",
            "priority": None,
            "route": None,
            "theme": theme["name"],
            "target": theme["to_generate"],
        })
    return groups


def group_key(record):
    block = record.get("block")
    if block == "hard_case":
        return (block, None, None, record.get("theme"))
    return (block, record.get("target_priority"), record.get("target_route"), None)


def group_dict_key(group):
    return (group["block"], group["priority"], group["route"], group["theme"])


def cmd_next(args):
    plan = load_plan()
    groups = build_slot_groups(plan)
    existing = load_jsonl(SYNTHETIC_PATH)

    done_counts = Counter(group_key(r) for r in existing)

    equipment_list = plan["factory"]["equipment"]
    equipment_cycle_pos = defaultdict(int)  # per factory group, how many equipment assignments handed out so far

    pending = []
    for group in groups:
        key = group_dict_key(group)
        done = done_counts.get(key, 0)
        remaining = max(0, group["target"] - done)
        for _ in range(remaining):
            slot = {
                "block": group["block"],
                "priority": group["priority"],
                "route": group["route"],
                "theme": group["theme"],
            }
            if group["block"] == "factory":
                idx = equipment_cycle_pos[key]
                slot["equipment"] = equipment_list[idx % len(equipment_list)]
                equipment_cycle_pos[key] += 1
            else:
                slot["equipment"] = None
            pending.append(slot)

    total_remaining = len(pending)
    chunk = pending[:CHUNK_SIZE]

    print(f"{total_remaining} slots remaining out of {plan['total_to_generate']} planned.")
    print(f"Next {len(chunk)}, in order (equipment cycles within a factory cell; repeat the cell/theme line that many times):")
    i = 0
    while i < len(chunk):
        slot = chunk[i]
        if slot["block"] == "factory":
            # Collapse a run of the same cell into one line, listing the
            # distinct equipment items covered (they cycle, so a run can
            # repeat an item).
            j = i
            equips = []
            while j < len(chunk) and chunk[j]["block"] == "factory" \
                    and chunk[j]["priority"] == slot["priority"] and chunk[j]["route"] == slot["route"]:
                equips.append(chunk[j]["equipment"])
                j += 1
            print(f"  x{j - i:<3d} block=factory cell={slot['priority']}/{slot['route']} equipment={', '.join(equips)}")
            i = j
        else:
            j = i
            while j < len(chunk) and chunk[j] == slot:
                j += 1
            if slot["block"] == "hard_case":
                print(f"  x{j - i:<3d} block=hard_case theme={slot['theme']}")
            else:
                print(f"  x{j - i:<3d} block=topup cell={slot['priority']}/{slot['route']}")
            i = j


# --------------------------------------------------------------- check

def cmd_check(args):
    plan = load_plan()
    records = load_jsonl(SYNTHETIC_PATH)

    print(f"{len(records)} records in {SYNTHETIC_PATH}")

    errors = []
    uids_seen = Counter()
    valid_themes = {t["name"] for t in plan["hard_case"]["themes"]}

    for i, r in enumerate(records):
        where = f"record {i} (uid={r.get('uid')!r})"
        missing = [f for f in REQUIRED_FIELDS if f not in r]
        if missing:
            errors.append(f"{where}: missing fields {missing}")
            continue

        if not UID_RE.match(r["uid"]):
            errors.append(f"{where}: uid does not match syn-xxxxx")
        uids_seen[r["uid"]] += 1

        if r["block"] not in VALID_BLOCKS:
            errors.append(f"{where}: invalid block {r['block']!r}")

        if r["target_priority"] not in PRIORITIES:
            errors.append(f"{where}: invalid target_priority {r['target_priority']!r}")
        if r["target_route"] not in ROUTES:
            errors.append(f"{where}: invalid target_route {r['target_route']!r}")

        if r["block"] == "hard_case":
            if r["theme"] not in valid_themes:
                errors.append(f"{where}: invalid theme {r['theme']!r} for hard_case")
        elif r["theme"] not in (None, ""):
            errors.append(f"{where}: block {r['block']} should not have a theme")

        if r["block"] == "factory" and not r.get("equipment"):
            errors.append(f"{where}: factory record missing equipment")
        if r["block"] != "factory" and r.get("equipment"):
            errors.append(f"{where}: block {r['block']} should not have equipment")

        words = r.get("text", "").split()
        if not (3 <= len(words) <= 15):
            errors.append(f"{where}: text has {len(words)} words, outside 3-15")

    dup_uids = [uid for uid, count in uids_seen.items() if count > 1]
    if dup_uids:
        errors.append(f"duplicate uids: {sorted(dup_uids)}")

    print(f"\n{len(errors)} validation errors:" if errors else "\nNo validation errors.")
    for e in errors[:50]:
        print(f"  {e}")
    if len(errors) > 50:
        print(f"  ... and {len(errors) - 50} more")

    groups = build_slot_groups(plan)
    done_counts = Counter(group_key(r) for r in records if "block" in r)

    print("\nProgress per group (done / target):")
    by_block_done = Counter()
    by_block_target = Counter()
    for group in groups:
        key = group_dict_key(group)
        done = done_counts.get(key, 0)
        by_block_done[group["block"]] += done
        by_block_target[group["block"]] += group["target"]
        label = group["theme"] if group["block"] == "hard_case" else f"{group['priority']}/{group['route']}"
        marker = "OK" if done >= group["target"] else "  "
        print(f"  [{marker}] {group['block']:10s} {label:40s} {done:4d} / {group['target']:4d}")

    print("\nProgress per block:")
    for block in ("topup", "factory", "hard_case"):
        print(f"  {block}: {by_block_done[block]} / {by_block_target[block]}")
    print(f"  total: {sum(by_block_done.values())} / {sum(by_block_target.values())}")


# --------------------------------------------------------------- dedup

def norm_key(text):
    text = text.lower()
    text = NONALNUM_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def cleaned_tokens(text_norm):
    words = text_norm.split()
    return frozenset(w for w in words if not NUM_RE.search(w))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


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


def cmd_dedup(args):
    synthetic = load_jsonl(SYNTHETIC_PATH)
    real = load_jsonl(REAL_LABELLED_PATH)

    if not synthetic:
        print(f"{SYNTHETIC_PATH} is empty or missing; nothing to dedup.")
        return

    # Combined corpus: real records first (never removed), then synthetic.
    n_real = len(real)
    texts = [r["text"] for r in real] + [r["text"] for r in synthetic]
    token_sets = [cleaned_tokens(norm_key(t)) for t in texts]

    df = Counter()
    for toks in token_sets:
        for t in toks:
            df[t] += 1

    buckets = defaultdict(list)
    for i, toks in enumerate(token_sets):
        key = min(toks, key=lambda t: (df[t], t)) if toks else "__EMPTY__"
        buckets[key].append(i)

    uf = UnionFind(len(texts))
    exact_pairs = 0
    near_pairs = 0
    for idxs in buckets.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                score = jaccard(token_sets[i], token_sets[j])
                if score >= JACCARD_THRESHOLD:
                    uf.union(i, j)
                    if score == 1.0:
                        exact_pairs += 1
                    else:
                        near_pairs += 1

    components = defaultdict(list)
    for i in range(len(texts)):
        components[uf.find(i)].append(i)

    drop_indices = set()
    dropped_vs_real = 0
    dropped_within_synthetic = 0
    for idxs in components.values():
        has_real = any(i < n_real for i in idxs)
        synthetic_idxs = sorted(i for i in idxs if i >= n_real)
        if not synthetic_idxs:
            continue
        if has_real:
            drop_indices.update(synthetic_idxs)
            dropped_vs_real += len(synthetic_idxs)
        else:
            keep, rest = synthetic_idxs[0], synthetic_idxs[1:]
            drop_indices.update(rest)
            dropped_within_synthetic += len(rest)

    kept = [synthetic[i - n_real] for i in range(n_real, len(texts)) if i not in drop_indices]

    print(f"Synthetic records in: {len(synthetic)}")
    print(f"Jaccard>={JACCARD_THRESHOLD} pairs found: {exact_pairs} exact, {near_pairs} near")
    print(f"Dropped as duplicate of a real text: {dropped_vs_real}")
    print(f"Dropped as near-duplicate within the synthetic set: {dropped_within_synthetic}")
    print(f"Remaining: {len(kept)}")

    with open(SYNTHETIC_PATH, "w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("next")
    sub.add_parser("check")
    sub.add_parser("dedup")
    args = parser.parse_args()

    {"next": cmd_next, "check": cmd_check, "dedup": cmd_dedup}[args.command](args)


if __name__ == "__main__":
    main()
