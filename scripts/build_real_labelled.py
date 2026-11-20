"""Build the final real-text label file from the two labelling passes, the
test/val adjudications, and the ruling overrides.

For every uid in data/interim/real_all_pools.jsonl:
  - if full_pass_a and full_pass_b agree on both priority and route, use
    that label (label_source "agree").
  - otherwise, if the pool is test or val, use the adjudicated label from
    data/interim/adjudicated_testval.jsonl (Fable, batch mode) or
    data/interim/adj_inline.jsonl (in-session), with label_source
    "adjudicated:<adjudicator>"; drop it (reason "adjudicated_drop") if the
    adjudicator's decision was DROP.
  - otherwise (a train-pool disagreement), use Pass B's label (label_source
    "passB-default"). Val/val-metric checks (docs/experiment-log.md section 3)
    found Pass B (val priority macro-F1 0.975) more accurate than Pass A
    (0.923), and Pass A systematically under-calls STOP_NOW -- dropping
    these disagreements, as earlier versions of this script did, removed
    mostly borderline STOP_NOW cases from train. No train uid is dropped
    for disagreement any more.

data/interim/rulings_overrides.jsonl is then applied on top of whatever
survives: its new_priority/new_rule/new_route (whichever are set)
replace the corresponding fields, and "+override" is appended to
label_source.

Output: data/interim/real_labelled.jsonl, one record per surviving uid,
with uid, pool, source, group_id, text, priority, route, rule,
label_source, rulebook_version.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

REAL_ALL_POOLS_PATH = Path("data/interim/real_all_pools.jsonl")
PASS_A_PATH = Path("data/interim/full_pass_a.jsonl")
PASS_B_PATH = Path("data/interim/full_pass_b.jsonl")
ADJUDICATED_TESTVAL_PATH = Path("data/interim/adjudicated_testval.jsonl")
ADJ_INLINE_PATH = Path("data/interim/adj_inline.jsonl")
OVERRIDES_PATH = Path("data/interim/rulings_overrides.jsonl")
OUTPUT_PATH = Path("data/interim/real_labelled.jsonl")

ADJUDICATED_POOLS = {"test", "val"}


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def index_by_uid(records):
    return {r["uid"]: r for r in records}


def build_adjudication_index():
    """uid -> (decision, priority, route, rule, rulebook_version, adjudicator)."""
    index = {}
    for record in load_jsonl(ADJUDICATED_TESTVAL_PATH):
        index[record["uid"]] = (
            record["decision"], record.get("priority"), record.get("route"),
            record.get("rule"), record["rulebook_version"], record["model"],
        )
    for record in load_jsonl(ADJ_INLINE_PATH):
        index[record["uid"]] = (
            record["decision"], record.get("priority"), record.get("route"),
            record.get("rule"), record["rulebook_version"], record["adjudicator"],
        )
    return index


def main():
    pool_records = load_jsonl(REAL_ALL_POOLS_PATH)
    pass_a = index_by_uid(load_jsonl(PASS_A_PATH))
    pass_b = index_by_uid(load_jsonl(PASS_B_PATH))
    adjudication = build_adjudication_index()
    overrides = index_by_uid(load_jsonl(OVERRIDES_PATH))

    output = []
    dropped_reasons = Counter()
    kept_per_pool = Counter()
    overrides_applied = 0
    passb_default_count = 0
    group_pools = defaultdict(set)

    for meta in pool_records:
        uid, pool = meta["uid"], meta["pool"]
        a, b = pass_a[uid], pass_b[uid]

        if a["priority"] == b["priority"] and a["route"] == b["route"]:
            priority, route, rule = a["priority"], a["route"], a["rule"]
            rulebook_version = a["rulebook_version"]
            label_source = "agree"
        elif pool in ADJUDICATED_POOLS:
            decision, priority, route, rule, rulebook_version, adjudicator = adjudication[uid]
            if decision == "DROP":
                dropped_reasons["adjudicated_drop"] += 1
                continue
            label_source = f"adjudicated:{adjudicator}"
        else:
            priority, route, rule = b["priority"], b["route"], b["rule"]
            rulebook_version = b["rulebook_version"]
            label_source = "passB-default"
            passb_default_count += 1

        override = overrides.get(uid)
        if override is not None:
            if override.get("new_priority") is not None:
                priority = override["new_priority"]
            if override.get("new_route") is not None:
                route = override["new_route"]
            if override.get("new_rule") is not None:
                rule = override["new_rule"]
            label_source = f"{label_source}+override"
            overrides_applied += 1

        output.append({
            "uid": uid,
            "pool": pool,
            "source": meta["source"],
            "group_id": meta["group_id"],
            "text": meta["text"],
            "priority": priority,
            "route": route,
            "rule": rule,
            "label_source": label_source,
            "rulebook_version": rulebook_version,
        })
        kept_per_pool[pool] += 1
        group_pools[meta["group_id"]].add(pool)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in output:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(output)} records to {OUTPUT_PATH}\n")

    print("=== Kept per pool ===")
    for pool in ("train", "val", "test"):
        print(f"  {pool}: {kept_per_pool[pool]}")

    print("\n=== Dropped, by reason ===")
    if dropped_reasons:
        for reason, count in dropped_reasons.items():
            print(f"  {reason}: {count}")
    else:
        print("  none")

    print(f"\n=== Train-pool disagreements given Pass B's label (passB-default): {passb_default_count} ===")

    print(f"\n=== Overrides applied: {overrides_applied} ===")

    print("\n=== Priority distribution per pool ===")
    for pool in ("train", "val", "test"):
        counts = Counter(r["priority"] for r in output if r["pool"] == pool)
        print(f"  {pool}: {dict(sorted(counts.items()))}")

    print("\n=== Route distribution per pool ===")
    for pool in ("train", "val", "test"):
        counts = Counter(r["route"] for r in output if r["pool"] == pool)
        print(f"  {pool}: {dict(sorted(counts.items()))}")

    all_test_val_uids = {m["uid"] for m in pool_records if m["pool"] in ADJUDICATED_POOLS}
    labelled_test_val_uids = {r["uid"] for r in output if r["pool"] in ADJUDICATED_POOLS}
    missing = all_test_val_uids - labelled_test_val_uids
    print(f"\n=== Every test/val uid labelled? {'YES' if not missing else 'NO'} ===")
    if missing:
        print(f"  missing ({len(missing)}): {sorted(missing)[:20]}{' ...' if len(missing) > 20 else ''}")

    multi_pool_groups = {g: pools for g, pools in group_pools.items() if len(pools) > 1}
    print(f"\n=== Any group_id in more than one pool? {'NO' if not multi_pool_groups else 'YES'} ===")
    if multi_pool_groups:
        for g, pools in list(multi_pool_groups.items())[:20]:
            print(f"  {g}: {sorted(pools)}")

    return output


if __name__ == "__main__":
    main()
