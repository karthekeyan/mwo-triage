"""Blind-label check for data/interim/synthetic_raw.jsonl.

Commands:
  export   Write data/interim/blind_sheet.jsonl: uid and text only, in a
           fixed random order (seed 7), with no target labels -- for
           labelling blind in a fresh session that doesn't remember the
           targets.
  compare  Read data/interim/blind_labels.jsonl ({uid, priority, route,
           rule}), compare priority and route against target_priority/
           target_route in data/interim/synthetic_raw.jsonl, and report
           agreement (priority, route, both) overall and per block, per
           cell (priority x route, for topup/factory) and per theme (for
           hard_case). Writes data/interim/synthetic_verified.jsonl with
           the synthetic_raw.jsonl records where both labels agree, each
           tagged with the blind rule code that was given for it.

No API calls.

Usage:
  python scripts/blind_check.py export
  python scripts/blind_check.py compare
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SYNTHETIC_PATH = Path("data/interim/synthetic_raw.jsonl")
BLIND_SHEET_PATH = Path("data/interim/blind_sheet.jsonl")
BLIND_LABELS_PATH = Path("data/interim/blind_labels.jsonl")
VERIFIED_PATH = Path("data/interim/synthetic_verified.jsonl")

EXPORT_SEED = 7


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


def cmd_export(args):
    records = load_jsonl(SYNTHETIC_PATH)
    sheet = [{"uid": r["uid"], "text": r["text"]} for r in records]
    random.Random(EXPORT_SEED).shuffle(sheet)

    BLIND_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BLIND_SHEET_PATH, "w", encoding="utf-8") as f:
        for row in sheet:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(sheet)} texts (uid + text only, no labels) to {BLIND_SHEET_PATH}")


def group_key(record):
    if record["block"] == "hard_case":
        return ("hard_case", record["theme"])
    return (record["block"], record["target_priority"], record["target_route"])


def group_label(key):
    if key[0] == "hard_case":
        return f"hard_case theme={key[1]}"
    return f"{key[0]:8s} cell={key[1]}/{key[2]}"


def pct(n, d):
    return f"{n}/{d} ({n / d:.1%})" if d else f"{n}/{d}"


def report_group(name, rows):
    total = len(rows)
    priority_ok = sum(1 for r in rows if r["priority_match"])
    route_ok = sum(1 for r in rows if r["route_match"])
    both_ok = sum(1 for r in rows if r["priority_match"] and r["route_match"])
    print(f"  {name}")
    print(f"    priority: {pct(priority_ok, total)}  route: {pct(route_ok, total)}  both: {pct(both_ok, total)}")


def cmd_compare(args):
    synthetic = {r["uid"]: r for r in load_jsonl(SYNTHETIC_PATH)}
    blind = load_jsonl(BLIND_LABELS_PATH)

    if not blind:
        print(f"{BLIND_LABELS_PATH} is empty or missing; nothing to compare.")
        return

    missing = [b["uid"] for b in blind if b["uid"] not in synthetic]
    if missing:
        raise ValueError(f"{len(missing)} blind-label uids are not in {SYNTHETIC_PATH}: {missing[:10]}...")

    seen = set()
    dup = [b["uid"] for b in blind if b["uid"] in seen or seen.add(b["uid"])]
    if dup:
        raise ValueError(f"duplicate uids in {BLIND_LABELS_PATH}: {dup[:10]}...")

    rows = []
    for b in blind:
        target = synthetic[b["uid"]]
        rows.append({
            "uid": b["uid"],
            "block": target["block"],
            "theme": target.get("theme"),
            "target_priority": target["target_priority"],
            "target_route": target["target_route"],
            "blind_priority": b["priority"],
            "blind_route": b["route"],
            "blind_rule": b.get("rule"),
            "priority_match": b["priority"] == target["target_priority"],
            "route_match": b["route"] == target["target_route"],
        })

    synthetic_uids = set(synthetic)
    blind_uids = {r["uid"] for r in rows}
    not_labelled = synthetic_uids - blind_uids
    if not_labelled:
        print(f"Note: {len(not_labelled)} of {len(synthetic_uids)} synthetic texts have no blind label yet "
              f"(comparing the {len(rows)} that do).")

    print(f"\n=== Overall ({len(rows)} compared) ===")
    report_group("overall", rows)

    print("\n=== Per block ===")
    by_block = defaultdict(list)
    for r in rows:
        by_block[r["block"]].append(r)
    for block in sorted(by_block):
        report_group(block, by_block[block])

    print("\n=== Per cell (topup/factory) and per theme (hard_case) ===")
    by_group = defaultdict(list)
    for r in rows:
        key = ("hard_case", r["theme"]) if r["block"] == "hard_case" else (r["block"], r["target_priority"], r["target_route"])
        by_group[key].append(r)
    for key in sorted(by_group, key=lambda k: (k[0], str(k[1]), str(k[2]) if len(k) > 2 else "")):
        report_group(group_label(key), by_group[key])

    verified = [{**synthetic[r["uid"]], "blind_rule": r["blind_rule"]}
                for r in rows if r["priority_match"] and r["route_match"]]
    with open(VERIFIED_PATH, "w", encoding="utf-8") as f:
        for r in verified:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(verified)} verified records (both labels agree) to {VERIFIED_PATH}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export")
    sub.add_parser("compare")
    args = parser.parse_args()

    if args.command == "export":
        cmd_export(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
