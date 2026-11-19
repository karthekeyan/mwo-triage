"""Compare two independent labelling passes: agreement, Cohen's kappa,
confusion matrices, and a disagreement table.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def cohen_kappa(pairs, labels):
    n = len(pairs)
    observed_agreement = sum(1 for a, b in pairs if a == b) / n

    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)
    expected_agreement = sum(a_counts[l] * b_counts[l] for l in labels) / (n * n)

    if expected_agreement == 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def print_confusion_matrix(pairs, labels, title):
    print(f"\n=== Confusion matrix: {title} (rows=A, cols=B) ===")
    counts = Counter(pairs)
    col_width = max(len(l) for l in labels) + 2
    header = " " * col_width + "".join(f"{l[:10]:>12}" for l in labels)
    print(header)
    for row_label in labels:
        row = f"{row_label:<{col_width}}"
        for col_label in labels:
            row += f"{counts.get((row_label, col_label), 0):>12}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--texts", required=True)
    args = parser.parse_args()

    a_records = {r["uid"]: r for r in load_jsonl(args.a)}
    b_records = {r["uid"]: r for r in load_jsonl(args.b)}
    text_records = {r["uid"]: r for r in load_jsonl(args.texts)}

    common_uids = sorted(set(a_records) & set(b_records))
    only_a = set(a_records) - set(b_records)
    only_b = set(b_records) - set(a_records)

    print(f"Pass A: {len(a_records)} records, Pass B: {len(b_records)} records")
    print(f"Common uids: {len(common_uids)}; only in A: {len(only_a)}; only in B: {len(only_b)}")

    priority_pairs = [(a_records[u]["priority"], b_records[u]["priority"]) for u in common_uids]
    route_pairs = [(a_records[u]["route"], b_records[u]["route"]) for u in common_uids]
    both_pairs = list(zip(priority_pairs, route_pairs))

    n = len(common_uids)
    priority_agree = sum(1 for a, b in priority_pairs if a == b)
    route_agree = sum(1 for a, b in route_pairs if a == b)
    both_agree = sum(1 for (pa, pb), (ra, rb) in both_pairs if pa == pb and ra == rb)

    print("\n=== Agreement ===")
    print(f"  priority: {priority_agree}/{n} = {priority_agree / n:.1%}")
    print(f"  route:    {route_agree}/{n} = {route_agree / n:.1%}")
    print(f"  both:     {both_agree}/{n} = {both_agree / n:.1%}")

    priority_labels = sorted({l for pair in priority_pairs for l in pair})
    route_labels = sorted({l for pair in route_pairs for l in pair})

    print("\n=== Cohen's kappa ===")
    print(f"  priority: {cohen_kappa(priority_pairs, priority_labels):.3f}")
    print(f"  route:    {cohen_kappa(route_pairs, route_labels):.3f}")

    print_confusion_matrix(priority_pairs, priority_labels, "priority")
    print_confusion_matrix(route_pairs, route_labels, "route")

    print("\n=== Disagreements ===")
    disagreements = [
        u for u in common_uids
        if a_records[u]["priority"] != b_records[u]["priority"]
        or a_records[u]["route"] != b_records[u]["route"]
    ]
    print(f"  {len(disagreements)} of {n} texts disagree on priority and/or route\n")

    for u in disagreements:
        a, b = a_records[u], b_records[u]
        text = text_records.get(u, {}).get("text", "<text not found>")
        print(f"  uid={u}")
        print(f"    text: {text!r}")
        print(f"    A: priority={a['priority']} route={a['route']} rule={a['rule']}")
        print(f"    B: priority={b['priority']} route={b['route']} rule={b['rule']}")
        print()


if __name__ == "__main__":
    main()
