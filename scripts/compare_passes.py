"""Compare two independent labelling passes: agreement, Cohen's kappa,
confusion matrices, label distributions, and disagreements (overall and,
when --texts records carry a "pool" field, broken down by pool).
"""

import argparse
import json
import random
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
    if n == 0:
        return float("nan")
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


def print_agreement_and_kappa(label, priority_pairs, route_pairs, both_pairs):
    n = len(priority_pairs)
    if n == 0:
        print(f"\n=== {label}: no records ===")
        return

    priority_agree = sum(1 for a, b in priority_pairs if a == b)
    route_agree = sum(1 for a, b in route_pairs if a == b)
    both_agree = sum(1 for (pa, pb), (ra, rb) in both_pairs if pa == pb and ra == rb)

    priority_labels = sorted({l for pair in priority_pairs for l in pair})
    route_labels = sorted({l for pair in route_pairs for l in pair})

    print(f"\n=== {label} (n={n}) ===")
    print(f"  priority agreement: {priority_agree}/{n} = {priority_agree / n:.1%}  kappa={cohen_kappa(priority_pairs, priority_labels):.3f}")
    print(f"  route agreement:    {route_agree}/{n} = {route_agree / n:.1%}  kappa={cohen_kappa(route_pairs, route_labels):.3f}")
    print(f"  both agreement:     {both_agree}/{n} = {both_agree / n:.1%}")


def print_label_distribution(label, records):
    priority_counts = Counter(r["priority"] for r in records)
    route_counts = Counter(r["route"] for r in records)
    print(f"\n=== Label distribution: {label} ===")
    print("  priority:", dict(sorted(priority_counts.items())))
    print("  route:   ", dict(sorted(route_counts.items())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--texts", required=True)
    parser.add_argument("--sample-disagreements", type=int, default=None,
                         help="Print only a random sample of this many disagreements instead of the full table.")
    parser.add_argument("--save-disagreements", default=None,
                         help="Write the full disagreement list (as jsonl) to this path.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    a_records = {r["uid"]: r for r in load_jsonl(args.a)}
    b_records = {r["uid"]: r for r in load_jsonl(args.b)}
    text_records = {r["uid"]: r for r in load_jsonl(args.texts)}

    common_uids = sorted(set(a_records) & set(b_records))
    only_a = set(a_records) - set(b_records)
    only_b = set(b_records) - set(a_records)

    print(f"Pass A: {len(a_records)} records, Pass B: {len(b_records)} records")
    print(f"Common uids: {len(common_uids)}; only in A: {len(only_a)}; only in B: {len(only_b)}")

    def pool_of(uid):
        return text_records.get(uid, {}).get("pool", "<unknown>")

    pools = sorted({pool_of(u) for u in common_uids})

    def pairs_for(uids):
        priority_pairs = [(a_records[u]["priority"], b_records[u]["priority"]) for u in uids]
        route_pairs = [(a_records[u]["route"], b_records[u]["route"]) for u in uids]
        both_pairs = list(zip(priority_pairs, route_pairs))
        return priority_pairs, route_pairs, both_pairs

    overall_priority_pairs, overall_route_pairs, overall_both_pairs = pairs_for(common_uids)
    print_agreement_and_kappa("Overall", overall_priority_pairs, overall_route_pairs, overall_both_pairs)

    if len(pools) > 1 or pools != ["<unknown>"]:
        for pool in pools:
            pool_uids = [u for u in common_uids if pool_of(u) == pool]
            pp, rp, bp = pairs_for(pool_uids)
            print_agreement_and_kappa(f"Pool: {pool}", pp, rp, bp)

    priority_labels = sorted({l for pair in overall_priority_pairs for l in pair})
    route_labels = sorted({l for pair in overall_route_pairs for l in pair})
    print_confusion_matrix(overall_priority_pairs, priority_labels, "priority")
    print_confusion_matrix(overall_route_pairs, route_labels, "route")

    print_label_distribution("Pass A", [a_records[u] for u in common_uids])
    print_label_distribution("Pass B", [b_records[u] for u in common_uids])

    disagreements = [
        u for u in common_uids
        if a_records[u]["priority"] != b_records[u]["priority"]
        or a_records[u]["route"] != b_records[u]["route"]
    ]
    print(f"\n=== Disagreements ===")
    print(f"  {len(disagreements)} of {len(common_uids)} texts disagree on priority and/or route")

    if args.save_disagreements:
        out_path = Path(args.save_disagreements)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for u in disagreements:
                a, b = a_records[u], b_records[u]
                text = text_records.get(u, {}).get("text", "<text not found>")
                pool = pool_of(u)
                record = {
                    "uid": u,
                    "pool": pool,
                    "text": text,
                    "a_priority": a["priority"],
                    "a_route": a["route"],
                    "a_rule": a["rule"],
                    "b_priority": b["priority"],
                    "b_route": b["route"],
                    "b_rule": b["rule"],
                }
                f.write(json.dumps(record) + "\n")
        print(f"  saved full list to {out_path}")

    to_print = disagreements
    sample_label = "all"
    if args.sample_disagreements is not None and args.sample_disagreements < len(disagreements):
        to_print = random.Random(args.seed).sample(disagreements, args.sample_disagreements)
        sample_label = f"random sample of {args.sample_disagreements}"

    print(f"\n=== Disagreements shown ({sample_label}) ===\n")
    for u in to_print:
        a, b = a_records[u], b_records[u]
        text = text_records.get(u, {}).get("text", "<text not found>")
        print(f"  uid={u} pool={pool_of(u)}")
        print(f"    text: {text!r}")
        print(f"    A: priority={a['priority']} route={a['route']} rule={a['rule']}")
        print(f"    B: priority={b['priority']} route={b['route']} rule={b['rule']}")
        print()


if __name__ == "__main__":
    main()
