"""Label work-order texts with Claude, following docs/labelling-rulebook.md.

Sends one tool-forced request per batch, validates the response against the
rulebook's label enums, and appends results to --output so a run can resume.
"""

import argparse
import json
import os
import random
from pathlib import Path

from anthropic import Anthropic, BadRequestError
from dotenv import load_dotenv

RULEBOOK_PATH = Path("docs/labelling-rulebook.md")
RULEBOOK_VERSION = "1.2"

PRIORITIES = ["STOP_NOW", "TODAY", "PLAN", "MONITOR"]
ROUTES = [
    "MECHANICAL",
    "ELECTRICAL",
    "HYDRAULIC_PNEUMATIC",
    "INSTRUMENTATION_CONTROLS",
    "LUBRICATION",
    "ESCALATE",
]

RULE_CODES = {
    "P1-SAFETY": "P1 safety trigger (fire, shock, hazardous leak, spraying/burst/hot-surface/people-area leak, safety device missing/damaged/bypassed, brake/steering failure, uncontrolled movement, load-bearing crack)",
    "P1-LINE-DOWN": "P1 production/asset trigger: line or critical machine explicitly stopped, no workaround",
    "P1-WORSENING": "P1 production/asset trigger: rapidly worsening failure (severe/rising vibration, knocking/grinding with rising temp, repeated trip, getting worse fast)",
    "P2": "P2 TODAY: degraded but running, or down with a workaround; leak not spraying/hot/people-area; abnormal but not severe noise/vibration; intermittent fault; running hot within limits; quality rejects; breakdown/u/s on one machine without line-down",
    "P3": "P3 PLAN: real defect, no immediate safety/production/quality effect (wear found, minor weep, loose non-safety fastener, routine replacement, repair request on still-working item)",
    "P4": "P4 MONITOR: no repair action yet, observation/trend/within-tolerance inspection/check with no symptom",
    "R-COMPONENT": "Route decided directly by the component/system table in section 4",
    "ESC-VAGUE": "ESCALATE: too vague, no component or symptom given",
    "ESC-MULTI": "ESCALATE: several disciplines equally likely, text doesn't say which",
    "ESC-NONMAINT": "ESCALATE: not a maintenance job (IT, cleaning, HR, purchasing, general question)",
    "ESC-UNREADABLE": "ESCALATE: unreadable or meaningless text, only codes, only an asset number",
    "TB1": "Tie-break rule 1: safety wins",
    "TB2": "Tie-break rule 2: repair target decides the route",
    "TB3": "Tie-break rule 3: diagnose-first discipline by symptom",
    "TB4": "Tie-break rule 4: faulty reading vs real problem",
    "TB5": "Tie-break rule 5: lubrication vs leak repair",
    "TB6": "Tie-break rule 6: leak location decides P1 vs P2",
    "TB7": "Tie-break rule 7: still unclear after rules 1-6, route ESCALATE",
    "TB8": "Tie-break rule 8: calibrate/program/set/adjust applies only to electronic brake controls, sensors or calibration -> INSTRUMENTATION_CONTROLS; a plain mechanical brake adjustment goes to MECHANICAL",
}

MAX_RETRIES = 3

PASS_SEEDS = {"A": 1001, "B": 1002}
SAMPLE_SEED = 42

# $ per million tokens, (input, output). Cache reads are 5% of the input price.
PRICING = {
    "sonnet": (2.0, 10.0),
    "opus": (4.0, 20.0),
}

SUBMIT_LABELS_TOOL = {
    "name": "submit_labels",
    "description": "Submit priority and route labels for every work order in the batch.",
    "input_schema": {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "uid": {"type": "string"},
                        "priority": {"type": "string", "enum": PRIORITIES},
                        "route": {"type": "string", "enum": ROUTES},
                        "rule": {
                            "type": "string",
                            "maxLength": 12,
                            "enum": list(RULE_CODES.keys()),
                            "description": "The rule code (from the system prompt's code list) that decided this label. Code only, no explanation.",
                        },
                    },
                    "required": ["uid", "priority", "route", "rule"],
                },
            }
        },
        "required": ["labels"],
    },
}

RULE_CODE_TABLE = "\n".join(f"- {code}: {meaning}" for code, meaning in RULE_CODES.items())

INSTRUCTIONS = f"""
You are labelling maintenance work orders for the mwo-triage dataset.

Label each work order strictly by the rulebook above. Use only what is
written in the text; do not assume facts that are not there. Ignore urgency
words ("urgent", "ASAP", "critical", "!!!") -- they do not change priority by
themselves.

For every work order, call the submit_labels tool with exactly one entry per
uid, in any order, giving priority, route, and a rule code. The rule code
must be exactly one of the following (code only, no explanation text):

{RULE_CODE_TABLE}
""".strip()


def load_jsonl(path):
    records = []
    if not Path(path).exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def pricing_for_model(model):
    model_lower = model.lower()
    for key, prices in PRICING.items():
        if key in model_lower:
            return prices
    raise ValueError(f"No pricing known for model '{model}'. Add it to PRICING.")


def build_system_blocks(rulebook_text):
    return [
        {
            "type": "text",
            "text": rulebook_text,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": INSTRUCTIONS,
        },
    ]


def build_user_message(batch):
    lines = [f"{i + 1}. uid={r['uid']} text={r['text']!r}" for i, r in enumerate(batch)]
    return "Label each of the following work orders:\n\n" + "\n".join(lines)


def validate_labels(batch, labels):
    expected_uids = {r["uid"] for r in batch}
    seen_uids = set()
    by_uid = {}
    for item in labels:
        uid = item.get("uid")
        if uid is None or uid in seen_uids:
            return None, f"duplicate or missing uid: {uid!r}"
        seen_uids.add(uid)
        if item.get("priority") not in PRIORITIES:
            return None, f"invalid priority for uid {uid}: {item.get('priority')!r}"
        if item.get("route") not in ROUTES:
            return None, f"invalid route for uid {uid}: {item.get('route')!r}"
        if item.get("rule") not in RULE_CODES:
            return None, f"invalid rule code for uid {uid}: {item.get('rule')!r}"
        by_uid[uid] = item
    if seen_uids != expected_uids:
        missing = expected_uids - seen_uids
        extra = seen_uids - expected_uids
        return None, f"uid mismatch: missing={missing} extra={extra}"
    return by_uid, None


def call_batch(client, model, system_blocks, batch, usage_log):
    message = build_user_message(batch)
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                system=system_blocks,
                tools=[SUBMIT_LABELS_TOOL],
                tool_choice={"type": "tool", "name": "submit_labels"},
                messages=[{"role": "user", "content": message}],
            )
        except BadRequestError as exc:
            if "tool_choice" not in str(exc):
                raise
            # Some models (e.g. extended-thinking models) reject forced
            # tool_choice and only support "auto"; instructions already
            # direct the model to call submit_labels. These models also
            # spend output budget on a thinking block before the tool call,
            # so give them extra headroom (seen: ~3k thinking tokens for a
            # 25-item batch) or the tool-call JSON gets truncated mid-array.
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                system=system_blocks,
                tools=[SUBMIT_LABELS_TOOL],
                tool_choice={"type": "auto"},
                messages=[{"role": "user", "content": message}],
            )
        usage_log.append(response.usage)

        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is None:
            last_error = "no tool_use block in response"
            continue

        labels = tool_use.input.get("labels", [])
        by_uid, error = validate_labels(batch, labels)
        if error is None:
            return by_uid
        last_error = error
        print(f"    attempt {attempt}/{MAX_RETRIES} failed validation: {last_error}")

    raise RuntimeError(f"batch failed after {MAX_RETRIES} attempts: {last_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pass-name", required=True, choices=["A", "B"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()

    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    rulebook_text = RULEBOOK_PATH.read_text(encoding="utf-8")
    system_blocks = build_system_blocks(rulebook_text)

    records = load_jsonl(args.input)

    if args.limit is not None and args.limit < len(records):
        records = random.Random(SAMPLE_SEED).sample(records, args.limit)

    already_done = {r["uid"] for r in load_jsonl(args.output)}
    todo = [r for r in records if r["uid"] not in already_done]
    print(f"Loaded {len(records)} records, {len(already_done)} already labelled, {len(todo)} to do.")

    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    random.Random(PASS_SEEDS[args.pass_name]).shuffle(batches)

    usage_log = []
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(batches, start=1):
        print(f"Batch {batch_idx}/{len(batches)} ({len(batch)} texts)...")
        try:
            by_uid = call_batch(client, args.model, system_blocks, batch, usage_log)
        except RuntimeError as exc:
            print(f"  SKIPPING batch: {exc}")
            continue

        with open(out_path, "a", encoding="utf-8") as f:
            for record in batch:
                label = by_uid[record["uid"]]
                out_record = dict(record)
                out_record["priority"] = label["priority"]
                out_record["route"] = label["route"]
                out_record["rule"] = label["rule"]
                out_record["model"] = args.model
                out_record["pass"] = args.pass_name
                out_record["rulebook_version"] = RULEBOOK_VERSION
                f.write(json.dumps(out_record) + "\n")

    total_input = sum(u.input_tokens for u in usage_log)
    total_output = sum(u.output_tokens for u in usage_log)
    total_cache_read = sum(getattr(u, "cache_read_input_tokens", 0) or 0 for u in usage_log)
    total_cache_write = sum(getattr(u, "cache_creation_input_tokens", 0) or 0 for u in usage_log)

    input_price, output_price = pricing_for_model(args.model)
    cache_read_price = input_price * 0.05

    cost = (
        (total_input + total_cache_write) / 1_000_000 * input_price
        + total_cache_read / 1_000_000 * cache_read_price
        + total_output / 1_000_000 * output_price
    )

    print("\n=== Token usage ===")
    print(f"  requests: {len(usage_log)}")
    print(f"  input tokens (non-cached): {total_input}")
    print(f"  cache write tokens: {total_cache_write}")
    print(f"  cache read tokens: {total_cache_read}")
    print(f"  output tokens: {total_output}")
    print(f"  estimated cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
