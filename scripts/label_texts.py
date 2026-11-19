"""Label work-order texts with Claude, following docs/labelling-rulebook.md.

Sync mode sends one tool-forced request per batch, validates the response
against the rulebook's label enums, and appends results to --output so a run
can resume. Batch mode submits every batch through the Message Batches API,
polls until it ends, validates and appends the results the same way, and
resends any failed or invalid batches in sync mode.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from anthropic import Anthropic, BadRequestError
from dotenv import load_dotenv

RULEBOOK_PATH = Path("docs/labelling-rulebook.md")
RULEBOOK_VERSION = "1.3"

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
    "TB8": "Tie-break rule 8: calibrate/program/set/adjust a sensor, controller or electronic brake control -> INSTRUMENTATION_CONTROLS; a purely mechanical adjustment goes to the owner team, usually MECHANICAL",
}

# Models known to reject a forced tool_choice (they only support "auto").
# Discovered via sync mode's BadRequestError fallback; listed here so batch
# mode can pick the right tool_choice up front instead of paying for a
# guaranteed-to-error batch request.
AUTO_ONLY_MODELS = {"claude-sonnet-5-5"}

MAX_RETRIES = 3

PASS_SEEDS = {"A": 1001, "B": 1002}
SAMPLE_SEED = 42

BATCH_POLL_SECONDS = 60

# $ per million tokens, (input, output). Cache reads are 5% of the input price.
PRICING_SYNC = {
    "sonnet": (2.0, 10.0),
    "opus": (4.0, 20.0),
}

# Message Batches API is priced at half the sync rate (plus Fable, batch-only
# so far). Cache reads are still 5% of the input price.
PRICING_BATCH = {
    "sonnet": (1.0, 5.0),
    "opus": (2.0, 10.0),
    "fable": (4.0, 20.0),
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


def pricing_for_model(model, rates):
    model_lower = model.lower()
    for key, prices in rates.items():
        if key in model_lower:
            return prices
    raise ValueError(f"No pricing known for model '{model}'. Add it to the pricing table.")


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


def request_kwargs(model, system_blocks, batch, tool_choice):
    """Build request parameters identically for sync calls and batch requests."""
    return {
        "model": model,
        "max_tokens": 8192,
        "system": system_blocks,
        "tools": [SUBMIT_LABELS_TOOL],
        "tool_choice": tool_choice,
        "messages": [{"role": "user", "content": build_user_message(batch)}],
    }


def initial_tool_choice(model):
    if model in AUTO_ONLY_MODELS:
        return {"type": "auto"}
    return {"type": "tool", "name": "submit_labels"}


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
    """Sync mode: one request per batch using the model's known-good
    tool_choice, retried up to MAX_RETRIES times. Still falls back from a
    forced tool_choice to "auto" on a BadRequestError, in case a model not
    yet listed in AUTO_ONLY_MODELS turns out to need it too."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                **request_kwargs(model, system_blocks, batch, initial_tool_choice(model))
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
                **request_kwargs(model, system_blocks, batch, {"type": "auto"})
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


def append_results(out_path, batch, by_uid, model, pass_name):
    with open(out_path, "a", encoding="utf-8") as f:
        for record in batch:
            label = by_uid[record["uid"]]
            out_record = dict(record)
            out_record["priority"] = label["priority"]
            out_record["route"] = label["route"]
            out_record["rule"] = label["rule"]
            out_record["model"] = model
            out_record["pass"] = pass_name
            out_record["rulebook_version"] = RULEBOOK_VERSION
            f.write(json.dumps(out_record) + "\n")


def usage_totals(usage_log):
    return {
        "input": sum(u.input_tokens for u in usage_log),
        "output": sum(u.output_tokens for u in usage_log),
        "cache_read": sum(getattr(u, "cache_read_input_tokens", 0) or 0 for u in usage_log),
        "cache_write": sum(getattr(u, "cache_creation_input_tokens", 0) or 0 for u in usage_log),
    }


def cost_for_usage(totals, model, rates):
    input_price, output_price = pricing_for_model(model, rates)
    cache_read_price = input_price * 0.05
    return (
        (totals["input"] + totals["cache_write"]) / 1_000_000 * input_price
        + totals["cache_read"] / 1_000_000 * cache_read_price
        + totals["output"] / 1_000_000 * output_price
    )


def print_usage_and_cost(model, *priced_usage_logs):
    """priced_usage_logs: one or more (usage_log, rates, label) groups, each
    priced at its own rates (e.g. batch-priced requests vs. sync-mode
    retries), so mixing modes never bills sync tokens at batch rates or
    vice versa."""
    grand_totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    total_requests = 0
    total_cost = 0.0

    print("\n=== Token usage ===")
    for usage_log, rates, label in priced_usage_logs:
        if not usage_log:
            continue
        totals = usage_totals(usage_log)
        cost = cost_for_usage(totals, model, rates)
        total_requests += len(usage_log)
        total_cost += cost
        for key in grand_totals:
            grand_totals[key] += totals[key]

        print(f"  [{label}] requests: {len(usage_log)}")
        print(f"  [{label}] input tokens (non-cached): {totals['input']}")
        print(f"  [{label}] cache write tokens: {totals['cache_write']}")
        print(f"  [{label}] cache read tokens: {totals['cache_read']}")
        print(f"  [{label}] output tokens: {totals['output']}")
        print(f"  [{label}] cost: ${cost:.4f}")

    print(f"  requests: {total_requests}")
    print(f"  input tokens (non-cached): {grand_totals['input']}")
    print(f"  cache write tokens: {grand_totals['cache_write']}")
    print(f"  cache read tokens: {grand_totals['cache_read']}")
    print(f"  output tokens: {grand_totals['output']}")
    print(f"  estimated cost: ${total_cost:.4f}")


def build_batches(records, output_path, batch_size, pass_name, limit):
    if limit is not None and limit < len(records):
        records = random.Random(SAMPLE_SEED).sample(records, limit)

    already_done = {r["uid"] for r in load_jsonl(output_path)}
    todo = [r for r in records if r["uid"] not in already_done]
    print(f"Loaded {len(records)} records, {len(already_done)} already labelled, {len(todo)} to do.")

    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    random.Random(PASS_SEEDS[pass_name]).shuffle(batches)
    return batches


def run_sync(client, model, system_blocks, batches, out_path, pass_name):
    usage_log = []
    for batch_idx, batch in enumerate(batches, start=1):
        print(f"Batch {batch_idx}/{len(batches)} ({len(batch)} texts)...")
        try:
            by_uid = call_batch(client, model, system_blocks, batch, usage_log)
        except RuntimeError as exc:
            print(f"  SKIPPING batch: {exc}")
            continue
        append_results(out_path, batch, by_uid, model, pass_name)
    return usage_log


def state_path_for(output_path):
    return Path(str(output_path) + ".batch_state.json")


def submit_batch(client, model, system_blocks, batches, state_path):
    requests = []
    batch_map = {}
    for idx, batch in enumerate(batches):
        custom_id = f"batch-{idx:04d}"
        batch_map[custom_id] = batch
        requests.append({
            "custom_id": custom_id,
            "params": request_kwargs(model, system_blocks, batch, initial_tool_choice(model)),
        })

    message_batch = client.messages.batches.create(requests=requests)
    print(f"Submitted batch {message_batch.id} with {len(requests)} requests.")

    state = {
        "batch_id": message_batch.id,
        "model": model,
        "batches": {cid: records for cid, records in batch_map.items()},
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    print(f"Saved batch state to {state_path}")
    return state


def poll_batch(client, batch_id):
    while True:
        message_batch = client.messages.batches.retrieve(batch_id)
        counts = message_batch.request_counts
        print(
            f"  status={message_batch.processing_status} "
            f"processing={counts.processing} succeeded={counts.succeeded} "
            f"errored={counts.errored} canceled={counts.canceled} expired={counts.expired}"
        )
        if message_batch.processing_status == "ended":
            return message_batch
        time.sleep(BATCH_POLL_SECONDS)


def collect_batch_results(client, batch_id, batch_map, out_path, model, pass_name, usage_log):
    """Validate each batch's result; return the list of batches (as record
    lists) that failed or came back invalid, for sync retry."""
    results_by_custom_id = {}
    for line in client.messages.batches.results(batch_id):
        results_by_custom_id[line.custom_id] = line.result

    failed_batches = []
    for custom_id, batch in batch_map.items():
        result = results_by_custom_id.get(custom_id)
        if result is None or result.type != "succeeded":
            reason = result.type if result is not None else "missing from results"
            print(f"  {custom_id}: {reason}, will retry in sync mode")
            failed_batches.append(batch)
            continue

        message = result.message
        usage_log.append(message.usage)
        tool_use = next((b for b in message.content if b.type == "tool_use"), None)
        if tool_use is None:
            print(f"  {custom_id}: no tool_use block, will retry in sync mode")
            failed_batches.append(batch)
            continue

        labels = tool_use.input.get("labels", [])
        by_uid, error = validate_labels(batch, labels)
        if error is not None:
            print(f"  {custom_id}: invalid ({error}), will retry in sync mode")
            failed_batches.append(batch)
            continue

        append_results(out_path, batch, by_uid, model, pass_name)

    return failed_batches


def run_batch(client, model, system_blocks, batches, out_path, pass_name, state_path, resume_only):
    if resume_only:
        if not state_path.exists():
            raise FileNotFoundError(f"--resume-batch given but no state file at {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        batch_id = state["batch_id"]
        batch_map = state["batches"]
        print(f"Resuming batch {batch_id} from {state_path}")
    else:
        state = submit_batch(client, model, system_blocks, batches, state_path)
        batch_id = state["batch_id"]
        batch_map = state["batches"]

    poll_batch(client, batch_id)

    batch_usage_log = []
    failed_batches = collect_batch_results(client, batch_id, batch_map, out_path, model, pass_name, batch_usage_log)

    sync_retry_usage_log = []
    if failed_batches:
        print(f"\nRetrying {len(failed_batches)} failed/invalid batch(es) in sync mode...")
        sync_retry_usage_log = run_sync(client, model, system_blocks, failed_batches, out_path, pass_name)

    return batch_usage_log, sync_retry_usage_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pass-name", required=True, choices=["A", "B"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--mode", choices=["sync", "batch"], default="sync")
    parser.add_argument("--resume-batch", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    rulebook_text = RULEBOOK_PATH.read_text(encoding="utf-8")
    system_blocks = build_system_blocks(rulebook_text)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.input)

    if args.mode == "sync":
        batches = build_batches(records, out_path, args.batch_size, args.pass_name, args.limit)
        usage_log = run_sync(client, args.model, system_blocks, batches, out_path, args.pass_name)
        print_usage_and_cost(args.model, (usage_log, PRICING_SYNC, "sync"))
    else:
        state_path = state_path_for(out_path)
        if args.resume_batch:
            batches = None
        else:
            batches = build_batches(records, out_path, args.batch_size, args.pass_name, args.limit)
        batch_usage_log, sync_retry_usage_log = run_batch(
            client, args.model, system_blocks, batches, out_path, args.pass_name, state_path, args.resume_batch
        )
        print_usage_and_cost(
            args.model,
            (batch_usage_log, PRICING_BATCH, "batch"),
            (sync_retry_usage_log, PRICING_SYNC, "sync retry"),
        )


if __name__ == "__main__":
    main()
