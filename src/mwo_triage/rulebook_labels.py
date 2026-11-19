"""Label enums and rule codes from docs/labelling-rulebook.md (v1.3), shared
by every script that asks Claude to apply the rulebook (labelling,
adjudication, ...).
"""

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

RULE_CODE_TABLE = "\n".join(f"- {code}: {meaning}" for code, meaning in RULE_CODES.items())
