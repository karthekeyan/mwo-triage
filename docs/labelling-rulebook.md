# mwo-triage Labelling Rulebook

Version: 1.0 (approved 2026-11-19)
Applies to: every example in the mwo-triage dataset, real or synthetic.

## 1. Purpose

This rulebook decides two labels for each maintenance work-order text:

1. **Priority**: how fast someone must respond.
2. **Route**: which maintenance team should take the job.

Every label in the dataset must be reproducible from this rulebook alone. If two careful people (or two independent labelling passes) read the same text with this rulebook, they should reach the same labels. Where the rulebook is silent, we update the rulebook first and then relabel. We never make ad-hoc decisions.

**Honest limit:** this rulebook reflects general industrial maintenance practice. It has not been reviewed by a certified maintenance engineer. It is not a safety standard and must not replace a site's own safety procedures.

## 2. How to read a work order

- **Label the condition, not the wording.** Words like "urgent", "ASAP", "critical" or "!!!" do not change the priority by themselves. Priority comes from the condition the text describes. (This deliberately counters priority inflation, a known problem in real work orders.)
- **Treat every text as a request.** Many real work orders are written as completed or planned actions ("c/o hyd hose", "replaced belt"). Label them as if the action had been requested for the condition described.
- **Use only what is written.** Do not assume facts that aren't there. If the text says "leak" but not "spraying", it is not spraying.
- **If there are several faults, label the most severe one.** Priority and route both come from that fault.

## 3. Priority labels

Check the labels in this order: **Stop now → Today → Plan → Monitor**. Take the first one that matches.

### P1 `STOP_NOW`: immediate response

The job must be attended immediately, and the equipment may need to be stopped or isolated. Use it when **any** of the following is described:

**Safety triggers (always P1, no exceptions):**
- Fire, smoke, burning smell, sparks or arcing
- Electric shock, exposed live conductors, damaged insulation on live cables
- Leak of fuel, gas, chemicals or another hazardous substance
- Hydraulic or pneumatic fluid **spraying**, a hose **burst**, or a leak onto a **hot surface**, an **ignition source**, or a **walkway or work area where people are**
- Guard, interlock, emergency stop or safety device **missing, damaged, bypassed or not working**
- Brake or steering failure on mobile equipment (trucks, loaders, forklifts, cranes)
- Uncontrolled movement, a dropped load, or lifting equipment failure
- A crack, deformation or failed weld in a **load-bearing or pressure-containing** structure

**Production or asset triggers:**
- The text explicitly states that a **production line or critical machine is stopped** ("line down", "production stopped") and gives no workaround
- **Rapidly worsening** failure: severe or rising vibration, knocking or grinding with a rising temperature, a repeated protective trip, or a "getting worse fast" description

### P2 `TODAY`: fix within this shift or day

The equipment is degraded but still running, or is down with a workaround available. Without action it is likely to fail or affect production or quality within about a day.
- A leak that is dripping or steady but **not** spraying, **not** onto a hot surface, and **not** in a people area
- A pressure, speed or output drop; a cycle running slow
- Abnormal noise or vibration that is **not** described as severe or rising
- An intermittent fault, an occasional trip, or a nuisance alarm stopping work now and then
- Running hot but still within operating limits; a temperature alarm that has cleared
- Quality rejects caused by the equipment
- "Breakdown", "not working" or "u/s" on a single machine, **without** a statement that a line or critical machine is stopped

### P3 `PLAN`: schedule in the next planned window

A real defect with no immediate effect on safety, production or quality. It can wait for planned maintenance (days to weeks).
- Wear found during inspection (worn belt not yet slipping, worn pads above the minimum, worn bushes)
- A minor weep or seepage with no measurable fluid loss
- Loose or missing non-safety fasteners, covers or panels
- Routine component replacement (tyre change-out, filter change, scheduled part swap)
- A request to repair, replace or modify something that is still working

### P4 `MONITOR`: observe, no repair action yet

No defect requires work yet. The condition should be watched or rechecked.
- An observation or reading noted for trending ("slight noise noticed, recheck next PM")
- An inspection finding that is within tolerance
- A request to inspect, check or investigate with **no** symptom described
- Oil sample, thermography or vibration route results flagged as "watch" or "trend"

## 4. Route labels

Route by **the component or system that must be repaired**, not by the machine name or the symptom.

| Label | Team | Typical components and work |
|---|---|---|
| `MECHANICAL` | Mechanical fitters | Bearings, gears, gearboxes (internals), shafts, couplings, belts, chains, sprockets, pulleys, rollers, idlers, tyres and wheels, mechanical brakes, structures, welds, frames, guards, fasteners, mechanical seals on rotating parts, buckets, liners, wear plates |
| `ELECTRICAL` | Electricians | Electric motors (windings, terminals), power cables, breakers, fuses, contactors, relays, power supplies, VFD/drive power faults, batteries, alternators, starters, lighting, earthing, isolators, electrical panels |
| `HYDRAULIC_PNEUMATIC` | Hydraulic and pneumatic fitters | Hydraulic hoses, fittings, cylinders, valves, hydraulic pumps and motors, accumulators, hydraulic oil leaks and pressure faults, compressed-air leaks, air lines, pneumatic cylinders and valves, FRL units, air compressors |
| `INSTRUMENTATION_CONTROLS` | Instrument and controls technicians | Sensors, transmitters, gauges, limit switches, proximity switches, encoders, load cells, PLCs, HMIs, control panels (logic, not power), alarms that are false or faulty, calibration, control-loop tuning, machine communication or network faults, cameras and vision sensors |
| `LUBRICATION` | Lube technicians | Greasing, oil top-up, oil and grease changes, auto-lube or central-lube system faults, missed lube points, oil sampling, lubricant contamination |
| `ESCALATE` | Planner or supervisor review | See section 5 |

## 5. When to route `ESCALATE`

Use `ESCALATE` when the text **cannot** be routed reliably to a single team:
- **Too vague:** no component or symptom ("machine not working", "check machine 4", "problem with line 2")
- **Several disciplines, equally likely,** and the text doesn't say which ("conveyor stopped" with no cause or clue)
- **Not a maintenance job:** IT, cleaning, housekeeping, HR, purchasing, a general question
- **Unreadable or meaningless:** garbled text, only codes, only an asset number

`ESCALATE` is a route, not a priority. **A priority is still required.** Apply the priority rules to whatever the text does say. If a vague text describes a stopped line, it is still `STOP_NOW` + `ESCALATE`. If it says nothing about condition, use `TODAY` + `ESCALATE`, because a planner must look at it today.

## 6. Tie-break rules

Apply these in order when the labels are unclear.

1. **Safety wins.** If any P1 safety trigger is present, the priority is `STOP_NOW`, whatever else the text says.
2. **Repair target decides the route.** "Hydraulic pump motor won't start" → `ELECTRICAL` (the motor). "Hydraulic pump whining, pressure low" → `HYDRAULIC_PNEUMATIC` (the pump).
3. **Diagnose-first discipline.** If the cause is unknown but the symptom points to one discipline, route there: "no power to conveyor" → `ELECTRICAL`; "belt off track" → `MECHANICAL`.
4. **Faulty reading or real problem?** If the text says the reading is wrong, erratic or missing ("sensor reading zero", "gauge stuck") → `INSTRUMENTATION_CONTROLS`. If the reading is believed and shows a real fault ("bearing temp high 95C") → route by that component (`MECHANICAL`).
5. **Lubrication or leak repair?** Adding, changing or applying lubricant, or a lube system fault → `LUBRICATION`. A leak that needs a seal, gasket or hose repaired → the owner of that component (`MECHANICAL` for a gearbox seal, `HYDRAULIC_PNEUMATIC` for a hydraulic hose).
6. **Leak location decides between P1 and P2.** A leak alone is `TODAY`. A leak that is spraying, bursting, on a hot surface, near ignition or in a people area is `STOP_NOW`.
7. **Still unclear after rules 1–6:** route `ESCALATE` and note the reason in the review log. If this happens repeatedly for the same kind of text, update this rulebook.

## 7. Common abbreviations

| Abbreviation | Meaning |
|---|---|
| c/o | change out (replace) |
| u/s | unserviceable (not working) |
| o/s | out of service |
| hyd | hydraulic |
| elec | electrical |
| lhs / rhs | left-hand side / right-hand side |
| fr / rr | front / rear |
| brg | bearing |
| eng | engine |
| p/m, PM | preventive maintenance |
| insp | inspection |
| repl / rplc | replace |
| lkg | leaking |
| temp | temperature |
| vib | vibration |
| ad | air drive / air-driven (context-dependent; if unsure, apply rule 7) |

## 8. Worked examples

| # | Work-order text | Priority | Route | Reason |
|---|---|---|---|---|
| 1 | Hydraulic press 3 leaking oil near main cylinder, pressure dropping during cycle | `TODAY` | `HYDRAULIC_PNEUMATIC` | Steady leak with degrading pressure. Not spraying, not on a hot surface, not in a people area (rule 6) |
| 2 | Hyd hose burst on press 3, oil spraying across walkway | `STOP_NOW` | `HYDRAULIC_PNEUMATIC` | Burst hose, spraying, people area (safety trigger) |
| 3 | Oil leaking from hyd line onto exhaust manifold | `STOP_NOW` | `HYDRAULIC_PNEUMATIC` | Leak onto a hot surface (fire risk) |
| 4 | Smoke coming from conveyor 2 drive motor | `STOP_NOW` | `ELECTRICAL` | Smoke (safety trigger); the motor is the repair target |
| 5 | Light curtain on robot cell 4 not working, operator bypassed it | `STOP_NOW` | `INSTRUMENTATION_CONTROLS` | Safety device not working and bypassed |
| 6 | Line 1 stopped, main gearbox seized | `STOP_NOW` | `MECHANICAL` | Stopped line stated explicitly; gearbox internals |
| 7 | Gearbox on mixer noisy, slight vibration | `TODAY` | `MECHANICAL` | Abnormal noise, not severe or rising |
| 8 | Bearing temp on fan 2 reading 95C and rising, loud grinding | `STOP_NOW` | `MECHANICAL` | Rapidly worsening, temperature rising with grinding |
| 9 | Temp sensor on oven zone 3 reading 0, oven running normally | `TODAY` | `INSTRUMENTATION_CONTROLS` | Faulty reading (rule 4); process affected without a correct reading |
| 10 | CNC 5 spindle stops randomly, fault clears on reset | `TODAY` | `ESCALATE` | Intermittent fault; could be electrical, controls or mechanical (section 5) |
| 11 | c/o position 3 tyre | `PLAN` | `MECHANICAL` | Routine replacement |
| 12 | Grease all lube points on stacker, auto lube not pumping | `TODAY` | `LUBRICATION` | Lube system fault; running without lubrication will damage bearings |
| 13 | Gearbox oil level low, top up | `PLAN` | `LUBRICATION` | Top-up with no damage described |
| 14 | Gearbox output seal weeping, no oil loss | `PLAN` | `MECHANICAL` | Minor weep (P3); seal repair is mechanical (rule 5) |
| 15 | Compressed air leak at FRL unit on line 2, hissing | `TODAY` | `HYDRAULIC_PNEUMATIC` | Steady air leak, wasting energy, not hazardous |
| 16 | Emergency stop button on press 1 cracked but still works | `STOP_NOW` | `MECHANICAL` | Safety device damaged (rule 1); the repair is a physical replacement |
| 17 | Panel cover loose on MCC room door | `PLAN` | `ELECTRICAL` | Non-safety loose panel; electrical room |
| 18 | Slight noise on pump 4, recheck at next PM | `MONITOR` | `MECHANICAL` | Observation for trending only |
| 19 | Machine 7 not working | `TODAY` | `ESCALATE` | Too vague to route; a planner must look today |
| 20 | Please reset my email password | `PLAN` | `ESCALATE` | Not a maintenance job |
| 21 | URGENT!!! paint booth light flickering | `PLAN` | `ELECTRICAL` | Urgency words ignored (section 2); a flickering light is a minor defect |
| 22 | Forklift 2 brakes not holding on ramp | `STOP_NOW` | `MECHANICAL` | Brake failure on mobile equipment |
| 23 | VFD on conveyor 6 tripping on overcurrent twice today | `TODAY` | `ELECTRICAL` | Intermittent trip; drive power fault |
| 24 | HMI screen on line 3 frozen, machine still running | `TODAY` | `INSTRUMENTATION_CONTROLS` | Operator has lost visibility; machine still producing |

Note on example 16: an emergency stop that "still works" but is cracked is still `STOP_NOW`, because rule 1 lists a damaged safety device as a trigger. The route is `MECHANICAL` because the job is replacing the physical button. If the text said the e-stop circuit was faulty, the route would be `INSTRUMENTATION_CONTROLS`.

Note on example 20: non-maintenance requests get `PLAN` because they carry no operational urgency, and `ESCALATE` because a planner must redirect them.

## 9. Labelling procedure (summary)

1. Each example is labelled **twice, independently**, with this rulebook, by two separate passes using different prompts.
2. If both passes agree, the label is kept.
3. If they disagree, a third adjudication pass decides using the tie-break rules and records which rule settled it. If it still can't decide, the example is dropped.
4. The agreement rate between the two passes is reported in the dataset card as a quality metric.
5. Any repeated disagreement pattern leads to a rulebook update, a version bump and relabelling of the affected examples.

## 10. Change log

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-11-19 | First draft for review |
| 1.0 | 2026-11-19 | Approved without changes |
