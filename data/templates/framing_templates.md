# Consequence-Contrast Framing Templates

> **Source of truth is [`data/contrast/framings.jsonl`](../contrast/framings.jsonl)** (140 framings:
> 50 real + 50 hypo train, 20 real + 20 hypo held-out). This file is the design rationale; edit the
> JSONL and rerun `scripts/00_build_consequence.py`, don't hand-maintain lists here.

Minimal pairs share the same `[TASK]`; only the real-vs-hypothetical framing varies.
**Design rules** (from `docs/tutorial.md` §4.3):

1. Diversify surface forms — if every hypothetical prompt says "exercise", the probe learns the token "exercise", not consequence. We push this across five axes: **route** (semantic mechanism), **POV**, **structure**, **register**, and **length**.
2. Hold out entire templates (not just examples) for generalization testing, and make held-out **route- and register-disjoint** from train so the test is truly out-of-distribution.
3. Keep task content identical across each pair.
4. Split templates into TRAIN and HELD-OUT before generating. Current size: 100 train / 40 held-out framings × 40 tasks.

## Route inventory (mechanism of real/hypothetical signalling)

**REAL — train:** audience, action, deployment, safety, financial, emergency, public, operational, embodied, time_pressure, medical, broadcast, verification, dependency, witnessed, immediacy.
**REAL — held-out (reserved):** liability, record, irreversible, contractual, legal, regulatory, escalation, commitment, cost_of_error, permanence.

**HYPO — train:** fiction, role_play, simulation, dream, thought_exp, non_execution, reenactment, training_data, game, what_if, abstract, fantasy, rehearsal, brainstorm, daydream.
**HYPO — held-out (reserved):** sandbox, counterfactual, parody, draft, privacy, idle_curiosity, stage_play, dry_run, metaphor, allegory.

Representative wordings live in the JSONL; a couple per route, varied in POV/structure/register.

## Tasks
`[TASK]` slots are filled from **published benchmark prompts only** (balanced harmful/harmless
sets per docs/tutorial.md §4.1 — de-duplicated, cyber-category capped). Harmful task text is
loaded at runtime from the benchmark libraries and is **not stored in this repo**.
