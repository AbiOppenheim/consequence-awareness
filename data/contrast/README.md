# data/contrast/ — the contrastive sets each direction is extracted from

Every direction is difference-in-means over one of these files. All three share the same idea:
two groups of prompts that differ in exactly one thing, so the mean difference isolates that
one axis.

| file | groups (pos − neg) | who authors it | status |
|---|---|---|---|
| `consequence.jsonl` | real − hypothetical framing | **us, from scratch** | benign seed present; expand with `/build-contrast-set` |
| `refusal.jsonl` | harmful − harmless request | Arditi's balanced splits | to populate (Week 1) |
| `persona.jsonl` | compliant − evasive persona | Zhong's trait configs | to populate (Week 1) |

## Normalized: two sources you edit, one file that's generated

The consequence set is **factored** so you can change one task or one framing in one place:

```
benign_tasks.jsonl   one row per task     (the [TASK] content, held constant across framings)
framings.jsonl       one row per framing  (the real/hypo wrapper, with a {task} slot + split)
      │  scripts/00_build_consequence.py  (cross product + validation)
      ▼
consequence.jsonl    GENERATED — do NOT hand-edit; rerun the build after editing a source
```

Rebuild after any edit:

```bash
python scripts/00_build_consequence.py
```

**benign_tasks.jsonl** — `{"task_id", "domain", "text"}`. Text is a capitalized standalone
sentence. Tasks must not contain framing vocabulary (real/fiction/imagine/…); the build fails
if they do, because a giveaway token would let the probe cheat.

**framings.jsonl** — `{"framing_id", "polarity": real|hypo, "split": train|heldout, "text"}`.
Text is a self-contained sentence ending in a `{task}` slot, so `framing + task` always reads
correctly. **`split` lives on the framing** — held-out is by TEMPLATE, never by example
(CLAUDE.md §2). This is what measures a *concept* rather than memorized surface vocabulary.

**consequence.jsonl** (generated) — one row per (framing × task):
`{"id", "framing", "split", "template_id"(=framing_id), "task_id", "task_source", "text"}`.
`framing`: real → label 1, hypo → label 0. Because each task appears under every framing, task
content cancels in `mean(real) − mean(hypo)` — the direction is framing-only by construction.

## Framing routes — why the held-out set is built this way

A framing signals "real" or "hypothetical" through some **semantic route**, and each framing
carries a `route` field naming it. The held-out set is deliberately **route-disjoint** from
train: it reaches the same real/hypothetical distinction through mechanisms the probe never saw.

| Polarity | Train routes (16 real / 15 hypo mechanisms) | Held-out routes (10 each, reserved, never in train) |
|---|---|---|
| real | audience, action, deployment, safety, financial, emergency, public, operational, embodied, time_pressure, medical, broadcast, verification, dependency, witnessed, immediacy | liability, record, irreversible, contractual, legal, regulatory, escalation, commitment, cost_of_error, permanence |
| hypo | fiction, role_play, simulation, dream, thought_exp, non_execution, reenactment, training_data, game, what_if, abstract, fantasy, rehearsal, brainstorm, daydream | sandbox, counterfactual, parody, draft, privacy, idle_curiosity, stage_play, dry_run, metaphor, allegory |

Why it matters: if train and held-out shared routes (e.g., both used fiction vocabulary), a
probe that keyed on "fiction words" would pass held-out and we'd wrongly conclude it learned the
*concept*. It's the difference between a within-distribution test and an out-of-distribution
one — only the latter licenses a claim about the concept rather than surface artifacts. A pass
on unseen routes is strong evidence for a consequence representation; a fail is an honest,
publishable finding that `v_C` is lexically scoped. `00_build_consequence.py` warns if any
held-out route reappears in train.

Discipline: **freeze the held-out framings before looking at any probe number, and never iterate
on them.** The generalization number only means something if the exam was written blind.

## Current status: expanded + re-split, audited, held-out FROZEN

140 framings (**100 train: 50 real / 50 hypo — 20 held-out per polarity, 40 total**;
26 distinct route-mechanisms) × 40 benign tasks = 5600 rows.
All **benign** (deliberate — keeps `v_C` uncontaminated by harmfulness; see decision #1 in the log).

Diversity is varied deliberately along five independent axes so no single surface feature is
confounded with polarity: **route** (semantic mechanism), **POV** (1st/2nd/3rd/impersonal),
**structure** (imperative/declarative/conditional/label), **register** (formal/casual/technical/terse),
and **length**. Held-out is disjoint from train not only in route but in register/vocabulary
(train real ≈ operational/deployment; held-out real ≈ legal/liability/permanence) so the
generalization test is genuinely out-of-distribution.

Audit (`scripts/audit_contrast.py`) PASSES all four checks:

| Check | Result |
|---|---|
| task-match | 0 mismatches |
| **vocab leakage (BoW), held-out AUC** | **0.755** (gate <0.9) — the honest baseline the probe must beat |
| length confound | 3.3% (gate <10%) |
| template balance | 40 rows/template (gate ≥8) |

**The held-out framings were RE-SPLIT** in this expansion (the previous 6 were superseded). This is
clean because no probe/held-out number had been computed yet — the exam is being rewritten before it
was ever taken, not after peeking. From **this commit** the held-out framings are **FROZEN again**:
do not edit them, and do not inspect held-out probe accuracy while iterating. Expanding **train**
framings later (respecting the reserved held-out routes) is fine and re-runs the audit; touching
held-out burns it and forces another re-split.

## Harmful task text is never stored here

For the real experiments, `[TASK]` slots are filled from **published benchmark suites only**
(de-duplicated, cyber-category capped — CLAUDE.md §6). Harmful task strings are loaded from
those libraries at runtime and are not committed to this repo (CLAUDE.md §7).
