# Research log

Newest entries at the top. One entry per working session (course requirement, CLAUDE.md §7).

---

## 2026-08-11 — PHASE 3: v_C generalizes to held-out framing routes (correlational)

Ran on the same Colab session as the gate; extraction + probes are CPU against the cached
activations. Notebook: `notebooks/phase2_3_gate_and_vc_colab.ipynb` (sections 8–12).

**Setup.** Consequence cache verified against the dataset (2000 rows, 1000 real — matches
`consequence.jsonl`). `02_extract_directions.py` produced `v_c_L{8..24}` + random nulls.
Minted `r_hat_L{layer}.pt` with Rule-2 sidecars from the gate cube (config layer L =
hidden_states[L] = cache index L-1 = Arditi index L, the mapping the gate confirmed).

**Layer selection — train templates only.** GroupKFold(5) over framing templates *within
train*: AUC 0.996–0.999 at every layer 8–24, selecting L18. **The spread is 0.003, so the
selection is effectively arbitrary — report the curve as flat, not "L18 is best."**

**Held-out reveal (40 unseen framings, routes disjoint from train, n=572, L18):**

| readout | AUC | note |
|---|---|---|
| trained probe | 0.973 (acc 0.918) | fitted on train framings |
| **v_C projection (unfitted)** | **0.932** | the direction itself — the stronger claim |
| random direction | 0.408 | null (single draw) |
| BoW text baseline | 0.753 | from `audit_contrast.py` |

At n=572 the AUC SE is ~0.011, so 0.932 vs 0.753 is ~16 SE — not noise.

**Interpretation, scoped.** A linear direction extracted from training framings transfers to
framing routes it never saw, well above a unigram text baseline and a random direction. This is
**correlational**: it shows the information is present and linearly readable, NOT that the model
uses it. Causality is Phase 5.

**Caveats logged before believing it (red-team pass):**
- **BoW is a weak proxy for "surface".** Beating 0.753 rules out *unigram* leakage only;
  layer-18 activations also encode bigrams, syntax and register. Section 10b adds TF-IDF
  uni/bigram and char 3-5gram baselines — **still to be run**; if those approach ~0.93 the
  claim shrinks a lot.
- **Flat layer curve is mildly deflationary.** Near-perfect separability already at L8 is more
  consistent with a lexically-available feature than a deep representation. Worth stating.
- Single-draw random null (0.408) replaced by a 100-draw distribution in 10b — **to be run**.
- Per-route breakdown of the 20 held-out routes — **to be run**; the headline averages them,
  and heterogeneity would narrow the claim.

**Code fixes made this session:**
- `03_probe.py` selected `best_layer` by **held-out accuracy** — selection on the test set.
  Now selects on train-template group CV (`probe.group_cv_auc`), reports both, and warns when
  the CV spread is <0.01 that the choice is arbitrary.
- `01_cache_acts.py` keyed caches on dataset NAME only, so an edited `.jsonl` would silently
  reuse a stale cache. Now records `source_sha256` and refuses to skip on a mismatch.

**Next:** run 10b and interpret; back artifacts up to Drive (they are on an ephemeral runtime);
then Phase 4 (`v_MP`) **before** Phase 5 (steering) — if `v_C` is persona relabeled, that
reframes the project, and it is better to learn it before spending the largest GPU budget.

---

## 2026-08-11 — PHASE 2 GATE PASSED: our r_hat reproduces Arditi's on Qwen2.5

**Ran on Colab Pro (L4 24GB), not a pod.** Steps: cache activations for the 256-prompt refusal
set with our `01_cache_acts.py`; run Arditi's unchanged `generate_directions` on the SAME prompts
via our Qwen2.5 adapter; compare on CPU. Notebook: `notebooks/phase2_rhat_gate_colab.ipynb`.

**Result — the gate:**

    cos(our r_hat, Arditi r_hat) at offset +1 = 0.9999   (mean over 27 comparable layers)

**Supporting checks (all run before believing it):**
- **Offset discrimination:** +1 = 0.9999 vs offset 0 = 0.7412, −1 = 0.5800, +2 = 0.7383,
  far-apart layers (|Δ|>3) = 0.1427. The comparison *could* have come out low; it didn't.
- **Layer indexing confirmed twice.** Our stored `k` = `hidden_states[k+1]` (block *output*);
  Arditi's `l` = `hidden_states[l]` (block *input*, via forward pre-hook) → offset +1. Two
  independent confirmations: (a) Arditi's layer 0 has **exactly zero** norm, because it is the raw
  embedding and every prompt ends with the same chat-template token, so mean(harmful) −
  mean(harmless) = 0 there (it produced a nan until we excluded it); (b) offset 0 (0.7412) equals
  our own adjacent-layer self-similarity (0.7412) to 4 dp, which is forced iff the mapping holds.
- **Permutation null (200 shuffles):** true 0.9999 vs null mean 0.0134, sd 0.3959, max 0.8058,
  **p = 0.0000** — the true labels beat every one of 200 permutations.
- **Outlier dominance:** top-10 of 3584 dims hold only 6.6% of the direction — distributed
  feature, not a few rogue dimensions.
- Random-*direction* noise floor |cos| p95 = 0.0325.

**Methodological corrections made (worth remembering):**
- A **single** permutation with a hard threshold was a bad test: the permutation null is wide
  (sd ≈ 0.40) though centered at ~0, so one draw of 0.38 looked alarming and was noise. Fixed the
  notebook to report the 200-shuffle distribution + p-value. **A null needs its distribution, not
  a threshold.**
- The **z-score (2.5) is the wrong statistic** here — the null is sign-symmetric and non-Gaussian.
  Use the p-value / null max.
- Random-*direction* floor (0.0325) ≠ random-*label-split* null (sd 0.40). Do not conflate them;
  the label-split null is the right control for a diff-in-means claim.

**What this licenses and what it does NOT.** Our activation caching, token position, layer
indexing, and chat-template handling are correct → we can trust `v_C` extracted by the same code.
It says **nothing** about jailbreaks, fiction-framing, or whether `v_C` exists. This is a methods
footnote, not a finding — the gate is near-tautological by design (both sides compute the same
math on the same prompts), so it catches gross plumbing errors only.

**Next (Phase 3, mostly CPU):** cache the consequence set (2000 rows) while the model is warm →
`02_extract_directions.py` for `v_C` per layer + random nulls → `03_probe.py` with the group-wise
split, reporting held-out-framing accuracy against the BoW baseline (held-out AUC 0.753).

---

## 2026-08-07 — Consequence set rebuilt by pair-sampling (140 framings)

**Context:** framings expanded to **140** (real/train 50, hypo/train 50, real/heldout 20,
hypo/heldout 20; 51 routes) to maximise framing diversity. Full cross product (140x40) = 5600
rows was more than needed and unbalanced per framing. This supersedes the old "frozen 1000
rows / 25 framings" set noted below — the earlier freeze is retired; the NEW held-out framings
(20 real + 20 hypo, routes disjoint from train) are the set to freeze going forward.

**Change:** rewrote `scripts/00_build_consequence.py` with a `--mode sample` (default; `--pairs`,
`--seed`) alongside the old `--mode full`. Sample mode draws **contrast pairs** — one task under
one real + one hypo framing (2 rows). This preserves the §2 invariant *by construction*: both
rows of a pair share the task, so the real and hypo classes have identical task multisets and
task content cancels in mean(real)-mean(hypo). Framings are cycled (uniform usage); tasks are
chosen least-used-first (uniform usage); pair budget is split train/held-out in proportion to
framing counts. Rows gained a `pair_id` linking the two halves of a pair. Schema otherwise
unchanged, so `load_consequence` and the probe grouping are unaffected.

**Built + verified (`--pairs 1000 --seed 0` -> 2000 rows):**
- 140/140 framings used, 14–15 rows each (max diff 1); all 40 tasks used, 24–26 per class.
- **real/hypo task multiset identical: True** (invariant holds); 1000 pairs, 0 malformed.
- train 1428 rows / held-out 572; routes disjoint train vs held-out for both polarities.
- `scripts/audit_contrast.py` **PASSES** all four: task-match 0 missing; BoW held-out AUC 0.753
  (train 1.000, gate <0.9); length diff 3.3% (<10%); template balance min 14 rows/template (>8).

**Discipline:** FREEZE the 40 held-out framings now — no edits, no peeking at held-out probe
accuracy while iterating. Expanding train framings + re-auditing is still allowed.

---

## 2026-08-07 — Phase 2 prep: r_hat gate machinery (local, CPU)

**Goal:** get everything ready to reproduce Arditi's refusal direction `r_hat` on Qwen2.5-7B as
a machinery-validation gate, so a short GPU session is all that remains. Prep done locally.

**Recon (both reference repos cloned into `external/`, read not run):**
- Arditi's harmful/harmless splits ship in-repo (`dataset/splits/*.json`, AdvBench vs Alpaca) —
  no dataset hunting. The gate needs only `generate_directions` + (optionally) `select_direction`
  (pipeline steps 1–4); the vLLM + Together-AI parts are steps 5–7 and are **not needed**.
- **Arditi's `QwenModel` targets the original Qwen** (`model.transformer.h`, `tokenizer.eod_id`)
  and crashes on Qwen2.5 (a Qwen2-arch model: `model.model.layers`, standard tokenizer). Handled
  by an adapter, without editing their code.
- Zhong's repo confirmed on **Qwen2.5-7B-Instruct (89 refs) + Meta-Llama-3.1-8B-Instruct (38)** —
  i.e. our exact two models; this is *why* Qwen2.5 is the shared substrate (v_MP, r_hat, v_C must
  share one activation space for cosines to be defined). Zhong also already ships a 6-direction
  geometry comparison, three-way metric, knockout, steering — our Phase 3–4 template. v_MP is the
  heavier reference and is deferred to Phase 4; it is NOT on the gate's critical path.

**Done (Part A — local prep, all CPU):**
- `scripts/phase2_build_refusal.py` → `data/contrast/refusal.jsonl` (256 rows, 128 harmful /
  128 harmless, verbatim from Arditi's train splits, seed 0). Both sides of the gate run on these
  same prompts.
- `data.py`: added `load_labeled` + a `load_contrast` schema-dispatcher (framing→consequence,
  label→labeled); `01_cache_acts.py` now calls `load_contrast`, so caching is schema-agnostic.
- `scripts/arditi_qwen25/{qwen25_model.py, run_extract.py, README.md}` — the adapter (`ModelBase`
  subclass for Qwen2.5; extraction-only, steering methods stubbed) + a thin runner that feeds our
  refusal prompts to Arditi's unchanged `generate_directions` and writes
  `artifacts/directions/r_hat_mean_diffs.pt` (+ Rule-2 sidecar). Runs in Arditi's venv on the pod;
  never imports our `src/`.
- Decision: **skip `select_direction`** for the gate — it needs generation + Qwen-specific refusal
  token ids (fragile on Qwen2.5) and only *picks* a layer. We pick the comparison layer on CPU,
  matched to v_C, and compare our diff-in-means at position −1 against the cube's −1 slice.
- Adapter `_tokenize` mirrors `acts.py` formatting exactly (canonical chat template,
  add_generation_prompt, left pad) so the cosine tests extraction machinery, not formatting.
- Verified: `refusal.jsonl` builds; dispatcher loads both schemas; all new/edited files
  `py_compile` clean. Adapter/runner not executed (need GPU + Arditi venv).

**Flag (needs reconciling before Phase 3, NOT touched here):** `data/contrast/consequence.jsonl`
is now **5600 rows / 140 templates / 1600 held-out** and is **untracked** — but this log's
2026-08-06 entry says the set was FROZEN at 1000 rows / 25 templates. Either a large train
expansion happened off-log or the file was regenerated. The held-out freeze is a scientific
invariant (§2); confirm what happened and whether the held-out templates are still the frozen set.

**Next (Phase 2, on the pod — short, ~24 GB, HF token only, no vLLM/Together):**
1. Clone Arditi into `external/` on the pod + own venv (`requirements.txt`).
2. `01_cache_acts.py --dataset refusal` and `--dataset consequence` (cache both while Qwen2.5 is
   loaded; trust the consequence cache only after the gate passes).
3. `scripts/arditi_qwen25/run_extract.py` → `r_hat_mean_diffs.pt`.
4. Back on CPU: mint unit `r_hat` at the v_C layer (pos −1) via `io.save_direction`; compute
   `cosine(our r_hat, Arditi r_hat)` per layer → the gate.

---

## 2026-08-06 — Repo scaffolding & pipeline skeleton

**Goal:** get the repository to the state where experiments can start — structure aligned to the
artifact contract, a runnable CPU pipeline, and a benign seed dataset.

**Done:**
- Restructured to the documented contract: `src/consequence/` package (`acts`, `directions`,
  `hooks`, `generate`, `probe`, `judge` + `config`, `io`, `data`), `configs/qwen.yaml`,
  `scripts/01…07`, `artifacts/{directions,activations,generations,scores,figures}/`,
  `external/`, `logs/`. Replaced `results/` and the flat `src/`.
- **Removed `transformer_lens` and `nnsight`** from deps (CLAUDE.md §4 forbids them). Moved to
  `pyproject.toml` (uv); plain HF + forward hooks only.
- Enforced Rule 2 in code: `io.save_direction()` refuses to write a `.pt` without a complete
  sidecar; `io.load_direction()` errors if the sidecar is missing.
- **Normalized the consequence dataset** into two editable sources —
  `benign_tasks.jsonl` (40 tasks, 20 domains, screened for framing-token leakage) and
  `framings.jsonl` (15 framings, `split` on the framing) — with `scripts/00_build_consequence.py`
  generating `consequence.jsonl` as a validated cross product (600 rows). Editing one task or one
  framing is a one-line change + rebuild; `consequence.jsonl` is generated, never hand-edited.
  Tasks are capitalized standalone sentences; framings end in a `{task}` slot so joins read cleanly.
  Still benign-only and light on framing diversity — expand framings via `build-contrast-set`.
- **Made the held-out set route-disjoint.** Each framing now has a `route` (the semantic
  mechanism it uses). Train real = {audience, action, deployment}; held-out real = {record,
  irreversibility, liability}. Train hypo = {fiction, simulation, non_execution,
  thought_experiment}; held-out hypo = {privacy, draft, idle_curiosity}. Held-out reaches the
  same real/hypo distinction through routes unseen in training, so passing it is evidence of a
  *concept*, not shared surface vocabulary. The build warns on any train/held-out route overlap;
  `route` is carried into `consequence.jsonl` for per-route held-out breakdowns later.
  **Discipline: freeze these held-out framings before probing; never iterate on them.**
- **Ran build-contrast-set.** Expanded train to 22 templates / 13 route-mechanisms (6 real +
  7 hypo), routes still disjoint from held-out. Added `scripts/audit_contrast.py` (the skill's 4
  mandatory checks). Audit PASSES: task-match 0 mismatches; **BoW vocab-leakage held-out
  AUC=0.712** (train 1.000), gate <0.9; length diff 0.5%; 40 rows/template. The held-out BoW
  signal is driven by **negation** ("never/no") that generalizes across hypo routes — this is the
  honest baseline the activation probe must beat, and it will be reported next to the probe number.
  880 rows total, all benign (deliberate, per decision #1).
- **Negation-balancing pass + FROZEN.** Added 3 negation-bearing *real* train framings
  (routes: deployment, emergency, emphatic_real) so "no/not/never" are no longer hypo-only tells.
  BoW held-out AUC **0.712 → 0.675**; the strong content-word leaks (`no`, `nothing`) dropped out
  of the top cues, leaving mostly function words + mild `never`/`purely`. Length confound 0.5% →
  5.0% (still <10%). Set `class_weight="balanced"` in `probe.py` and the audit LR (train is now
  560 real / 440 hypo; diff-in-means is mean-based so unaffected, held-out stays 3/3). Final set:
  **25 framings / 13 route-mechanisms × 40 tasks = 1000 rows.**
  **HELD-OUT FRAMINGS ARE NOW FROZEN** — no further edits, no inspecting held-out probe accuracy
  while iterating. Expanding train (reserved routes untouched) + re-auditing is still allowed.
- Smoke-tested the CPU path end-to-end (02→03→04→07) against a fabricated activation cache with
  a planted real/hypo signal: extraction + sidecars, probe held-out split, geometry + random
  null band, and figures all run. Fabricated artifacts deleted afterward (never committed).
- Local `.venv` for CPU stages (numpy pinned `<2` for the older-Mac torch 2.2.2 wheel).

**Decisions / notes:**
- `acts.py` imports `transformers` lazily so CPU stages that only touch `.npz` don't need the
  GPU stack (Rule 3).
- `docs/{project_plan,compute_estimation}.md` still mention TransformerLens/nnsight — stale vs
  CLAUDE.md §4. Left the grant/plan prose as-is; flag for a cleanup pass before the write-up.

**Next (Week 1):**
1. Build the real `consequence.jsonl` with the `build-contrast-set` skill (≥10 templates,
   held-out templates, benchmark `[TASK]` text).
2. On the pod: install `transformers`+`accelerate`, run stage 01 on Qwen2.5-7B-Instruct.
3. Clone `refusal_direction` + `refusal-downstream-persona` into `external/` (own venvs),
   export `r_hat.pt` / `v_mp.pt` with sidecars; sanity-check our diff-in-means `r̂` against theirs.

---

## 2026-08-07 — Framing set expanded 25→140 and held-out re-split

Expanded `data/contrast/framings.jsonl` to the target sizes requested: **50 real + 50 hypo
train, 20 real + 20 hypo held-out (140 framings × 40 benign tasks = 5600 rows)**. Goal was
maximal diversity with no surface feature confounded with polarity.

- **Five diversity axes varied independently per line:** route (semantic mechanism), POV
  (1st/2nd/3rd/impersonal), structure (imperative/declarative/conditional/label), register
  (formal/casual/technical/terse), and length. 26 distinct route-mechanisms total.
- **Held-out is disjoint from train in route AND register/vocabulary**, not just route:
  train-real leans operational/deployment; held-out-real leans legal/liability/permanence.
  train-hypo leans fiction/simulation; held-out-hypo leans sandbox/parody/allegory/metaphor.
  This keeps the generalization test genuinely OOD rather than a paraphrase test.
- Authored via a one-off generator (`scratchpad/gen_framings.py`) purely to emit valid JSONL
  and check length balance; the wording is hand-authored and `framings.jsonl` remains the
  committed source. Rebuilt with `00_build_consequence.py`.
- **Audit passes all four:** task-match 0 mismatches; **BoW held-out AUC 0.755** (gate <0.9,
  the honest baseline the probe must beat); length confound 3.3%; 40 rows/template.

**Held-out RE-SPLIT (freeze reset):** the previous 6 held-out framings were superseded. This is
clean — no probe/held-out number had been computed yet, so the exam is rewritten before it was
ever taken, not after peeking. From this commit the new held-out framings are **FROZEN again**.

**Note:** still all benign tasks (decision #1 unchanged). Real harmful `[TASK]` text comes from
benchmark suites at runtime, never committed.
