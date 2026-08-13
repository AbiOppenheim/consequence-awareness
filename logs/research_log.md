# Research log

Newest entries at the top. One entry per working session (course requirement, CLAUDE.md §7).

---

## 2026-08-13 — Refactor: one script per research step, results cached on disk

No new science this session. The pipeline was restructured because the notebook had become the
implementation: layer selection, the held-out reveal, the geometry table and the v_MP extraction
existed **only as notebook cells**, duplicating (and diverging from) `03_probe.py` and
`04_geometry.py`, which were stale and had never been run. Every number lived in kernel state,
so a Colab disconnect meant recomputing minutes of CV, and three cells each reloaded the 375 MB
activation cache and refit the same probes.

**New contract.** `src/consequence/results.py`: each analysis step writes
`artifacts/results/<step>.json` = the numbers plus a fingerprint over (input file hashes,
parameters, the analysis code that ran). A step re-runs only when one of those moved, and prints
which. Committed, because these are the numbers the write-up cites.

- Scope is the CPU layer only. Stage 01 (`.npz` keyed on the dataset hash) and stage 05
  (resumable per condition/alpha) keep their own caches — an edit to an unrelated source file
  must never invalidate a GPU artifact.
- Steps: `02b_gate` (gate + its 4 red-team checks + mints `r_hat_L*.pt`), `03_probe`
  (layer selection, **train only** — it no longer computes anything about held-out),
  `09_heldout` (the reveal), `10_redteam_heldout`, `04_geometry` (rewritten: per-layer, both
  v_MP framings), `11_calibrate_alpha`.
- The selected layer and the alpha ladder are read from disk by everything downstream;
  `05_generate.py` defaults to `--layer auto --alphas auto`. Nothing is retyped into a command.
- `02_extract_directions.py --kind v_mp` mints the persona directions, so all three directions
  come from the same validated diff-in-means. Previously v_MP was minted by a notebook cell —
  three directions extracted three ways cannot be compared by cosine and blamed on geometry.
- The notebook is now thin: run a step, render its stored result. Outputs cleared (the saved
  ones were a mix of two sessions and included errors from bugs since fixed).

**Discipline the store buys us, not just speed:**
- `09_heldout` refuses to run if `v_c_L*.json` does not say `split: train`. That check is the
  generalization claim; without it the reveal is circular.
- Re-running the reveal after anything changes archives the previous result as
  `heldout.prev-<sha>.json` and prints a warning to report both and label the second post-hoc.
- Re-selecting the layer invalidates the reveal, the red-team and the ladder automatically.

**Verified** against a fabricated fixture (planted signal, tiny d_model, an "Arditi cube" built
to agree at offset +1): all six steps run; re-runs are no-ops; a param change recomputes only
that step; a layer re-selection cascades; editing `probe.py` invalidates the steps that import
it and not the gate; the split guard fires. Fixture deleted, nothing written into `artifacts/`.

**Second steering bug, caught by the smoke test (section 16) before the sweep.** `steer_hook`
and `knockout_hook` assumed a decoder block returns `(hidden_states, ...)`. Transformers ~4.54+
returns a **bare tensor**, so `output[0]` did not raise — it silently took the first **batch
row** — and the tuple the hook returned surfaced as `AttributeError: 'tuple' object has no
attribute 'dtype'` three layers downstream inside Qwen2's `input_layernorm`, nowhere near the
hook. Fixed by dispatching on the container type (`_edit_resid`); it must be an isinstance
check, not a try/except, because the tensor path fails silently rather than raising.

This is the second bug in the same 15 lines (the first steered one layer off, logged 08-11), and
both were invisible to the ad-hoc stub used to verify the first — **that stub returned tuples,
so it tested the assumption instead of challenging it.** `scripts/test_hooks.py` now runs the
hooks against BOTH block conventions on a stub stack, asserting on which hidden_states layer
actually moved, that the shift is identical across batch rows, that shape and container type
survive, and that knockout zeroes the component. Verified it fails against the old
implementation. Committed this time, not ad-hoc: run it before any sweep. The smoke test earned
its place — this would have cost an hour of GPU and produced generations that looked fine.

**Third generation bug, and the worst one: RIGHT padding in `generate()`.** A decoder-only model
continues from the last token of the sequence, so every prompt shorter than the longest in its
batch was continuing from `<pad>` instead of from its own final token. Nothing raises;
transformers warns once per call, buried in loading bars. Reproduced on a stub: with the old
code a batch of three prompts returns **`['', 'ooo', '']`** — only the longest prompt generates
at all.

**This bug would have manufactured a positive result.** Empty completions from a steered
condition read as refusal to any judge, so "steering toward real restores refusal" would have
come out strongly, at every alpha, for `v_C` *and* the random null — and the null would have
looked equally good, which is the one thing that might have given it away. Fixed by setting
`padding_side="left"` inside `generate()` and restoring it after.

Deliberately NOT set globally on the tokenizer: `acts.cache_activations` needs the opposite.
A plain forward pass takes `position_ids = arange(seq_len)`, so left padding would shift every
real token's RoPE position by the number of pads. Right padding puts real tokens at 0..n-1 and
the existing mask fallback finds the true last token — **the cached activations and the 0.9999
gate are unaffected and stay valid.** `generate()` is the one path that rebuilds position_ids
from the attention mask, which is exactly why it wants the other convention.
`scripts/test_generate.py` pins this with a stub that echoes the last input token; verified it
fails against the old code.

Also fixed: `["steer_rhat"] if r_hat else []` in the sweep's meta sidecar — truthiness of a
multi-element tensor raises. It crashed *after* all 15 smoke generations, so it cost nothing,
but it would have wasted the full sweep's model-load time.

**Sweep cost, measured.** 17 conditions (1 baseline + 4 conditions x 4 alphas) over 100 attacks
= 1,700 generations, but XSTest ships **250** safe prompts = 4,250 more, so ~6,000 total and
**71% of the GPU time is the over-refusal control**. At 256 new tokens / batch 16 that is
roughly 60-95 min on an L4 — worth knowing against the ~12h experiment budget. XSTest does not
need the full ladder: the guard is "at the alpha where steering restores refusal, does it also
refuse safe prompts?", and one rung answers it. The notebook now runs XSTest at the headline
alpha only, saving ~3,000 generations. The attack sweep keeps the full ladder.

`step()` also now streams stdout live (stderr to a temp file, shown only on failure). It
buffered everything, so an hour-long cell that prints one line per condition was
indistinguishable from a hung one.

**⚠️ Carried over and still open — the Phase 3/4 numbers need one CPU re-run.** Commit `b7ede1c`
(2026-08-12) fixed stage 02 to extract `v_C` from train templates only; before that it did
difference-in-means over **all 2000 rows, held-out included**, and section 10 then scored that
vector on those same 572 rows. So the logged **0.932 unfitted held-out AUC, the per-route table,
and every cosine in the Phase 4 geometry table came from a leaked `v_C`**. The trained-probe
0.973 is unaffected (fit on train rows only). Expect the numbers to move a little, not a lot —
but "should be fine" is not a number. Restore artifacts from Drive and run steps 02 → 03 → 09 →
10 → 04 on CPU before any of this goes in the write-up. Do it before the GPU session for the
sweep, not after.

---

## 2026-08-11 — PHASE 3 VERIFIED + PHASE 4: v_C is distinct from BOTH r_hat and persona

All CPU against cached activations (persona sets needed one ~3 min GPU pass to cache).

### Phase 3 red-team (notebook 10b) — the result survives

- **Per-route generalization is UNIFORM.** 19 of 20 held-out routes AUC >= 0.877, median ~0.94.
  Only `sandbox` weak (0.736) — semantically reasonable, a sandbox arguably *is* a real system.
  The headline is not an average masking a few strong routes.
- **Held-out RISES with depth** (unfitted v_C): 0.875 (L8) -> 0.917 (L14) -> 0.932 (L18) ->
  **0.955 (L22)** -> 0.947 (L24); probe 0.933 -> 0.992. **This retires the earlier "flat layer
  curve => shallow lexical feature" concern, which was based on TRAIN-CV (flat at 0.996-0.999);
  held-out is not flat.** A surface-lexical feature would not improve with depth.
- **Stronger text baselines (held-out):** TF-IDF unigram 0.787, **uni+bigram 0.847**, char
  3-5gram 0.771. Like-for-like (both fitted): probe **0.973** vs best text **0.847**.
  So v_C wins clearly, but **surface form reaches ~85% of the way — state the number, do not
  claim a categorical "beyond surface vocabulary."**
- **Random-direction null, 100 draws:** mean 0.479, 5-95% [0.263, 0.698]. Random directions can
  reach 0.70, so a wide null; 0.932 is comfortably outside it.
- Train-CV selected **L18** but held-out peaks at L22. Unbiased selection landing slightly short
  of optimal is what honest selection looks like. **Headline stays L18 = 0.932**; the layer curve
  is supporting evidence, not licence to upgrade the number.

### Phase 4 geometry (v_MP = our reimplementation of Zhong's compliant_v2 model_persona)

At L18, against a random-direction null band of p95 = 0.032:

| pair | cosine |
|---|---|
| v_C . r_hat | **+0.083** (rises to +0.209 by L24) |
| v_C . v_MP (system framing) | **+0.287** |
| v_C . v_MP (user-turn framing) | **+0.338** |
| v_MP(sys) . v_MP(user-turn) | **0.908** |
| r_hat . v_MP | -0.225 at L18, **-0.414 by L24** |

- **Both deflationary explanations fail.** v_C is neither r_hat relabeled (cos 0.083) nor the
  persona direction relabeled (cos ~0.31, i.e. ~10% shared variance). Claim 1 is supported for
  this contrast set and model.
- **The 0.908 framing agreement matters**: building v_MP with the instruction in the system
  prompt vs the user turn gives essentially the same axis, so cos(v_C, v_MP) is not an artifact
  of prompt structure — the confound the two-framing design was built to rule out.
- **Unplanned validation of our v_MP.** r_hat . v_MP grows increasingly NEGATIVE with depth
  (-0.414 at L24): compliant persona and refusal are anti-aligned late in the stack, which is
  exactly Zhong's "refusal lives downstream of persona". Since we had no reference vector for
  v_MP, this is independent evidence the reimplementation captures something real, and partly
  closes the limitation logged earlier.

### Caveats to carry into the write-up
- cos(v_C, v_MP) ~ 0.29-0.34 is **not** zero. Say "distinct with modest overlap", not orthogonal.
- Still **correlational**. Nothing here shows the model USES v_C. Phase 5 is the causal test.
- Scope to this real-vs-hypothetical contrast set, Qwen2.5-7B, and these framings (Kirch et al.).
- On Colab the persona *builder* failed (Zhong's repo not cloned there, no traits.zip); it worked
  because persona.jsonl/persona_ut.jsonl are committed. Numbers are valid; regeneration needs
  their repo cloned.

### Code fix this session (pre-sweep, caught before any GPU spend)
`hooks.apply_hooks` steered at the **wrong layer**: our convention is config layer L =
hidden_states[L] = cache index L-1, but a post-forward hook on `blocks[L]` rewrites block L's
OUTPUT = hidden_states[L+1]. Verified with a stub decoder stack (steering "at L4" first changed
hs[5]); now hooks `blocks[L-1]` with a range check, re-verified at four layers. Had this shipped,
every steering result would have been one layer off target.

**Next:** Phase 5 — the causal sweep. `scripts/phase5_build_eval.py` downloads XSTest (safe,
committed) and published fiction/role-play attacks (gitignored, used verbatim).

## 2026-08-11 — Phase 4 prep: Zhong's repo is a partial release; v_MP rebuilt from their spec

**Finding (scoping read, no GPU).** `violazhong/refusal-downstream-persona` **cannot produce
`v_MP` as shipped**: `src/extract_vectors.py` (and 5 other files) import an `ithou` package
that is **not in the public release**, there is no `data/` directory, and path constants point
at `/root/repositories/i-and-thou-vector-private` — the working implementation is private.
The original Phase-4 plan ("clone, run, export v_mp.pt") is therefore impossible. Cost of
discovering this: a 10-minute read, not a booked GPU session.

**What they DID release, and it is enough.** `configs/traits.zip` (243 trait YAMLs) and, in
`src/compliant_residual.py`, the exact Qwen2.5 configuration for their model-persona vector:

    trait = compliant_v2 | vector = model_persona | position = prompt_end | layer = 20

`prompt_end` is our own token convention, and our diff-in-means is validated (gate, cos 0.9999).
So we reimplement their method with their trait definitions. **Label it as a reimplementation,
never as "Zhong's v_MP" — and note the limitation: unlike r_hat there is no reference vector to
check it against.** An email to the authors for the `ithou` package is worth sending in parallel.

**Built (`scripts/phase4_build_persona.py`):** trait `compliant_v2`, 8 positive x 8 negative
instructions paired by index, over 55 questions (their 15 trait questions + our 40 benign tasks
as a neutral control, tagged by `source`). 880 rows per framing, 440/440, question-matched
(verified) so question content cancels in the difference of means.

**Two framings on purpose** — the choice is a confound for the very comparison we want:
- `persona.jsonl` — instruction in the SYSTEM prompt (faithful to their model_persona)
- `persona_ut.jsonl` — instruction in the USER turn (structurally identical to how v_C and
  r_hat were extracted, so a low cos(v_C, v_MP) cannot be blamed on prompt structure)
Agreement between the two is itself reported.

**Sign convention:** v_MP = mean(compliant) - mean(restrictive) points TOWARD compliance,
roughly opposite in spirit to r_hat (harmful - harmless). Compare magnitudes.

**Supporting code changes:**
- `acts.py::_render` now accepts `{"system", "user"}` prompts. The plain-string path is
  **byte-identical** (asserted in test), so the r_hat gate that validated this function
  still applies.
- `data.py::load_contrast` dispatched on `framing` before `label`, which misrouted the persona
  set (its `framing` key meant system-vs-userturn, not real-vs-hypo). Now dispatches on the
  unambiguous `label` first, and the persona builder uses `prompt_framing` to avoid the clash.
  Caught by a loader test before any GPU time.

**Pending from Phase 3:** section 10b (per-route breakdown, held-out across all layers, TF-IDF
baselines, 100-draw null) is written but **not yet run**. The TF-IDF result is what decides how
strongly the Phase 3 claim can be worded.

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
