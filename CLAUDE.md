# CLAUDE.md — consequence-direction

Persistent memory for this repo. Read fully before your first action in a session.

---

## 1. What this project is

We test whether some jailbreaks work by shifting a model's internal sense of whether the
situation is **real** — rather than by suppressing refusal directly.

Concretely: extract a **consequence direction** `v_C` from contrastive prompts that differ
only in real-vs-hypothetical framing, then ask whether steering along it *causally* changes
refusal under fiction-framed jailbreaks.

This is a 5-week course project with a **~30 hour total human budget**, of which roughly
12 hours are experiment time. It ships as a public blog post, not a paper.

### The two claims

1. **Existence + distinctness.** A linear direction `v_C` exists, generalizes to *held-out
   framing templates*, and is geometrically distinct from the refusal direction `r̂`
   (Arditi et al. 2024) and the persona direction `v_MP` (Zhong & Li 2026).
2. **Causality.** Steering toward "real" along `v_C` restores refusal during fiction-framed
   jailbreaks, beyond what random-direction and persona baselines explain.

### Every outcome is a publishable result

- `v_C` exists, is distinct, steering works → new causal defense target.
- `v_C` is just `v_MP` relabeled → useful negative result.
- `v_C` exists but steering does nothing → correlate, not cause; defense work stays on refusal.
- `v_C` doesn't generalize to held-out templates → the probe learned surface tokens. Say so.

**Never optimize toward a positive result.** A clean negative result scores full marks on
this course's rubric. If you find yourself proposing a change that would make the result
"work," stop and flag it instead.

---

## 2. Scientific invariants (non-negotiable)

These exist because they are the specific ways this project can produce a wrong answer.

- **Held-out *templates*, never held-out *examples*.** At least 3 framing templates are
  reserved and touched by nothing until the final evaluation. Splitting examples within
  the same templates measures memorization of surface vocabulary, not the concept.
- **Every intervention needs a random-direction null.** A steering or ablation effect with
  no random-direction control is not evidence. Run the null in the same sweep, same seeds.
- **Never report StrongREJECT alone.** Score every generation with the three-way
  refusal / bypass / degenerate label. Zhong shows a StrongREJECT of 0.07 can conceal
  42% real bypass plus 56% incoherent output. High steering strength breaks fluency, and a
  naive refusal rate reads gibberish as a successful defense.
- **Matched task content across every contrast pair.** The only thing that varies is framing.
- **Scope all claims to the real-vs-hypothetical contrast and the specific attacks tested.**
  Kirch et al. found jailbreak-relevant features are attack-family-specific and partly
  non-linear. Do not write "the consequence direction" as if it were universal.
- **Fixed seeds everywhere.** Record them in the artifact sidecar.
- **One variable per experiment.** If a run changes layer *and* alpha *and* prompt set, it
  is not an experiment, it is a demo.

---

## 3. Repo contract

```
data/contrast/consequence.jsonl   ← the only file we author from scratch
artifacts/directions/*.pt         ← unit vectors, the interface between everything
artifacts/activations/*.npz       ← cached once on GPU, reused forever on CPU
external/<repo>/                  ← cloned, run, NEVER imported
```

**Rule 1 — `src/` never imports `external/`.** Reference repos (`refusal_direction`,
`refusal-downstream-persona`) run as subprocesses in their own venvs. They communicate with
us only by writing a `.pt` file into `artifacts/directions/`. This is deliberate: it keeps
three unfamiliar codebases from becoming one integration problem.

**Rule 2 — every direction ships a sidecar `.json`** with: model id, layer, token position,
source contrast file, n_pairs, seed, git SHA, and the command that produced it. A `.pt` with
no sidecar is deleted, not debugged.

**Rule 3 — GPU only for stages 01 (cache activations) and 05 (generate).** Probes, cosines,
figures, and analysis run on CPU against cached artifacts. Before proposing a GPU run, check
whether the cached activations already answer the question. Pod time is the one cost that
scales with our sloppiness.

**Rule 4 — stages are resumable and artifact-keyed.** Re-running a stage with unchanged
inputs must be a no-op, not a recompute.

---

## 4. Fixed configuration

| | |
|---|---|
| Primary model | `Qwen2.5-7B-Instruct` — d_model 3584, 28 layers |
| Second model (drop if time-pressed) | `Llama-3.1-8B-Instruct` — d_model 4096, 32 layers |
| Framework | plain HuggingFace `transformers` + PyTorch forward hooks |
| Precision | bf16, frozen weights |
| Probes | `sklearn.linear_model.LogisticRegression`, CPU |
| Judge | OpenAI API, `gpt-4.1-mini`-class bulk classifier, structured outputs |

**Do not add TransformerLens or nnsight.** Zhong's code — our closest template — is plain HF
plus hooks, and a hook is ~15 lines. A second framework buys nothing and costs fp32 caching
OOMs and porting friction.

**There is no training in this project.** No fine-tuning, no LoRA, no checkpoints. If a
proposed step involves a gradient step on model weights, it is out of scope.

---

## 5. Working with me

I have a CS background and solid classical ML: NLP, embeddings, Random Forests. I have **not
worked with deep learning internals before**. So:

- When introducing a transformer or mech-interp concept, anchor it to embeddings, linear
  classifiers, or feature engineering. See the `explain-for-my-background` skill.
- Do not write code I would not be able to defend to a facilitator. If a step is subtle,
  explain the mechanism *before* writing the implementation.
- Prefer 20 obvious lines to 5 clever ones.
- When I ask "why," I want the reasoning, not reassurance.

**Before any new experiment, state four things:** the hypothesis, the result expected if it
holds, the result that would change our mind, and whether a cheaper version would give the
same signal. If you cannot answer the third, the experiment is not ready.

---

## 6. Reference material

Papers (in project folder / arXiv):
- Arditi et al., *Refusal Is Mediated by a Single Direction* — 2406.11717 → source of `r̂`
- Zhong & Li, *Refusal Lives Downstream of Persona* — 2606.26161 → source of `v_MP`, the
  three-way metric, projection knockout, and our exact two models
- Marks & Tegmark, *The Geometry of Truth* — 2310.06824 → the methodological template
- Kirch et al., *What Features in Prompts Jailbreak LLMs* — 2411.03343 → the scoping caution
- *From Adversarial Poetry to Adversarial Tales* — 2601.08837 → proposes this exact test,
  runs no experiments; we are its first empirical check

Repos in `external/`:
- `andyrdt/refusal_direction` — run once for a reference `r̂` to validate our extraction
- `violazhong/refusal-downstream-persona` — run once for `v_MP`; read their appendix for layers
- `saprmarks/geometry-of-truth` — read for understanding; do not wire into the pipeline

Datasets: AdvBench (de-duplicate, cap the ~46% cyber slice before extracting `r̂`), Alpaca,
StrongREJECT (`walledai/StrongREJECT`), JailbreakBench, XSTest (over-refusal control).

---

## 7. Hard limits

- **Never author novel jailbreaks.** Only published attacks from published suites. This is
  defensive interpretability; the write-up reports aggregate numbers and mechanisms, and
  redacts harmful completions.
- Never commit model weights, activation caches, or API keys. `artifacts/` and `data/raw/`
  are gitignored; sidecar `.json` files are committed.
- Never edit anything under `external/`. If a reference repo needs a change, wrap it.
- Update `logs/research_log.md` at the end of every working session — it is a course
  requirement and the facilitators' only window into progress.
