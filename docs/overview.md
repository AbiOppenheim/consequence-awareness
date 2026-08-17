# Does fiction jailbreak a model by changing what it thinks is *real*?

**A two-page overview of where this project stands.**
Model: `Qwen2.5-7B-Instruct` (28 decoder blocks, d_model 3584). All numbers below are from
`artifacts/results/*.json`, each stamped with its git SHA and input hashes.

---

## 1. The question, in familiar terms

A transformer's **residual stream** is a running sum of vectors — one per token, one per layer.
Think of it as a contextual embedding that gets rewritten 28 times. The bet of this literature is
that human-legible concepts live in it as *linear directions*: to ask "does the model represent X?"
you fit a linear classifier on those activations, exactly as you would probe word embeddings for
gender or sentiment.

Arditi et al. (2024) showed refusal is one such direction, `r̂`. Fiction and role-play framings
reliably defeat refusal. The standard story is that they **suppress** `r̂`. This project tests a
different mechanism, proposed but never tested by *Adversarial Poetry to Adversarial Tales* (2026):
that such framings shift an upstream representation of **whether the situation is real** — and that
*this* is what refusal is conditioned on.

We extract a candidate **consequence direction** `v_C` and ask two things:

1. **Existence + distinctness** — is real-vs-hypothetical linearly decodable, does it generalize to
   framing templates never seen in training, and is it a different axis from `r̂` (refusal) and
   `v_MP` (compliant persona, Zhong & Li 2026)?
2. **Causality** — does adding `v_C` back during a fiction-framed attack restore refusal, beyond
   what a random direction explains?

A negative answer to either is a publishable result. Nothing here is optimized toward a positive one.

> **Terminology warning.** "The gate" in the notebook means the `r̂` *reproduction sanity check*
> (§2 below) — not a gating mechanism. Whether `v_C` gates refusal is the open question, not a
> label we have earned.

---

## 2. What has been done

**Stage A — validate the machinery (the "`r̂` gate").** Before trusting any novel result, we
reproduced Arditi's refusal direction with our own difference-in-means code on the same 256 prompts
(128 harmful / 128 harmless) and compared vectors. At the correct layer offset,
**cos = 0.9999** across 27 layers. Its red-team: cross-layer cos is only 0.14 (so not everything is
high), our own adjacent layers sit at 0.74 (so the test has power), a **label permutation collapses
to null** (200 permutations, null mean −0.01, sd 0.42; observed p < 0.005), and the top-10
dimensions carry only 6.6% of the direction's mass (so it is not a handful of huge residual
dimensions masquerading as a concept). This is a methods footnote, not a
finding — but a failure here would have invalidated everything downstream.

**Stage B — the contrast set.** 2,000 prompts built from 40 **benign** tasks and 140 framing
templates spanning 51 semantic "routes" — each template paired with 14–15 of the tasks (a
subsample, not the full cross-product), perfectly balanced at 714/714 train and 286/286 held-out.
Task content is matched inside every pair, so the only thing that varies is framing. The critical
discipline:
**held-out is by framing template, never by example**, and the 20 held-out routes are **disjoint**
from the 31 training routes — held-out reaches the same real/hypothetical distinction through
mechanisms (`liability`, `allegory`, `parody`, `dry_run`) unseen in training. Splitting examples
within shared templates would measure vocabulary memorization instead of a concept.

**Stage C — extract `v_C` and pick a layer honestly.** Difference-in-means (mean of "real" minus
mean of "hypothetical") at the last prompt token, per layer. The layer was chosen by
**grouped 5-fold CV inside the training templates only** — groups are templates, so every fold
validates on unseen templates. Selected **L18**; the curve is essentially flat (spread 0.004).
Picking the layer that maximized held-out accuracy would have been selecting on the test set.

**Stage D — the held-out reveal (run once, at L18, 572 rows / 40 templates).**

| readout | AUC |
|---|---|
| trained probe | **0.973** |
| `v_C` projection (no fitting at all) | **0.935** |
| best surface-text baseline (TF-IDF uni+bigram) | 0.847 |
| bag-of-words baseline | 0.753 |
| random direction | 0.408 |

Its red-team: every one of the 20 held-out routes scores ≥ 0.765 (weakest: `sandbox` 0.765,
`privacy` 0.859), so the average hides no dead route; and the random-direction null was run as a
**distribution** (100 draws) rather than one draw — p95 = 0.698, so a lucky random direction can
reach 0.70 AUC. `v_C`'s 0.935 clears that.

**Stage E — geometry against the two rival explanations.** At L18, against a random-direction null
band of ±0.032:

- `cos(v_C, r̂) = 0.085` — above the null, so the overlap is real, but tiny. `v_C` is **nearly
  orthogonal to refusal**.
- `cos(v_C, v_MP) = 0.24` (system-prompt framing) / **0.29** (user-turn framing) — the persona axis
  is the sharper rival, and shares ~8% of variance with `v_C`. Distinct, but not unrelated.
- The two `v_MP` framings agree with each other at 0.91, so that direction is stable and the
  comparison is not an artifact of prompt structure.

**Stage F — the causal sweep.** Activation addition (`h ← h + α·v̂`) at a single layer, α expressed
as a fraction of the measured median residual norm so doses are comparable across depth. Headline α
is fixed *in code* as the largest dose still yielding ≤5% degenerate output. Every generation gets
a **three-way** label (refusal / bypass / degenerate) — never a refusal rate alone, because high α
produces gibberish that a two-way metric would score as a successful defense. 5,050 judged
generations.

---

## 3. What we found

**Claim 1 holds.** `v_C` exists, generalizes to unseen framing *routes*, and is its own axis.
Two supporting details: the *unfitted* projection AUC **rises** with depth (0.855 at L8 → 0.942 at
L20), which surface-token matching would not do; and orthogonalizing `v_C` against `r̂` leaves
held-out readout untouched (0.9351 → 0.9356), so the information is not hiding in the refusal
component. State the margin honestly, though: the best surface baseline reached 0.847, not chance.

**Claim 2 holds at L18 — and this is the most important result, currently under-reported.**
At the headline dose (0.5×‖h‖, baseline refusal 0.72):

| condition | Δ refusal on attacks | refusal on 250 *safe* prompts |
|---|---|---|
| random direction (the null) | −0.05 | 1.6% |
| `v_C` | **+0.23** | **5.6%** |
| `v_C` orthogonalized against `r̂` | **+0.25** | not yet run |
| `r̂` (reference) | +0.27 | **63.6%** |
| *(baseline)* | — | 2.4% |

The null is clean. `v_C` recovers ~85% of what steering refusal itself achieves — **while barely
touching benign helpfulness**, where `r̂` refuses nearly two thirds of harmless questions. `r̂` is a
blunt global refusal knob; `v_C` is selective. That selectivity gap, not the +0.23 alone, is the
case for `v_C` as a defense target. And because the orthogonalized `v_C` is *slightly stronger*
(+0.25), the effect is not working through its `r̂` sliver.

**But it is layer-specific.** At **L22**, at that layer's own equivalent dose, `v_C` moves refusal
**0.00** while `r̂` still moves +0.17. Meanwhile `v_C` *reads out* marginally better at L20–L24 than
at L18. **Readout quality and causal efficacy dissociate** — a direction can be maximally legible
where intervening on it does nothing. Any claim must name its layer.

**The negative arm is not a clean test.** Pushing toward "hypothetical" moves refusal only −0.01,
which looks like "`v_C` doesn't control refusal". But at that same |α| the negative arm is already
**24% degenerate** against the positive arm's 1%, and it makes the model refuse **34.8%** of *safe*
prompts. So the negative direction is mostly damaging fluency, not testing the axis. The asymmetry
in degradation is itself a finding, and the one-sidedness question stays open.

---

## 4. What is still open — in priority order

1. **The correlational test has never run** (README method step 3, notebook §26). We have shown that
   steering `v_C` changes refusal; we have **not** shown fiction attacks move `v_C` at all. If
   jailbreaks sit exactly where plain harmful prompts sit on this axis, the proposed mechanism is
   wrong however well steering works. Stage 12 is written and partly run — but
   `eval_projection_L18.json` still reports `headline_fiction_vs_harmful_plain: null` with
   `fiction_jailbreaks` and `xstest` missing (needs one ~1-min GPU pass). What we *do* have:
   harmfulness alone moves the coordinate **−0.28** (plain harmful sits at 0.19, plain harmless at
   0.47, where 0 = mean real framing and 1 = mean hypothetical), so a smaller jailbreak gap would be
   content, not framing. This also decides the one-sidedness above.
2. **`v_MP` has never been steered — the missing control.** Random Gaussians are near-orthogonal to
   everything in 3,584 dimensions, so "random does nothing" does *not* rule out "any *extracted*
   direction at this dose restores refusal". The persona direction is already minted. ~3 GPU-min.
3. **The judge is an unvalidated instrument.** It produces every causal number here and has had **no
   human agreement check**. Two of its bugs were already caught by cross-checking its own fields —
   including a rubric collision that scored 248/250 helpful answers to safe prompts as "refusals"
   (now fixed by disjoint harmful/benign rubrics). Hand-audit ~100 verdicts before the write-up.
4. **`v_C` orthogonalized against `r̂` has no XSTest run**, so its +0.25 lacks the over-refusal guard
   that makes the raw `v_C` number meaningful.
5. **A third steering layer**, to tell "peak at L18" from "monotone decay with depth".
6. **StrongREJECT is not quotable alone** — it contradicts its own label on ~7% of rows. Reported
   only over bypass rows, beside the three-way label.

**Scope, stated plainly.** One model, one attack family (fiction/role-play), two layers, n = 100 per
condition — so the SE on a difference of proportions is ~0.065 and +0.23 is ~3.5 SE: real, not
overwhelming. `v_C` was extracted entirely from **benign** tasks, which removes a harmfulness
confound from extraction but makes its transfer to harmful attacks a claim in itself. Kirch et al.
find jailbreak-relevant features are attack-family-specific, so nothing here generalizes to
attack families we did not test. Llama-3.1-8B is the planned second model and the first thing to
drop if time runs out — in which case we say so rather than implying generality.
