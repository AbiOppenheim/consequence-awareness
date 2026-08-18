# Does fiction jailbreak a model by changing what it thinks is *real*?

**A two-page overview of where this project stands.**
Model: `Qwen2.5-7B-Instruct` (28 decoder blocks, d_model 3584). All numbers below are from
`artifacts/results/*.json`, each stamped with its git SHA and input hashes.

---

## 1. The question, in familiar terms

A transformer's **residual stream** is a per-token accumulator: the token embedding, plus the sum
of everything the 28 attention and MLP blocks have written into it so far. Each block *reads* the
stream and *adds* its output back — it never overwrites. That additivity is why this whole
literature works: if a concept is a linear direction, you can read it off with a dot product and
push on it by adding a vector. We read the stream at one layer boundary, at the **last prompt
token**.

The bet is that human-legible concepts live there as *linear directions*: to ask "does the model
represent X?" you fit a linear classifier on those activations, exactly as you would probe word
embeddings for gender or sentiment. Two readouts matter here and they are not the same thing — a
**trained probe** (fitted weights) and an **unfitted difference-in-means direction** (mean of one
class minus mean of the other, no fitting at all). The second is the stronger claim.

Arditi et al. (2024) showed refusal is one such direction, `r̂`. Fiction and role-play framings
reliably defeat refusal, and the usual assumption is that they **suppress** `r̂` — but that is an
extrapolation. Arditi's suppression result (§5.1) is for **GCG-style adversarial suffixes**, on
Qwen-1.8B-Chat, measured as cosine similarity of last-token activations: correlational, one attack
family, and not fiction framing. This project tests a different mechanism, proposed but never
tested by *Adversarial Poetry to Adversarial Tales* (2026): that narrative framings shift an
upstream representation of **whether the situation is real** — and that *this* is what refusal is
conditioned on.

We extract a candidate **consequence direction** `v_C` and ask two things:

1. **Existence + distinctness** — is real-vs-hypothetical linearly decodable, does it generalize
   to framing templates never seen in training, and is it a different axis from `r̂` (refusal) and
   `v_MP` (compliant persona, Zhong & Li 2026)?
2. **Causality** — does **activation addition** along `v_C` (`h ← h + α·v̂`, Rimsky et al. 2024)
   during a fiction-framed attack restore refusal, beyond what a random direction explains?

A negative answer to either is a publishable result. Nothing here is optimized toward a positive one.

> **Terminology.** "The gate" in the notebook means the `r̂` **extraction-validation check**
> (§2 below) — not a gating mechanism. And we avoid "gate" for `v_C`: Zhong uses that word in a
> specific sense (persona gates whether refusal is *expressed*, at the late-layer expression
> stage). What we show is that adding `v_C` **shifts the refusal decision** — sufficiency, not
> gating.

---

## 2. What has been done

**Stage A — validate the machinery (the "`r̂` gate").** Before trusting any novel result, we
reproduced Arditi's refusal direction with our own difference-in-means code on the same 256 prompts
(128 harmful / 128 harmless) and compared vectors. Their layer indexing is offset from ours by one
block, which we found by **sweeping** offsets rather than assuming: −1, 0, +1, +2 score
0.58 / 0.74 / **0.9999** / 0.74. Reporting the losing offsets is what makes 0.9999 mean anything.
Its red-team: cross-layer cos is only 0.14 (so not everything is high), our own adjacent layers sit
at 0.74 (so the test has power), a **label permutation collapses to null** (200 permutations, null
mean −0.01, sd 0.42; observed p < 0.005), and the top-10 dimensions carry only 6.6% of the
direction's mass (so it is not a handful of huge residual dimensions masquerading as a concept).
This is a methods footnote, not a finding — but a failure here would have invalidated everything
downstream.

**Stage B — the contrast set.** 2,000 prompts built from 40 **benign** tasks and 140 framing
templates spanning 51 semantic **routes**. A *route* is the mechanism by which a template signals
real or hypothetical — `liability`, `allegory`, `dry_run` — as opposed to its wording. Each
template is paired with 14–15 tasks (a subsample, not the full cross-product), giving 1,000 pairs,
perfectly balanced at 714/714 train and 286/286 held-out. Task content is matched inside every pair
(0 mismatches), so the only thing that varies is framing. The critical discipline: **held-out is by
framing template, never by example** — 100 train templates, 40 held out — and the 20 held-out
routes are **disjoint** from the 31 training routes. Held-out reaches the same real/hypothetical
distinction through mechanisms unseen in training. Splitting examples within shared templates would
measure vocabulary memorization instead of a concept.

**Stage C — extract `v_C` and pick a layer honestly.** Difference-in-means (mean "real" minus mean
"hypothetical") at the **prompt-end token** — the same position convention used for all four
directions, which is what makes their cosines comparable. The layer was chosen by **grouped 5-fold
CV inside the training templates only** — groups are templates, so every fold validates on unseen
templates. Selected **L18**. Picking the layer that maximized held-out accuracy would have been
selecting on the test set.

*State this weakness plainly:* train CV AUC runs 0.9955–0.9994 across all nine candidate layers.
The selection metric is **at ceiling everywhere**, so the flat curve is saturation, not
reassurance — L18 was effectively drawn from nine indistinguishable candidates. That was harmless
while we expected a smooth effect with depth. It is not harmless now that the causal effect turns
out to be dead at L22 (§3).

**Stage D — the held-out reveal (run once, at L18, 572 rows / 40 templates).**

| readout | AUC |
|---|---|
| trained probe | **0.973** |
| `v_C` projection (no fitting at all) | **0.935** |
| best lexical baseline (TF-IDF uni+bigram) | 0.847 |
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
  is the sharper rival. The component of `v_C` along `v_MP` accounts for 8.6% of its squared norm.
  Distinct, but not unrelated.
- The two `v_MP` framings agree with each other at 0.91, so that direction is stable and the
  comparison is not an artifact of prompt structure.
- **`v_MP` is our reimplementation of Zhong's method** (their repo ships a partial release), which
  we flagged as unvalidated. It is checkable after all: Zhong's Table 8 reports
  cos(`v_MP`, `r̂`) = **−0.279** for Qwen2.5-7B at L20; ours is **−0.286** at the same layer. That
  reproduces the one number their paper publishes about this vector, to within 0.007.

**Stage F — the causal sweep.** Activation addition at a single layer, α expressed as a fraction of
the measured median residual norm (‖h‖ = 71.4 at L18) so doses are comparable across depth.
Headline α = 35.7 = 0.5·‖h‖, fixed *in code* as the largest dose still yielding ≤5% degenerate
output; coherence collapses at 1.0·‖h‖. Every generation gets a **three-way** label
(refusal / bypass / degenerate) — never a refusal rate alone, because high α produces gibberish
that a two-way metric would score as a successful defense. **6,950 judged generations** across 25
(condition, α) cells at L18 plus the L22 and over-refusal sets.

---

## 3. What we found

**Claim 1 holds.** `v_C` exists, generalizes to unseen framing *routes*, and is its own axis.
Two supporting details: the *unfitted* projection AUC **rises** with depth (0.855 at L8 → 0.942 at
L20, then plateaus: 0.940 at L22, 0.941 at L24), which surface-token matching would not do; and
orthogonalizing `v_C` against `r̂` — component removal from the direction vector,
`v⊥ = v_C − (v_C·r̂)r̂` renormalized, *not* Arditi's ablation of activations — leaves held-out
readout untouched (0.9351 → 0.9356), so the information is not hiding in the refusal component.
State the margin honestly, though: the best lexical baseline reached 0.847, not chance.

**Claim 2 holds at L18, and the case rests on selectivity.** Baseline refusal is 0.72 on attacks
and 0.024 on 250 safe XSTest prompts. Selectivity = attack refusal gained per point of over-refusal.

| condition | Δ refusal (attacks) | Δ refusal (safe) | selectivity |
|---|---|---|---|
| `+v_C` | **+0.23** | **+0.03** | **7.2** |
| `+v_C ⊥ r̂` | **+0.25** | *not run* | — |
| `+r̂` (reference) | +0.27 | +0.61 | 0.44 |
| `+v_MP` (user-turn) | **−0.25** | +0.01 | — |
| `−v_MP` (user-turn) | +0.14 | +0.47 | 0.30 |
| `−v_MP` (system) | +0.08 | +0.52 | 0.15 |
| random direction (the null) | −0.05 | −0.01 | — |

The null is clean. `v_C` recovers ~85% of what steering refusal itself achieves — **at a twentieth
of the cost to helpfulness**, where `r̂` restores refusal by refusing 61% more harmless prompts.
That is an off switch, not a defense. And because the orthogonalized `v_C` is *slightly stronger*
(+0.25), the effect is not working through its `r̂` sliver.

**The specificity control is the single strongest result.** A random Gaussian is near-orthogonal to
every real feature in 3,584 dimensions, so "random does nothing" was close to guaranteed and ruled
out nothing. `v_MP` is the real control: same pipeline, same layer, same dose, 0.29-correlated with
`v_C`, and **statistically tied** with it at predicting which attacks succeed (0.640 vs 0.667,
paired 95% CI [−0.040, +0.096]). Readout cannot separate them at all. Intervention separates them
completely: at +α they move refusal in **opposite directions**, and reversed, `v_MP` restores
refusal only by becoming a blunt refuse-everything switch — 24× less selective than `v_C`. This is
distinctness evidence a cosine cannot give. *Caveat:* Zhong steers `v_MP` at L20 and locates the
Qwen persona effect in L20–L22; we steer it at L18 to hold the layer fixed against `v_C`, so the
control runs outside the window where the rival is documented to act.

**But it is layer-specific.** At **L22**, at that layer's own equivalent dose, `v_C` moves refusal
**+0.00** while `r̂` still moves +0.17 and the null still moves −0.05. Meanwhile `v_C` *reads out*
marginally better at L20–L24 than at L18. **Readout quality and causal efficacy dissociate** — a
direction can be maximally legible where intervening on it does nothing. Any claim must name its
layer. (The L22 arm has no over-refusal control, which does not undermine a null but means the two
layers were not evaluated equally.) Note the contrast with Zhong: their persona→refusal coupling
lives in a *late* window and early intervention fails, where `v_C` shows the opposite depth
profile — a third angle on "`v_C` is not persona."

**Steering is one-sided, and the projection explains why.** Pushing toward "hypothetical" moves
refusal only −0.01. That looks like "`v_C` doesn't control refusal", but the attacks have nowhere
left to go: **84%** of them (85.6% at n = 500) already sit past the mean-hypothetical pole. The
honest framing of that number is against ordinary hypothetical prompts, **47%** of which also sit
past their own mean — not against zero. What the projection does *not* explain is that at the same
|α| the negative arm is **24% degenerate** against the positive arm's 1%, and makes the model refuse
**34.8%** of *safe* prompts. Pushing toward "hypothetical" should not make a model more
restrictive; `−v_C` most likely lands off-manifold rather than on a clean signed axis, but we have
not shown that.

---

## 4. What is still open — in priority order

1. **The judge is an unvalidated instrument.** It produces every causal number here and has had
   **no human agreement check**. Two of its bugs were already caught by cross-checking its own
   fields — including a rubric collision that scored 248/250 helpful answers to safe prompts as
   "refusals" (fixed by disjoint harmful/benign rubrics). A full second judging pass over identical
   generations gives mean |drift| **0.013** (max 0.080 over 21 conditions), so effects are ~18×
   judge noise — but the random null itself moved +0.00 → −0.05 between passes, which is why every
   effect is quoted against a band, not a point. That measures the judge's **consistency**; it says
   nothing about its **accuracy**. Hand-audit ~50 verdicts before the write-up.
2. **`v_C` orthogonalized against `r̂` has no XSTest run**, so its +0.25 lacks the over-refusal
   guard that makes the raw `v_C` number meaningful.
3. **The correlational test is weak, not absent.** Within 495 attacks, `v_C` predicts which succeed
   at AUC **0.640** (bootstrap CI [0.587, 0.692]) but **p = 0.07** against 200 random directions —
   a real in-sample association, only marginally specific to `v_C`, and the persona rival scores
   0.667 on the same split. Not a length artifact (0.632 after residualising on ‖h‖). The obvious
   version of this test — attacks vs. plain harmful prompts — looks emphatic (0.971) and is
   **untestable**: the sets differ ~10× in length, `r̂` separates them better (0.997), and the raw
   vector norm alone reaches 0.899. Kept as a labelled diagnostic, never a finding.
4. **The `−v_C` over-refusal (34.8%) is unexplained** — see §3.
5. **The L18 layer choice was not identified by the data** (Stage C), and the causal effect is
   layer-specific.
6. **StrongREJECT is not quotable alone** — it contradicts its own label on **8.0%** of rows
   (376/4,700). Reported only over bypass rows, beside the three-way label.

**Scope, stated plainly.** One model, one attack family (fiction/role-play), two layers for the
causal test and one for the headline claim, n = 100 per (condition, α) cell. The design is
**within-prompt** — the same 100 attacks, greedy decoding, seed 0, under every condition — so the
right test is McNemar or a paired bootstrap, not the unpaired two-proportion SE; even the
conservative unpaired figure at the observed rates is ~0.050, putting +0.23 at ~4.6 SE. `v_C` was
extracted entirely from **benign** tasks. That cuts both ways: it removes a harmfulness confound
from extraction, and it makes transfer to harmful attacks a substantive claim in its own right, for
which the +0.23 is the only evidence. Kirch et al. find jailbreak-relevant features are
attack-family-specific, so nothing here generalizes to families we did not test.

**Deliberately not run, and stated as limitations rather than gaps:** a second model
(Llama-3.1-8B), a second attack family, a third steering layer, and differential ablation —
superseded by the bidirectional persona control, which answers specificity rather than necessity.
Each would add a limitation to the write-up rather than change a conclusion.
