---
name: explain-for-my-background
description: Explain transformer internals, mechanistic interpretability concepts, or unfamiliar code by anchoring them to classical ML — embeddings, linear classifiers, feature engineering, Random Forests. Use this whenever the user asks what something means, why a step works, how a reference repo's code operates, or says they are confused, as well as before implementing any step involving a concept they have not used before. The user has a CS background and solid classical ML/NLP but has not worked with deep learning internals, so default to bridging rather than assuming.
---

# Explaining to a classical-ML background

The user knows: embeddings, cosine similarity, linear classifiers, train/test splits,
overfitting, feature engineering, Random Forests, NLP pipelines.

The user has not worked with: transformer internals, forward hooks, activation-space
interventions, PyTorch training loops, the mech-interp literature's vocabulary.

The gap is not intelligence or rigor — it is unfamiliar names for familiar objects. Almost
everything in this project has a classical analogue. Lead with the analogue.

## The bridges that work

| Mech-interp concept | Anchor |
|---|---|
| Residual stream | A running embedding at each token position, updated additively by every layer. The layer-ℓ, last-token vector is "an embedding of everything the model has figured out so far." |
| Linear representation hypothesis | `king − man + woman ≈ queen`, moved from word vectors to internal activations. Same claim: concepts are directions. |
| Difference-in-means direction | Class-mean difference — a nearest-centroid classifier's decision axis. |
| Linear probe | Logistic regression on activations as features. It is literally sklearn. |
| Probe ≠ causal | Feature importance ≠ causal effect. A Random Forest can rank a feature highly that changes nothing when you intervene on it. |
| Steering (activation addition) | Editing a feature value at inference and re-running — an intervention, not a prediction. |
| Projection knockout | Feature ablation, but on a continuous direction rather than a column. |
| Forward hook | A callback registered on a layer, fired mid-forward-pass, that can read or overwrite that layer's output. |
| Layer sweep | Hyperparameter selection over depth — and it needs a held-out set for the same reason. |
| Attention head / MLP | Components that read from the residual stream, compute, and add their result back. |

## How to explain

1. **Anchor first, name second.** "This is a nearest-centroid decision axis; the literature
   calls it difference-in-means" — not the reverse.
2. **Concrete shapes.** Say `(n_prompts, 28, 3584)`, not "the activations." Shapes remove
   more confusion than prose.
3. **Say what would go wrong.** Understanding a mechanism means knowing its failure mode.
   "If the chat template isn't applied, you cache activations for a string the model never
   sees, and everything downstream is plausible garbage."
4. **One paragraph, then check.** Do not deliver a lecture. Explain, then ask whether to go
   deeper or move on.
5. **Never say "it just works."** If the reason is not clear, say the reason is not clear.

## Explaining reference-repo code

Three repos get read but only two get run. When walking through `external/` code:

- Say which of our four artifacts the code produces or consumes — that is the only thing
  that matters, since we never import it.
- Distinguish the method (five lines) from the scaffolding (selection sweeps, eval harnesses,
  config plumbing). Arditi's repo is large because it *selects* the best (layer, position)
  and ships scoring, not because the math is hard.
- Skip anything we are not using. Reading code we will not run is a poor use of a 30-hour
  budget.

## Before implementing anything new

State, briefly and in this order:

1. What this step measures.
2. Why it is the cheapest way to measure it.
3. What result would mean the hypothesis is wrong.
4. Which existing code does most of the work, and what is genuinely new here.

If (4) is "all of it," stop — this project's design assumes ~15% original code, and a step
that needs a lot of new code is usually a step that has been over-scoped.

## What the user must be able to defend

They will present this to facilitators and a cohort. So they need to be able to explain, in
their own words and without notes:

- why difference-in-means finds a causally relevant direction more reliably than a trained probe
- why a random-direction null is required for every intervention
- why held-out *templates* and not held-out *examples*
- why a probe result cannot support a causal claim
- what the three-way metric protects against

If code is written that they could not defend on those points, explain it before moving on.
That is not overhead; it is the deliverable.
