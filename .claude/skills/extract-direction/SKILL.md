---
name: extract-direction
description: Cache residual-stream activations and extract, save, load, validate, or compare directions (v_C, r_hat, v_MP, random nulls) via difference-in-means. Use this whenever the user mentions directions, difference-in-means, activation caching, the residual stream, layer sweeps, cosine similarity, geometry, probes on activations, or wants to reproduce Arditi's refusal direction. Also use it before writing any code that touches artifacts/directions/ or artifacts/activations/, since the sidecar-metadata contract and the token-position conventions are easy to get silently wrong.
---

# Extracting and comparing directions

## The one idea

Every direction in this project is the same object: a **unit vector in the residual stream**
(ℝ^3584 for Qwen2.5-7B). `r̂`, `v_MP`, `v_C`, and the random null differ only in which
contrast produced them. This is why external repos can be run as black boxes — they emit a
vector, and a vector is the whole interface.

For the user's background: the residual stream at layer ℓ, last token, is an embedding of
"everything the model has worked out so far." Difference-in-means is the same move as
`king − man + woman`, applied to activations instead of word vectors.

## The recipe

```python
def extract_direction(acts_pos, acts_neg):
    v = acts_pos.mean(0) - acts_neg.mean(0)
    return v / v.norm()
```

That is the entire method, and it is the one used by Arditi, Rimsky (CAA), Marks & Tegmark,
and Zhong. Simple difference-in-means identifies the directions most *causally* implicated in
behavior, often better than a trained probe — a probe optimizes for decodability, which is
not the same thing.

## Conventions that must not drift

- **Chat template applied**, always. Extract activations from the prompt as the model would
  actually receive it, not the raw string. A mismatch here is the most common silent bug and
  it produces plausible-looking garbage.
- **Token position: last prompt token**, before generation begins. Record it in the sidecar;
  if you ever sweep positions, that is a separate artifact, not an overwrite.
- **All layers cached in one pass.** Caching is cheap; re-running the GPU because you only
  saved layer 14 is not.
- **Cast to fp32 before averaging**, store as fp32. bf16 mean over hundreds of vectors loses
  meaningful precision.
- **Fit `v_C` on `split == "train"` rows only.** Held-out templates are for evaluation.

## Sidecar contract

Every `artifacts/directions/*.pt` has a twin `*.json`:

```json
{"name": "v_C", "model": "Qwen/Qwen2.5-7B-Instruct", "layer": 14,
 "token_position": "last_prompt", "contrast_file": "data/contrast/consequence.jsonl",
 "split": "train", "n_pairs": 152, "dtype": "float32", "seed": 0,
 "git_sha": "a1b2c3d", "command": "python scripts/02_extract_directions.py --layer 14"}
```

A `.pt` with no sidecar gets deleted. By Week 4 there will be a dozen vectors and no memory
of which is which — the sidecar is what makes the blog post's numbers traceable.

## Week-1 validation gate

Before trusting any extraction code, reproduce a known answer:

1. Run `external/refusal_direction` unmodified on Qwen in its own venv. Save its output as
   `artifacts/directions/r_hat_arditi.pt`.
2. Re-derive `r̂` with our own five-line difference-in-means on the same balanced contrast set.
3. Compute `cos(ours, arditi)` per layer.

**Target: > 0.9 at the layer Arditi's pipeline selects.** Above that, activation caching,
chat templating, token position, and extraction are all validated at once, and we own an
implementation we fully understand. Below that, there is a bug — and Week 1 is the cheapest
possible moment to find it. Do not proceed to `v_C` until this passes.

Note Arditi's pipeline *selects* a best (layer, position) pair across a sweep; ours does not.
Compare against his selected layer, not against a fixed one.

## Balancing before extracting `r̂`

AdvBench has heavy near-duplication and roughly 46% cyber-security prompts, which skews a
difference-in-means refusal direction toward "is this about computers." De-duplicate and cap
the cyber slice before extraction. Note the resulting category distribution in the sidecar.

## Geometry comparison

Report, per layer, against a random-direction null:

- `cos(v_C, r̂)` — is consequence just refusal?
- `cos(v_C, v_MP)` — is consequence just persona relabeled?
- `cos(r̂, v_MP)` — the reference point, already known from Zhong

**The null is not optional and it is not `0`.** In 3584 dimensions, random unit vectors have
cosine ≈ 0 with standard deviation ≈ 1/√3584 ≈ 0.017. Sample 1000 random pairs, report the
95th percentile, and plot it as a band. A cosine of 0.05 looks like nothing and is roughly
3σ — the band is what tells the reader which it is.

Interpretation: near the null band → distinct axes. Above ~0.7 → suspect the same mechanism
under two names, and say so plainly; that is a clean negative result, not a failure.

## Probes

`sklearn` `LogisticRegression` on cached activations, per layer, CPU, seconds. Report
accuracy on held-out **templates**.

State this in any write-up: **a probe reading a direction does not show the model uses it.**
A feature can be perfectly decodable and causally inert. That gap is the entire reason the
steering experiment exists as a separate test.
