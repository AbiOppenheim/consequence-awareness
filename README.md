# Do Jailbreaks Work by Making Models Treat Harm as Fiction?

**Probing and steering a "consequence-awareness" direction in 7–8B chat models.**

*A 5-week mechanistic-interpretability project for BlueDot's Technical AI Safety course.*

## The question

Fiction, role-play, and "hypothetical" framings reliably weaken model refusals. Existing accounts say jailbreaks suppress a refusal direction (Arditi et al., 2024) or shift activations into safe-looking regions (JailbreakLens, 2024). This project tests a third mechanism: that these framings shift the model's internal sense of whether the situation is **real** — whether its output has consequences — and that this signal gates refusal.

A 2026 interpretability agenda (*From Adversarial Poetry to Adversarial Tales*) proposes exactly this probe-and-steer test for narrative jailbreaks but runs no experiments. **This is its first empirical test.**

## Method (5 steps)

1. **Contrastive dataset** — ~100–200 minimal pairs, same task, differing only in real-vs-hypothetical framing; ≥10 diverse templates with held-out phrasings so the probe can't learn surface vocabulary.
2. **Probe** — per-layer linear probes on residual-stream activations of Qwen2.5-7B-Instruct (Llama-3.1-8B as robustness check); extract candidate direction `v_C` by difference-in-means.
3. **Correlational test** — do published fiction-framed jailbreaks actually move the model along `v_C`?
4. **Causal test** — steer toward "real" (`h ← h + α·v_C`) during jailbreaks; measure refusal restoration vs. random-direction and refusal-direction baselines, using a three-way refusal/bypass/degenerate metric.
5. **Distinctness** — cosine similarity + differential ablation of `v_C` vs. the refusal direction `r̂` (Arditi) and the compliant-persona direction `v_MP` (Zhong & Li, 2026).

**Every outcome is a result:** a distinct causal direction → new defense target; "it's persona relabeled" → useful negative result; "readable but causally inert" → defense work stays on the refusal pathway.

## Repo layout

```
docs/       project plan, full tutorial/design doc, compute estimation
src/        extraction, probing, steering, geometry (novel ~15%; the rest is forked)
data/       framing templates + dataset build rules (no harmful content stored here)
notebooks/  exploration
results/    figures and aggregate numbers only
```

Forked foundations: [`andyrdt/refusal_direction`](https://github.com/andyrdt/refusal_direction) · [`violazhong/refusal-downstream-persona`](https://github.com/violazhong/refusal-downstream-persona) · [`saprmarks/geometry-of-truth`](https://github.com/saprmarks/geometry-of-truth)

## Status

- [x] Research question locked; experimental design + week-by-week plan (`docs/project_plan.md`)
- [x] Full design doc / tutorial with failure modes priced in (`docs/tutorial.md`)
- [x] Compute & resource estimation (`docs/compute_estimation.md`)
- [x] Dataset construction rules + starter framing templates (`data/`)
- [ ] Week 1 — pipeline validated; Arditi `r̂` reproduced on Qwen
- [ ] Week 2 — consequence dataset + probes; correlational read
- [ ] Week 3 — steering experiment vs. baselines; direction geometry
- [ ] Week 4 — red-teaming (paraphrase controls, XSTest, second model)
- [ ] Week 5 — public blog post

## Responsible use

This is defensive interpretability. The project uses only published harmful-prompt benchmarks (StrongREJECT, JailbreakBench, HarmBench) and published attacks; it develops and releases **no new jailbreaks**. Reported outputs are mechanisms, defense implications, and aggregate numbers — harmful completions are never stored in this repo or shown in the write-up.

## Key references

- Arditi et al. (2024), *Refusal in LLMs Is Mediated by a Single Direction* — [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
- Zhong & Li (2026), *Refusal Lives Downstream of Persona in Chat Models* — [arXiv:2606.26161](https://arxiv.org/abs/2606.26161)
- Marks & Tegmark (2023), *The Geometry of Truth* — [arXiv:2310.06824](https://arxiv.org/abs/2310.06824)
- Rimsky et al. (2024), *Contrastive Activation Addition* — [ACL 2024](https://aclanthology.org/2024.acl-long.828/)
- Kirch et al. (2025), *What Features in Prompts Jailbreak LLMs?* — probe-transfer caution
- Qi et al. (2024), *Safety Alignment Should Be Made More Than Just a Few Tokens Deep*
