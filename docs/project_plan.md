# Do Jailbreaks Work by Making Models Treat Harm as Fiction?

**Field:** Mechanistic Interpretability / Adversarial Robustness · **Duration:** 5 weeks, ~30 hours · **Compute:** single GPU (RunPod), 7–8B open model

## Research question

Some jailbreaks — role-play, hypothetical framing, fiction — may work by shifting a model's internal sense of whether the situation is *real*, rather than by directly suppressing refusal. This project trains a linear probe to find a "consequence-awareness" direction in the residual stream and then tests, causally, whether steering that direction toward "real" restores refusal during fiction-framed jailbreaks.

## Why it matters (theory of change)

Jailbreaks remain a live pathway to misuse of increasingly capable models. Existing accounts say jailbreaks suppress a refusal signal (Arditi et al., 2024) or shift activations into safe-looking regions (JailbreakLens, 2024); recent work shows harmfulness and refusal are encoded separately (2025). A 2026 interpretability agenda ("From Adversarial Poetry to Adversarial Tales") proposes exactly this probe-and-steer test for narrative jailbreaks **but runs no experiments** — this project is its first empirical test. Both outcomes are informative: if steering toward "real" restores refusal, consequence awareness is a causal mechanism and a concrete training target complementing refusal-direction defenses; if not, the mechanism is ruled out and defense work points back at the refusal pathway.

## Method overview

1. **Contrastive dataset:** ~100–200 minimal prompt pairs sharing the same task but differing only in real-vs-hypothetical framing, with diverse templates and held-out phrasings to prevent the probe learning surface vocabulary.
2. **Probe:** per-layer linear probes on residual-stream activations of Qwen2.5-7B or Llama-3.1-8B (TransformerLens / nnsight); report layer-wise accuracy on held-out pairs.
3. **Correlational test:** read the probe on fiction-framed jailbreaks from a public attack suite — does the model internally register "hypothetical"?
4. **Causal test:** steer along the direction toward "real" during jailbreaks; measure refusal restoration against two baselines (random direction; the refusal direction), sweeping steering strength and reporting output coherence.
5. **Novelty extension:** compare the consequence direction to the refusal direction (cosine similarity, differential ablation) — same mechanism or genuinely distinct?

## Weekly plan (mapped to course deliverables)

| Week | Milestone | Banked output |
| --- | --- | --- |
| 1 | Env setup; reproduce Arditi refusal direction; question locked | Pipeline validated; first figure |
| 2 | Dataset + probes trained; correlational read on jailbreaks | First results (complete finding on its own) |
| 3 | Steering experiment vs. baselines; direction comparison | Main causal result; blog outline |
| 4 | Red-team own claims (paraphrase controls, bug hunt); write | Blog post MVP |
| 5 | Polish figures, limitations, next steps | Final post + presentation |

## Risks and mitigations

**Probe learns template artifacts** → diverse templates, paraphrase controls, held-out phrasings. **Steering breaks fluency before restoring refusal** → strength sweep, coherence reporting. **Time overrun** → every experiment runs in minutes-to-hours on one GPU; each week banks a self-contained result, so the post can ship from any stage. **Degenerate probe at week 2** → pivot shares ~80% of code: truth-probe generalization under role-play (Marks & Tegmark style), or persona/shallow-alignment KL analysis.

## Deliverable

A public blog post making 1–3 evidence-backed claims: probe accuracy, whether fiction jailbreaks shift the internal "real" signal, and whether that shift is causal — with honest limitations and concrete next steps.
