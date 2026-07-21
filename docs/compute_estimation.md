# Compute & Resource Estimation — Rapid Grant Proposal

**Project:** *Do Jailbreaks Work by Making Models Treat Harm as Fiction?* (consequence-awareness probing & steering)
**Duration:** 5 weeks · **Scale:** single GPU, 7–8B open-weight models · **Prepared:** July 2026

> **Framing note for the grant form:** this project involves **no model training or fine-tuning**. All experiments are inference-time: caching residual-stream activations, extracting difference-in-means directions, training tiny linear probes (CPU-trivial), and steering/ablating via forward hooks. This is why the compute budget is small and defensible — every experiment runs in minutes-to-hours on one mid-range GPU.

---

## 1. Experimental Workflow

**Phase A — Environment & pipeline validation (Week 1)**
1. Provision GPU pod; install PyTorch, TransformerLens, nnsight, HF `transformers`; clone the three reference repos (`refusal_direction`, `refusal-downstream-persona`, `geometry-of-truth`).
2. Download gated models (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct; HF access requested in advance).
3. Reproduce Arditi's refusal direction `r̂` on Qwen (difference-in-means on a balanced harmful/harmless set: de-duplicated AdvBench/HarmBench vs. Alpaca) and Zhong's persona direction `v_MP` using their released scripts. This validates extraction, hooking, and scoring end-to-end.

**Phase B — Dataset construction & probing (Week 2)**
4. Build the consequence-contrast dataset: 100–200 minimal pairs, same task, real vs. hypothetical framing, ≥10 diverse framing templates, with held-out templates/paraphrases reserved for generalization testing. (CPU/LLM-assisted; near-zero GPU cost.)
5. Cache last-token residual-stream activations at every layer for all pairs (single batched forward pass per prompt; ~40–80 MB of cached vectors).
6. Extract `v_C` by difference-in-means per layer; train per-layer linear probes; report held-out accuracy including unseen-template generalization.

**Phase C — Correlational & geometric analysis (Weeks 2–3)**
7. Run published fiction/role-play jailbreak prompts through the model; read the `v_C` probe — does the model internally register "hypothetical"?
8. Geometry: cosine similarities among `v_C`, `r̂`, `v_MP`, and random-direction nulls, across layers.

**Phase D — Causal intervention (Week 3) — the GPU-heaviest phase**
9. Steering sweep: activation addition of `+α·v_C` ("toward real") during fiction jailbreaks, across ~5–6 steering strengths × 2–3 layer windows × ~100 attack prompts × ~256 generated tokens, against two baselines (random direction; refusal direction). ≈ 40–60 generation configurations.
10. Differential ablation (projection knockout) of `v_C` vs. `v_MP` to test mechanistic distinctness.

**Phase E — Robustness, evaluation & red-teaming (Week 4)**
11. Score every output with the three-way refusal/bypass/degenerate label + StrongREJECT score + coherence check (API-based LLM judging).
12. Over-refusal control on XSTest (250 prompts); paraphrase controls; shallow-alignment first-token check.
13. Replicate the core result on Llama-3.1-8B-Instruct (second model).

**Phase F — Write-up (Week 5)**
14. Final figures, limitations, blog post. Minimal GPU use (occasional re-runs for figures).

---

## 2. Technical & Compute Requirements

### 2.1 Models & architecture

| Model | Params | d_model | Layers | bf16 weights on disk/VRAM |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct (primary) | 7.6 B | 3,584 | 28 | ~15.2 GB |
| Llama-3.1-8B-Instruct (robustness check, Week 4) | 8.0 B | 4,096 | 32 | ~16.1 GB |

No architecture modification, LoRA, or fine-tuning — models run frozen in bf16 with forward hooks. Linear probes are ~4K-parameter classifiers trained on CPU in seconds.

### 2.2 Hardware requirements

| Resource | Minimum | Recommended | Rationale |
|---|---|---|---|
| **GPU VRAM** | 24 GB (RTX 4090 / A5000) | **48 GB (RTX A6000 / A40)** | 7–8B bf16 weights ≈ 15–16 GB. 24 GB works with HF + custom hooks and modest batches, but TransformerLens caches activations in fp32 by default and full-layer caching + batched generation during steering sweeps is far more comfortable at 48 GB. 48 GB removes an entire class of OOM debugging for a first-time setup — worth ~$0.15/hr extra. |
| **System RAM** | 32 GB | 64 GB | Model loading, activation caches spilled to CPU, dataset handling. |
| **CPU cores** | 8 vCPU | 16 vCPU | Tokenization, probe training, scoring orchestration. |
| **Storage** | 80 GB | **100 GB persistent volume** | Two models (~31 GB) + envs/repos (~20 GB) + datasets (<1 GB) + activation caches (~5 GB) + outputs/logs + headroom. **No training checkpoints exist in this project.** |

### 2.3 GPU-hour estimate (pod wall-clock, includes interactive development)

| Week | Activity | Pod-hours |
|---|---|---|
| 1 | Setup, model downloads, reproduce `r̂` and `v_MP` | 14 |
| 2 | Activation caching, `v_C` extraction, probes, correlational read | 14 |
| 3 | Steering sweeps + ablations (heaviest: ~40–60 generation configs) | 22 |
| 4 | Second model (Llama), XSTest, paraphrase controls, re-scoring | 20 |
| 5 | Figure re-runs, spot checks | 6 |
| | **Baseline total** | **~76 → budget 80** |

Note: pure compute is well under this; the estimate deliberately includes interactive debugging time with the pod live, which is the realistic cost driver for a first-time setup. Per-second billing means the pod is stopped between sessions.

### 2.4 Platform recommendations (beginner-friendly)

| Platform | Est. cost | Pros | Cons |
|---|---|---|---|
| **RunPod (recommended)** — RTX A6000 48GB Secure Cloud @ $0.49/hr; RTX 4090 24GB @ $0.34/hr Community | ~$40 for 80 hrs (A6000) | Per-second billing; persistent network volumes ($0.07/GB/mo) so environment survives pod shutdown; one-click PyTorch/Jupyter templates; large community docs; already named in the project plan | Storage billed separately even when pod is off; Community Cloud can have availability gaps |
| **Lambda Labs** — A6000/A100 on-demand | ~$60–80 for 80 hrs | Very clean UX, pre-installed Lambda Stack (PyTorch ready), reliable data-center hardware | Fewer cheap 24–48 GB options; instances sometimes sold out; storage persistence less flexible than RunPod volumes |
| **Google Colab Pro+** — $50/mo, A100 40GB when available | $100 for ~2 months | Zero setup, notebook-native, good for Week-1 learning and probe prototyping | GPU allocation not guaranteed; session time limits kill long steering sweeps; poor fit for repo-based pipelines and persistent environments — viable only as a backup/scratchpad |

(Vast.ai is cheaper still but its marketplace variability — mixed drivers, unvetted hosts — makes it a poor first-time choice; mentioned for completeness.)

---

## 3. Grant Budget with Buffer

### 3.1 Baseline

| Item | Calculation | Cost |
|---|---|---|
| GPU compute | 80 pod-hrs × $0.49/hr (RTX A6000 48GB, RunPod Secure) | $39 |
| Persistent storage | 100 GB × $0.07/GB/mo × ~1.5 months | $11 |
| API evaluation credits (Claude API) | ~5–8K outputs judged (refusal/bypass/degenerate + StrongREJECT + coherence), Haiku-class judge with Sonnet spot-checks | $40 |
| **Baseline subtotal** | | **$90** |

### 3.2 Buffer (30%)

A 30% safety factor is applied to cover: failed/OOM runs during hook debugging, steering-strength and layer-window re-sweeps after red-teaming, paraphrase-control re-runs, the Week-2 pivot contingency (truth-probe generalization, ~80% shared code), and re-scoring after evaluator prompt fixes.

| | |
|---|---|
| Baseline | $90 |
| Buffer (30%) | $27 |
| **Buffered compute & tooling subtotal** | **$117 → round to $120** |

### 3.3 Claude Max subscription (5 weeks)

Claude Max bills monthly at $100/month with no weekly proration. Five weeks (35 days) spans **two billing cycles**, so the exact cost to cover the project window is:

**2 × $100 = $200** *(the second cycle also covers post-project revisions to the public blog post at no extra cost)*

**Justification.** This project's stated novel contribution is only ~15% new code, but that 15% sits on top of three research codebases (Arditi, Zhong, Marks) that must be forked, understood, and stitched together by a researcher explicitly bridging from classical ML into deep learning and mech-interp. Claude Max funds:
- **Agentic coding across the three repos** (Claude Code): adapting Zhong's hook/knockout code to the consequence contrast, debugging TransformerLens fp32-caching OOMs, wiring the three-way scoring harness — the single biggest schedule risk for a first-time interpretability setup.
- **Dataset construction at quality:** generating and paraphrasing 100–200 minimal pairs across ≥10 framing templates with held-out phrasings — the component the tutorial calls "the whole ballgame" — where LLM-assisted template diversification directly defends against the probe-learns-surface-tokens failure mode (Kirch et al.).
- **Fixed, predictable cost:** equivalent daily usage through the metered API would be substantially more expensive and unpredictable; a flat $100/month is the budget-safe option for sustained daily use over 5 weeks.
- **Write-up:** figures, limitations framing, and the public blog post that is the course deliverable.

### 3.4 Total request

| Category | Amount |
|---|---|
| Compute + storage + API eval (incl. 30% buffer) | $120 |
| Claude Max, 2 billing cycles covering 5 weeks | $200 |
| Contingency to grant rounding | $30 |
| **Total grant request (nearest $50)** | **$350** |

### Paste-ready line for the "What specifically would this grant fund?" field

> "$120 GPU compute & evaluation — ~80 hrs RTX A6000 48GB on RunPod ($0.49/hr) + 100GB persistent storage + Anthropic API credits for automated refusal/bypass/degenerate scoring, incl. 30% buffer for failed runs and re-sweeps; $200 Claude Max (2 months) for agentic coding across the three forked interpretability repos, contrastive-dataset construction, and write-up; $30 contingency. Total: $350."

---

## Assumptions & pricing sources

- RunPod list prices as of July 2026: RTX A6000 48GB $0.49/hr (Secure), RTX 4090 24GB $0.34/hr (Community) / $0.69 (Secure), network storage $0.07/GB/month. Per-second billing; pod stopped between work sessions.
- No training, no fine-tuning, no checkpoints — frozen-model inference with forward hooks throughout.
- Anthropic API judge costs assume a Haiku-class model for bulk scoring; a 2× volume overrun is absorbed by the buffer.
- If A6000 availability is poor, fallback is RTX 4090 24GB with HF + custom hooks (bf16, smaller batches) at lower hourly cost — budget unchanged.
