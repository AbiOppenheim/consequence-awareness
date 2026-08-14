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
configs/qwen.yaml     fixed config (model, layers, alphas, judge, seed) — feeds every sidecar
data/contrast/        contrast sets: consequence.jsonl (authored) + refusal/persona (from repos)
data/eval/            downloaded attack + over-refusal prompt sets (no completions stored)
src/consequence/      acts · directions · hooks · generate · probe · judge  (+ config, io, data)
scripts/              thin CLI, one per stage: 01_cache_acts … 07_figures
artifacts/            directions/ activations/ generations/ scores/ figures/  (gitignored except sidecars)
external/             reference repos — cloned, run in own venvs, NEVER imported
docs/                 project plan, tutorial/design doc, compute estimation
logs/research_log.md  course requirement — updated every session
```

**The artifact contract.** Every direction (`r̂`, `v_MP`, `v_C`, random) is the same object — a
unit vector in ℝ³⁵⁸⁴ — so the interface between our code and the three reference repos is a
**file, not an import**. Each `.pt` ships a sidecar `.json` (model, layer, token pos, contrast,
n_pairs, seed, git SHA). `src/` never imports `external/`. Only stages 01 (cache) and 05
(generate) need a GPU; everything else runs on a laptop against cached artifacts.

Forked foundations: [`andyrdt/refusal_direction`](https://github.com/andyrdt/refusal_direction) · [`violazhong/refusal-downstream-persona`](https://github.com/violazhong/refusal-downstream-persona) · [`saprmarks/geometry-of-truth`](https://github.com/saprmarks/geometry-of-truth)

## Setup & pipeline

```bash
# local env for the CPU stages (02, 03, 04, 07). On the GPU pod, also install transformers+accelerate.
python -m venv .venv && source .venv/bin/activate
pip install -e .                       # or: pip install numpy scikit-learn pyyaml matplotlib torch

python scripts/00_build_consequence.py         # cross-product benign_tasks.jsonl × framings.jsonl
```

Each research step is one script. Steps that cost real time store their numbers in
`artifacts/results/<step>.json`, keyed on a hash of their inputs, parameters and code
(`consequence.results`) — **re-running a step with nothing changed is a no-op that reprints the
saved numbers**, and a step whose dataset, selected layer or analysis code moved underneath it
recomputes and says which. `notebooks/phase2_3_gate_and_vc_colab.ipynb` runs these steps and
renders their stored results; it computes nothing itself.

| stage | script | GPU? | reads → writes |
|---|---|---|---|
| 01 | `01_cache_acts.py --dataset D` | ✅ | contrast jsonl → `activations/D_*.npz` |
| 02 | `02_extract_directions.py --dataset D [--kind v_mp]` | — | activations → `directions/{v_c,v_mp_D}_L*.pt` (+ random null) |
| 02b | `02b_gate.py` | — | our + Arditi's refusal dirs → `results/gate.json`, `directions/r_hat_L*.pt` |
| 03 | `03_probe.py` | — | activations → `results/layer_select.json` (**train templates only**) |
| 04 | `04_geometry.py` | — | directions → `results/geometry.json` (cosines vs `r̂`/`v_MP` + null band) |
| 09 | `09_heldout.py` | — | held-out reveal → `results/heldout.json` (**run once**) |
| 10 | `10_redteam_heldout.py` | — | per-route, per-layer, TF-IDF, 100-draw null → `results/heldout_redteam.json` |
| 11 | `11_calibrate_alpha.py [--layer L]` | — | residual norms → `results/alpha_ladder_L*.json` |
| 05 | `05_generate.py --eval E` | ✅ | eval prompts under steer/knockout → `generations/*.jsonl` |
| 06 | `06_judge.py [--resume]` | — | generations → refusal/bypass/degenerate + StrongREJECT |
| 08 | `08_sweep_analysis.py` | — | scores → `results/sweep_L*.json` + the four readings |
| 13 | `13_orthogonalize.py` | — | `v_C` minus its `r̂` component → `directions/v_c_orth_r_hat_L*.pt`, `results/orthogonalize.json` |
| 07 | `07_figures.py` | — | analysis JSONs → figures |

Steps 09, 10, 11, 13 and 05 read the selected layer from `results/layer_select.json`, and 05
reads the α ladder from `results/alpha_ladder_L{layer}.json` (`--layer auto --alphas auto`, the
defaults), so the layer and the ladder are never retyped into a command or carried in a notebook
variable. The ladder is per-layer because residual norms grow with depth: the same raw α is a
different dose at L22 than at L18, and layers are only comparable as fractions of ‖h‖.

`05_generate.py --extra-direction STEM` adds any minted direction as an extra swept condition,
appending to the same output file so resumption keeps what already ran. That is how the
orthogonalized `v_C` is tested causally.

**Offline tests** (no GPU, no API, no cost): `test_hooks.py` (steering hits the layer `v_C` was
measured at, under both HF block conventions), `test_generate.py` (batched generation is
padding-safe), `test_judge.py` (judge request shape, per-row failures, and that a failed
judgment is never counted as a refusal). Each was verified to fail against the bug it pins.

`r_hat` is minted per layer by step 02b from the cube that Arditi's repo writes into
`artifacts/directions/` — see `external/README.md`. `v_MP` is our reimplementation of Zhong's
`compliant_v2 model_persona`, extracted through the same stage 02 as `v_C`.

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
