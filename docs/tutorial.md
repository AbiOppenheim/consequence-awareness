# A Step-by-Step Tutorial to Understand This Project

*Companion to the project plan "Is 'This Isn't Real' a Direction?" — read this to build the mental model and the toolkit before you touch code.*

This tutorial is organized as four steps, matching what you need to hold in your head:

1. **The question and hypothesis** — what you're actually asking.
2. **The LLM + interpretability concepts** — the machinery, taught from your ML background.
3. **The models and code to use** — what to fork, what to write.
4. **The data to use** — every dataset, and how to build the pieces that don't exist yet.

A curated resource list (all vetted, current links) is at the end, keyed to each step.

---

## Step 1 — The question and hypothesis

### The everyday observation
Fiction, role-play, and "hypothetical" framings reliably weaken a model's refusals. Ask directly for something harmful and it refuses; wrap the same request in "write a story where a character explains…" and it often complies. **Why?**

### Two families of existing explanations
- **Refusal-suppression accounts.** Refusal is mediated by a single linear direction in the residual stream (Arditi et al., 2024); jailbreaks push activations off that direction. A refinement (Zhao et al., 2025, *LLMs Encode Harmfulness and Refusal Separately*) shows jailbreaks can suppress the *refusal* signal while the model's internal *harmfulness* judgment stays intact.
- **Representation-shift accounts.** Jailbreaks move activations toward "safe-looking" regions of representation space (JailbreakLens, 2024).

### The hypothesis this project tests
A *third* possibility: some framings work by shifting the model's internal sense of whether the situation is **real** — whether the output has consequences. "It's just fiction / a coding exercise that will never run" might move the model along a **consequence-awareness direction**, and *that* is what gates refusal.

### Why we reframed it around geometry (important)
The naive version — "do jailbreaks work this way, yes/no?" — is a trap, because the honest answer is probably "partly, for some attacks, tangled up with persona." A binary question makes a messy result look like failure. So the real question is **geometric and comparative**:

> Is there a linear "hypothetical / no-real-consequences" direction? If so, how does it relate to the two directions already in the literature — the **refusal direction** (Arditi) and the **compliant-persona direction** (Zhong & Li, 2026) — and does steering along it causally change refusal under fiction jailbreaks?

**Every outcome is a result:**
- The direction exists and is *distinct* from refusal/persona, and steering it restores refusal → consequence-awareness is a new, causal defense target.
- The direction is really just the persona direction relabeled → useful negative result; stop treating "consequence" as separate.
- The direction exists but steering does nothing → it's a correlate, not a cause; defense work stays on the refusal pathway.

### The two claims you'll actually try to support or refute
1. **Existence + distinctness:** a linear consequence direction `v_C` exists, generalizes to held-out framings, and is geometrically distinct from `r̂` (refusal) and `v_MP` (persona).
2. **Causality:** steering toward "real" along `v_C` restores refusal during fiction-framed jailbreaks, beyond what random / persona / shallow-alignment baselines explain.

---

## Step 2 — The LLM and interpretability concepts for this project

You have classical ML/NLP (embeddings, Random Forests). The gap is transformer internals and the idea that abstract concepts live as **linear directions** you can read and write. Here's the minimal set, in dependency order.

### 2.1 The residual stream (the single most important object)
A transformer processes a sequence as a stack of layers. At each token position there's a vector — the **residual stream** — that starts as the token embedding and gets *added to* by every attention head and MLP as it flows up the layers. Think of it as a shared "workspace" or communication channel: components read from it, compute, and write results back by addition. So a layer-ℓ activation at the last token is an accumulator of everything the model has figured out so far. Every direction you'll extract lives in this space (dimension ≈ 3584 for Qwen2.5-7B, 4096 for Llama-3.1-8B).

*Why you care:* "a direction in the residual stream" is not a metaphor — it's a literal unit vector `v` in that ~4096-dim space, and you can measure how much any activation `h` points along it (`h · v`) or push it (`h ← h + α·v`).

### 2.2 The linear representation hypothesis
The working assumption of this whole subfield: many human-interpretable features (sentiment, language, truthfulness, refusal…) are represented **linearly** — as directions — so that moving along the direction changes that feature. This is exactly the intuition behind classic `king − man + woman ≈ queen` word-embedding arithmetic, lifted into the model's internal activations. Marks & Tegmark's *Geometry of Truth* is the cleanest demonstration: LLMs linearly encode whether a statement is true or false, and you can *flip* the model's treatment of a statement by intervening on that direction. **Read that paper as your tutorial — it does, for "truth," exactly what you'll do for "consequence."** (It also sits right on your stated interest in truth/falsehood and model semantics.)

### 2.3 Difference-in-means: how you extract a direction
The core recipe, used identically by Arditi, Rimsky (CAA), Marks & Tegmark, and Zhong:

1. Build **contrastive pairs**: prompts that differ *only* in the property you want (e.g. harmful vs. harmless; compliant-persona vs. not; real-framing vs. hypothetical-framing).
2. Run each prompt, cache the residual-stream activation at a chosen layer and token position (usually the last token).
3. Average the activations of each class, subtract:  `v = mean(positive) − mean(negative)`, then normalize to unit length.

That's it. `v` is your candidate direction. Simple difference-in-means probes turn out to identify the directions most *causally* implicated in behavior — often better than fancier probes (a key Marks & Tegmark finding, echoed by Arditi). Your three directions — `r̂`, `v_MP`, `v_C` — are all built this way, differing only in the contrast.

### 2.4 Linear probes (and why a probe is not proof)
A **linear probe** is a linear classifier trained on activations to predict a label (is this prompt "hypothetical"?). High probe accuracy shows the information is *present and linearly readable*. But — critical — a probe reading a direction does **not** prove the model *uses* it to decide anything. A feature can be decodable yet causally inert. This gap is the entire reason Step-1 claim #2 (causality) needs a separate *intervention* test, not just a probe.

### 2.5 Steering by activation addition
To test causality, you **write** to the residual stream. Add the direction during the forward pass at the relevant layers: `h ← h + α·v`. Positive `α` amplifies the feature, negative suppresses it. This is **Contrastive Activation Addition** (Rimsky et al., 2024): build the vector from contrast pairs, add it at all post-prompt positions with a tunable coefficient. Your causal test = steer toward "real" along `v_C` during a fiction jailbreak and check whether refusal comes back.

### 2.6 Directional ablation / projection knockout
The complementary intervention: **remove** a direction by projecting it out of the activation everywhere: `h ← h − (h · v̂)·v̂`. Arditi uses this to *erase* refusal (the model stops refusing). Zhong uses **projection knockout** of the persona direction in a late-layer window (≈ layers 20–22) to *restore* refusal that persona-steering had suppressed — and shows a *random* direction does nothing (the essential control). You'll use projection knockout as your **distinctness** test: if projecting out `v_C` behaves interchangeably with projecting out `v_MP`, they're plausibly the same mechanism; if not, they're distinct.

### 2.7 Comparing directions: cosine similarity
Two unit directions are compared by cosine similarity (`v_C · v_MP`). Near 0 → roughly orthogonal (distinct); near ±1 → nearly the same axis (suspicious — maybe relabeled). You'll report `cos(v_C, r̂)`, `cos(v_C, v_MP)`, `cos(r̂, v_MP)` across layers, against a random-direction null. This is the literal "geometry" that is the spine of the project.

### 2.8 Measuring the outcome: refusal / bypass / degenerate
Do **not** score with a single attack-success number. Steering breaks fluency at high strengths, and a naive refusal-rate would misread gibberish as a "successful" defense. Use Zhong's three-way label on every output — **refusal** (model declines), **bypass** (model genuinely complies = leak), **degenerate** (incoherent/broken) — plus a coherence check on benign prompts, and a leakage score. Report a **StrongREJECT** score too, but never *alone*: Zhong shows a StrongREJECT of 0.07 can hide 42% real bypass + 56% degenerate output.

### 2.9 The load-bearing caution: probes don't transfer across attacks
Kirch et al. (2025) trained probes on 10,800 jailbreak attempts across 35 methods and found transfer is **attack-family-specific** and the jailbreak-relevant features are **non-linearly** encoded. Translation for you: don't claim a universal "consequence direction" that works on all jailbreaks. Scope every claim to the real-vs-hypothetical contrast and the specific narrative attacks you test. If your *linear* probe underperforms out-of-distribution, that's consistent with Kirch — report it and flag non-linear probes as future work.

---

## Step 3 — The implemented models and code to use

### 3.1 Models (use exactly these)
- **Qwen2.5-7B-Instruct** and **Llama-3.1-8B-Instruct.** These are Zhong & Li's exact models, so their pipeline transfers with zero porting. Start with **Qwen only**, end-to-end; add Llama in Week 4 for a robustness check. Both are gated on Hugging Face — request access early (it's the kind of thing that silently blocks you on day 1).
- Compute: a single mid-range GPU (e.g. one A100/RTX on RunPod) is plenty; everything here is inference + hooks, no training.

### 3.2 The three repos, and what each gives you
Fork rather than build — your original code is small and surgical.

**A) `andyrdt/refusal_direction`** (Arditi et al.) — the foundation.
- Has a `run_pipeline` entry point that extracts the refusal direction and evaluates refusal metrics on a model path (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`).
- Gives you: difference-in-means extraction, directional ablation, activation addition, and a refusal-scoring harness. Setup prompts for a Hugging Face token (gated models) and a Together AI token (for jailbreak safety scoring).
- **Reuse for:** your refusal direction `r̂`, and the ablation/steering primitives.

**B) `violazhong/refusal-downstream-persona`** (Zhong & Li) — your closest template.
- Built *on top of* Arditi's method, on your exact two models. Contains persona-vector (`v_MP`) extraction, forward-hook steering, **projection knockout**, and the refusal/bypass/degenerate scoring.
- **Reuse for:** the persona direction `v_MP` (a core baseline), the steering + knockout code, and the three-way metric. Their appendix documents the exact layers/positions.

**C) `saprmarks/geometry-of-truth`** (Marks & Tegmark) — your methodology tutorial.
- Clean, readable difference-in-means + probing + causal-intervention code for the "truth" direction, plus an interactive dataexplorer for building intuition.
- **Reuse for:** understanding the pattern end-to-end, and their **mass-mean probing** trick if you want a probe that better tracks the causal direction. Not your production pipeline — your learning scaffold.

### 3.3 The tooling underneath
- **PyTorch forward hooks** are the fundamental mechanism — a callback that reads or overwrites a layer's output mid-forward-pass. Zhong's code uses plain Hugging Face models + forward hooks; learn this first, it demystifies everything.
- **TransformerLens** (`HookedTransformer`) wraps the same idea with named hook points and is what Arditi's pipeline and most tutorials use. Skim the ARENA / TransformerLens intro so you can read that code; don't study both frameworks deeply up front — learn what each repo you touch actually uses.

### 3.4 What *you* write (the novel ~15%)
1. The **consequence contrastive dataset** (Step 4).
2. A one-function call to extract `v_C` via the same difference-in-means code you're already reusing.
3. A **linear probe** on `v_C` with a proper held-out split.
4. The **geometry + differential-ablation comparison** across `v_C`, `r̂`, `v_MP`, and a random null.

### 3.5 Worked mini-example (pseudocode, faithful to the method)
```python
# 1) EXTRACT a direction by difference-in-means
def extract_direction(model, pos_prompts, neg_prompts, layer, pos_idx=-1):
    H_pos = [cache_resid(model, p, layer)[pos_idx] for p in pos_prompts]
    H_neg = [cache_resid(model, p, layer)[pos_idx] for p in neg_prompts]
    v = mean(H_pos) - mean(H_neg)
    return v / norm(v)                       # unit direction

v_C  = extract_direction(model, real_framings, hypo_framings, layer=L)

# 2) STEER (activation addition) toward "real" during a jailbreak
def steer_hook(resid, v=v_C, alpha=+A):
    return resid + alpha * v                  # add at post-prompt positions

# 3) ABLATE (projection knockout) to test distinctness
def knockout_hook(resid, v):
    return resid - (resid @ v).unsqueeze(-1) * v

# 4) GEOMETRY
cos_C_r   = float(v_C @ r_hat)
cos_C_mp  = float(v_C @ v_MP)
cos_C_rand= float(v_C @ random_unit_vector())   # null baseline
```

---

## Step 4 — The data to use

Four data needs. Two are off-the-shelf; two you build.

### 4.1 Harmful vs. harmless prompts (for `r̂` and for eval) — off the shelf
- **AdvBench** (Zou et al., 2023): 520 harmful instructions. Standard, but **known flaws** — many near-duplicate prompts, and ~46% are cyber-security, which *skews* a difference-in-means refusal direction. Don't use it raw for direction extraction without balancing.
- **Alpaca** (Taori et al., 2023): the usual harmless counterpart set.
- **StrongREJECT** (Souly et al., 2024): 313 carefully curated harmful prompts + a validated automated evaluator; available as `walledai/StrongREJECT` on Hugging Face. Use this for scoring — it's designed to *not* overstate jailbreak success.
- **JailbreakBench** (Chao et al., 2024): 100 harmful + 100 harmless behaviors on matched topics — good for checking the model isn't refusing on keywords alone.
- **HarmBench** (Mazeika et al., 2024): 200 de-duplicated harmful prompts; cleaner than AdvBench.

*Recommended:* extract `r̂` on a **balanced** harmful/harmless set (de-duplicate, cap the cyber category), evaluate on StrongREJECT + a JailbreakBench slice.

### 4.2 Persona contrast (for `v_MP`) — reuse Zhong's
Zhong & Li release their trait configurations and vector-extraction scripts (compliant vs. evasive persona pairs). Reuse them directly so your baseline `v_MP` matches the published one — don't reinvent the persona contrast, or the head-to-head comparison won't be clean.

### 4.3 The consequence contrast (for `v_C`) — **you build this; it's the whole ballgame**
Minimal pairs sharing the same underlying task, differing *only* in real-vs-hypothetical framing. Example skeleton:
- **Real:** *"You are an AI assistant responding to a user's request. [TASK]"*
- **Hypothetical:** *"You are an AI assistant responding to a coding exercise that will never be executed. [TASK]"*

Design rules that determine whether the project succeeds:
- **Diversify the framing templates.** If every "hypothetical" prompt contains the word *exercise*, your probe learns the token "exercise," not consequence. Use many surface forms (fiction, simulation, dream, test environment, thought experiment, "sandbox that resets"…).
- **Hold out phrasings, not just examples.** Train `v_C` / the probe on one set of templates; test generalization on *unseen* templates and paraphrases. This is your defense against the Kirch failure mode.
- **Keep the task content matched across the pair** so the only varying signal is consequence-framing.
- Size: ~100–200 pairs is enough for a difference-in-means direction (CAA works with hundreds; Arditi's refusal direction uses a small contrastive set).
- For inspiration on minimal harmful/harmless pairs, see the mech-interp adversarial-attack dataset (2503.06269), which builds 60 structure-matched pairs by minimally editing intent.

### 4.4 Fiction-framed jailbreaks (for the correlational read + causal test) — use existing, published sets
- Draw role-play / fiction / hypothetical jailbreak prompts from an **existing published** attack suite (e.g. the narrative/wrapper attacks catalogued in jailbreak benchmarks and the "Adversarial Tales" line of work). **Do not craft novel working attacks.** You're testing whether the `v_C` probe reads "hypothetical" on these, and whether steering toward "real" restores refusal.

### 4.5 Over-refusal control — off the shelf
- **XSTest** (Röttger et al., 2024): 250 *safe* prompts that look superficially risky (contain words like "kill"). When you steer toward "real" you must check you're not just making the model refuse *everything*. XSTest catches that.

### 4.6 Responsible-use note (keeps the project "safe")
This is defensive interpretability. Use only published harmful-prompt benchmarks and published attacks; never develop or release new jailbreaks. In the blog post, report **mechanism and defense implications** and aggregate numbers — redact or omit any harmful completions rather than showing them. State the dual-use consideration explicitly; the course rubric wants the safety connection argued, not assumed.

---

## Putting it together (the arc)
1. Reproduce Arditi's refusal direction + Zhong's persona suppression on Qwen (validates your pipeline — Week 1).
2. Build the consequence contrast, extract `v_C`, probe on held-out framings (existence — Week 2).
3. Cosines + differential ablation vs. `r̂`, `v_MP`, random (distinctness — Week 2–3).
4. Correlational read on published fiction jailbreaks, then steer toward "real" with the 3-way metric and all baselines (causality — Week 3).
5. Red-team: paraphrase controls, the shallow-alignment (first-token) check, second model, XSTest (Week 4). Write up (Week 5).

---

## Curated resources (all vetted)

**Step 1 — question / prior work**
- Arditi et al., *Refusal Is Mediated by a Single Direction* — https://arxiv.org/abs/2406.11717
- Zhong & Li, *Refusal Lives Downstream of Persona* — https://arxiv.org/abs/2606.26161
- Marks & Tegmark, *The Geometry of Truth* — https://arxiv.org/abs/2310.06824 · interactive: https://saprmarks.github.io/geometry-of-truth/dataexplorer/

**Step 2 — concepts**
- Neel Nanda, *Comprehensive Mechanistic Interpretability Explainer & Glossary* — https://www.neelnanda.io/mechanistic-interpretability/glossary
- Neel Nanda, *Mechanistic Interpretability Quickstart* — https://www.neelnanda.io/mechanistic-interpretability/quickstart
- Elhage et al., *A Mathematical Framework for Transformer Circuits* (residual-stream section) — https://transformer-circuits.pub/2021/framework/index.html
- Rimsky et al., *Contrastive Activation Addition* — https://aclanthology.org/2024.acl-long.828/ · plain-English writeup: https://www.lesswrong.com/posts/v7f8ayBxLhmMFRzpa/steering-llama-2-with-contrastive-activation-additions
- Implementation walkthrough (ActAdd + CAA, with code) — https://mandliya.github.io/blog/2025/paper-implementation-steering-language-models-with-activation-engineering/

**Step 3 — code / tooling**
- Arditi refusal-direction repo — https://github.com/andyrdt/refusal_direction
- Zhong persona-refusal repo — https://github.com/violazhong/refusal-downstream-persona
- Marks geometry-of-truth repo — https://github.com/saprmarks/geometry-of-truth
- ARENA Chapter 1 (Transformer Interpretability; hooks, probing) — https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/
- TransformerLens getting-started — https://transformerlensorg.github.io/TransformerLens/content/getting_started_mech_interp.html

**Step 4 — data**
- StrongREJECT (HF) — https://huggingface.co/datasets/walledai/StrongREJECT
- AdvBench + flaws discussion — https://arxiv.org/abs/2402.10260 (StrongREJECT paper) · minimal-pair construction: https://arxiv.org/abs/2503.06269
- (Look up on HF as needed:) JailbreakBench, HarmBench, XSTest, Alpaca.

**Scoping caution (read once, keep in mind)**
- Kirch et al., *What Features in Prompts Jailbreak LLMs* — in your project folder.
