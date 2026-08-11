---
name: red-team-result
description: Stress-test a result before believing it, writing it up, or reporting it to a facilitator. Use this whenever a number comes back, a plot looks good, an experiment "works", the user says something is surprising or exciting, or before anything goes into the blog post, research log, or a 1-on-1. Also use it when a result looks bad, since the same checks distinguish a real negative result from a bug. Always run this before claiming causality.
---

# Red-teaming a result

*"The mark of a good researcher is a deep commitment to skepticism of your results."* — Neel Nanda

Run this whenever a number arrives. It takes ten minutes and it is the difference between a
project that meets the bar and one that stands out.

## The four questions

1. **Is there a simpler, more boring explanation?**
2. **Could a bug produce exactly this?**
3. **What would a skeptical outsider say first?**
4. **What is this being compared against, and is that baseline any good?**

Question 4 is the one that gets skipped. A result only means something relative to something
else, and there is a natural bias toward polishing the new method while leaving the baseline
weak.

## Boring explanations, ranked by likelihood

Check in this order — the top ones are both most common and cheapest to test.

| # | Boring explanation | Test |
|---|---|---|
| 1 | The probe learned surface vocabulary | bag-of-words baseline on prompt text, held-out templates |
| 2 | The effect is vector norm, not direction | random-direction null at matched α |
| 3 | Chat template or token position mismatch | decode the exact tokenized input and eyeball it |
| 4 | The "defense" is incoherent output | three-way metric; look at 20 raw generations |
| 5 | The model now refuses everything | XSTest over-refusal control |
| 6 | Train/test leakage across templates | confirm held-out templates were never touched |
| 7 | Sampling noise | re-run with 3 seeds; is the effect bigger than the spread? |
| 8 | Class imbalance or length confound | check both class distributions |
| 9 | `v_C` is `r̂` wearing a hat | cosine against null band; differential ablation |

## Always look at raw outputs

Before believing any aggregate, read **20 actual generations** spanning conditions — some
scored refusal, some bypass, some degenerate. Every real bug in this pipeline is visible
within 20 samples and invisible in a bar chart. If the user has not looked at raw outputs,
say so before discussing the aggregate.

## Specific traps for this project

- **A cosine near zero is not "distinct" without the null band.** In 3584-d, random pairs sit
  at 0 ± 0.017. Report the 95th-percentile band.
- **A probe is not causal evidence.** If the write-up says "the model uses the consequence
  direction" and the only evidence is probe accuracy, that claim is unsupported. High probe
  accuracy shows the information is *present and linearly readable* — nothing more.
- **Attack-family scope.** Kirch et al. found jailbreak features are family-specific and
  partly non-linear. Any sentence of the form "jailbreaks work by…" needs narrowing to the
  attacks actually tested.
- **Cherry-picked layer.** If the effect exists at exactly one layer and vanishes at its
  neighbours, that is a red flag, not a precise localization. Report the full layer curve.
- **Post-hoc alpha.** Choosing the steering strength after seeing which one worked is
  selection. Pre-register the ladder, report all of it.

## Grading your own claim before it ships

For each claim in the write-up:

- What single experiment would most cheaply falsify it? Has it been run?
- Is the strength of the language matched to the evidence? ("suggests" vs "shows" vs "proves")
- Is the baseline stated in the same sentence as the number?
- Would a reader who disagrees find the counter-evidence in the post, or have to ask?

## When the result is negative

A negative result is a full result on this course's rubric — but only if the checks above
rule out "the pipeline was broken." Confirm the Week-1 validation gate still passes
(`cos(our r̂, Arditi's r̂) > 0.9`) before writing "we found no effect." A pipeline that
reproduces a known positive result and then finds nothing is evidence; one that has never
reproduced anything is silence.

Then write it up as: what we expected, what we found, what would change the conclusion,
and what someone should try next.

## Never do this

Do not adjust the split, the alpha, the layer, or the dataset in response to seeing a
disappointing number, and then report the improved number as if it were the first one. If a
change is made after seeing results, the honest move is to report both and label the second
as post-hoc. Flag it out loud if it starts happening.
