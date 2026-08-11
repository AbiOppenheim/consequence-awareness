# data/eval/ — evaluation prompt sets for the causal sweep

These are the prompts stage 05 generates on and stage 06 judges. They are **downloaded from
published suites**, never authored here (CLAUDE.md §7: no novel jailbreaks).

| file | source | role |
|---|---|---|
| `fiction_jailbreaks.jsonl` | JailbreakBench / StrongREJECT fiction & role-play slices | the attacks we try to re-refuse by steering toward "real" |
| `xstest.jsonl` | XSTest (`walledai/xstest` or Röttger et al.) | over-refusal control: 250 *safe* prompts that look risky |

XSTest is the guard on the causal claim: steering toward "real" must restore refusal on real
jailbreaks **without** making the model refuse everything. If XSTest refusal shoots up too, the
"defense" is just global over-refusal, not consequence awareness.

## Expected row shape

```json
{"id": "...", "text": "<prompt>", "source": "jailbreakbench", "attack_family": "fiction"}
```

Stage 05 reads `text` (falls back to `prompt`). Keep one attack family per file, or tag
`attack_family` so results can be scoped per family (Kirch et al.: jailbreak features are
attack-family-specific — do not average across families and call it "the" effect).

## Populating (Week 2–3)

Download into `data/raw/` first (gitignored), then filter/normalize into these jsonl files.
Store only prompt text and metadata here — never model completions (those live, gitignored,
under `artifacts/generations/`). Redact any completions before they reach the write-up.
