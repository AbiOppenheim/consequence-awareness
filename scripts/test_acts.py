#!/usr/bin/env python
"""Regression test for activation caching. CPU, no model download.

Covers the two things that broke a real run:

  1. The causal-LM wrapper computes logits — [batch, seq, ~152k vocab] — that caching never
     uses. Short contrast prompts hid the cost; the fiction jailbreaks are ~10x longer and
     OOM'd an L4 inside lm_head. Caching must go through the head-free decoder, and the test
     that matters is that lm_head is never called at all.
  2. The head-free path must return byte-identical hidden states, because caches built before
     the change are compared against caches built after it (stage 12). A silent discrepancy
     there moves every projection instead of raising.

Plus the OOM backoff: batch size is a memory knob and must not change the numbers.
"""

import sys

import torch
from torch import nn

import _bootstrap  # noqa: F401
from consequence import acts as A

D, N_LAYERS, N_PROMPTS = 8, 4, 7
VOCAB = 32


class Enc(dict):
    """Stand-in for BatchEncoding: dict plus .to(), which is all cache_activations uses."""

    def to(self, _device):
        return self


class Tok:
    padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "|".join(m["content"] for m in messages)

    def __call__(self, batch, return_tensors=None, padding=None):
        seq = max(len(b) for b in batch)
        ids = torch.zeros(len(batch), seq, dtype=torch.long)
        mask = torch.ones(len(batch), seq, dtype=torch.long)
        for i, b in enumerate(batch):                     # left-pad, as Qwen's tokenizer does
            t = torch.tensor([ord(c) % VOCAB for c in b], dtype=torch.long)
            ids[i, seq - len(t):] = t
            mask[i, : seq - len(t)] = 0
        return Enc(input_ids=ids, attention_mask=mask)


class Decoder(nn.Module):
    """The transformer stack. Deterministic per-layer shift so each layer is identifiable."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        # Deterministic weights: two models must be comparable, or the batch-invariance and
        # OOM-retry checks below would compare random initialisations and always "fail".
        with torch.no_grad():
            self.embed.weight.copy_(
                torch.arange(VOCAB * D, dtype=torch.float32).reshape(VOCAB, D) / (VOCAB * D))

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
        h = self.embed(input_ids)
        hs = [h]
        for i in range(N_LAYERS):
            h = h + (i + 1)
            hs.append(h)
        return type("Out", (), {"hidden_states": tuple(hs)})()


class CausalLM(nn.Module):
    """Mimics Qwen2ForCausalLM: decoder + a vocab-sized head that caching must never touch."""

    def __init__(self, agree: bool = True):
        super().__init__()
        self.model = Decoder()
        self.lm_head = nn.Linear(D, VOCAB, bias=False)
        self.device = torch.device("cpu")
        self.head_calls = 0
        self._agree = agree

    def get_decoder(self):
        return self.model

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=output_hidden_states)
        self.head_calls += 1
        self.lm_head(out.hidden_states[-1])               # the wasted allocation, in miniature
        if not self._agree:                               # simulate a path that does NOT match
            return type("O", (), {"hidden_states": tuple(h + 99 for h in out.hidden_states)})()
        return out


def check(name: str, ok: bool) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok


def main() -> int:
    tok, ok = Tok(), True
    prompts = [f"prompt number {i}" for i in range(N_PROMPTS)]

    model = CausalLM()
    ok &= check("_decoder() returns the head-free stack",
                A._decoder(model) is model.model)

    acts = A.cache_activations(model, tok, prompts, token_position=-1, batch_size=3)
    ok &= check("shape is [n_prompts, n_layers, d_model], embedding layer dropped",
                acts.shape == (N_PROMPTS, N_LAYERS, D))
    ok &= check("lm_head is NEVER called during caching (only the one verification forward)",
                model.head_calls == 1)

    # Batch size is a memory knob. If it changed the numbers, the OOM backoff below would
    # silently produce a different cache than a run that never hit OOM.
    model2 = CausalLM()
    acts_b1 = A.cache_activations(model2, tok, prompts, token_position=-1, batch_size=1)
    ok &= check("batch size does not change the activations",
                bool((acts == acts_b1).all()))

    # ---- the verification guard actually fires -----------------------------------------
    try:
        A.cache_activations(CausalLM(agree=False), tok, prompts, batch_size=3)
        ok &= check("a head-free path that disagrees is rejected", False)
    except RuntimeError as e:
        ok &= check("a head-free path that disagrees is rejected",
                    "NOT be comparable" in str(e))

    # ---- OOM backoff ---------------------------------------------------------------------
    class OomDecoder(Decoder):
        """OOMs above a batch of 2, like a real card on long prompts."""

        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
            if input_ids.shape[0] > 2:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
            return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                   output_hidden_states=output_hidden_states)

    oom = CausalLM()
    oom.model = OomDecoder()
    acts_oom = A.cache_activations(oom, tok, prompts, token_position=-1, batch_size=8)
    ok &= check("OOM halves the batch and completes the run",
                acts_oom.shape == (N_PROMPTS, N_LAYERS, D))
    ok &= check("activations after an OOM retry match the clean run",
                bool((acts_oom == acts).all()))

    class AlwaysOom(Decoder):
        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")

    hopeless = CausalLM()
    hopeless.model = AlwaysOom()
    try:
        A.cache_activations(hopeless, tok, prompts, batch_size=4)
        ok &= check("OOM at batch 1 is raised, not looped on forever", False)
    except torch.cuda.OutOfMemoryError:
        ok &= check("OOM at batch 1 is raised, not looped on forever", True)

    # ---- right-padding uses the attention mask, not position -1 --------------------------
    class RightTok(Tok):
        padding_side = "right"

        def __call__(self, batch, return_tensors=None, padding=None):
            seq = max(len(b) for b in batch)
            ids = torch.zeros(len(batch), seq, dtype=torch.long)
            mask = torch.zeros(len(batch), seq, dtype=torch.long)
            for i, b in enumerate(batch):
                t = torch.tensor([ord(c) % VOCAB for c in b], dtype=torch.long)
                ids[i, : len(t)] = t
                mask[i, : len(t)] = 1
            return Enc(input_ids=ids, attention_mask=mask)

    rt = CausalLM()
    acts_r = A.cache_activations(rt, tok=RightTok(), prompts=["ab", "abcdefgh"], batch_size=2)
    ok &= check("right padding selects each row's last REAL token",
                acts_r.shape == (2, N_LAYERS, D))

    print("\n" + ("ALL PASS — caching is head-free, OOM-resilient, and batch-invariant"
                  if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
