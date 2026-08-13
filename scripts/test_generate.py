#!/usr/bin/env python
"""Regression test for batched generation. CPU, no model download.

    python scripts/test_generate.py

The bug this pins down: a decoder-only model continues from the LAST token of the sequence, so
with RIGHT padding every prompt shorter than the longest in its batch continues from `<pad>`
instead of from its own final token. Nothing raises — the completions just come back empty or
unrelated, and transformers' one-line warning about it is buried in a wall of loading bars.

The stub model below echoes the last input token, which is the property that matters: under
right padding a short prompt echoes the pad, under left padding it echoes its own last token.
So this test fails against the unfixed code instead of merely exercising it.
"""

import sys

import torch

import _bootstrap  # noqa: F401
from consequence.generate import generate


class Enc(dict):
    def to(self, device):
        return self


class StubTok:
    """One token per character, so expectations are readable."""

    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self.padding_side = "right"          # the transformers default that caused the bug

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[-1]["content"]

    def __call__(self, batch, return_tensors=None, padding=True):
        ids = [[ord(c) for c in text] for text in batch]
        width = max(len(row) for row in ids)
        padded, mask = [], []
        for row in ids:
            pad = [self.pad_token_id] * (width - len(row))
            if self.padding_side == "left":
                padded.append(pad + row); mask.append([0] * len(pad) + [1] * len(row))
            else:
                padded.append(row + pad); mask.append([1] * len(row) + [0] * len(pad))
        return Enc(input_ids=torch.tensor(padded), attention_mask=torch.tensor(mask))

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["".join(chr(t) for t in row if t != self.pad_token_id) for row in ids.tolist()]


class StubModel:
    """Echoes the last input token — the one behaviour that makes padding side observable."""

    device = "cpu"

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=4, **kw):
        return torch.cat([input_ids, input_ids[:, -1:].repeat(1, max_new_tokens)], dim=1)


def check(name: str, ok: bool) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok


def main() -> int:
    tok, model = StubTok(), StubModel()
    prompts = ["abc", "hello", "xy"]          # deliberately different lengths in one batch
    ok = True

    out = generate(model, tok, prompts, max_new_tokens=3, batch_size=16)
    print(f"  completions: {out}")

    # Each prompt must continue from ITS OWN last character, not from the batch's padding.
    ok &= check("every prompt continues from its own last token",
                out == ["ccc", "ooo", "yyy"])
    ok &= check("no completion is empty (right padding yields '' for short prompts)",
                all(c for c in out))
    ok &= check("tokenizer padding_side restored after the call",
                tok.padding_side == "right")

    # Same prompts, one per batch: padding cannot apply, so this is the ground truth that
    # batching must reproduce exactly.
    unbatched = generate(model, tok, prompts, max_new_tokens=3, batch_size=1)
    ok &= check("batched output identical to unbatched", out == unbatched)

    tok.padding_side = "left"                 # must work regardless of the incoming setting
    ok &= check("correct when the tokenizer already pads left",
                generate(model, tok, prompts, max_new_tokens=3, batch_size=16) == out)

    print("\n" + ("ALL PASS — batched generation is padding-safe"
                  if ok else "FAILURES ABOVE — completions are not what the model produced"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
