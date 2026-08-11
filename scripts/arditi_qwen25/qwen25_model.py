"""A Qwen2.5 wrapper for Arditi's extraction code — the adapter (CLAUDE.md Rule 1).

Arditi's pipeline is generic *except* for one class per model family: a `ModelBase` subclass
that tells the generic code four model-specific things —
    1. how to load the model,
    2. how to load / configure the tokenizer,
    3. how to turn a raw instruction into chat-formatted input tensors,
    4. where the decoder blocks live (so forward hooks can read the residual stream),
plus the end-of-instruction tokens (which token positions to read).

Their shipped `QwenModel` targets the ORIGINAL Qwen: it reads blocks via `model.transformer.h`
and does `tokenizer.eod_id` / `pad_token='<|extra_0|>'`. Qwen2.5 is a *Qwen2*-architecture
model — blocks live at `model.model.layers` and it uses a standard fast tokenizer — so their
class crashes on load. Rather than edit their file (forbidden), we provide THIS compatible
subclass and hand it to their unchanged `generate_directions`. That is "wrapping from outside".

Scope: extraction only. We compute the difference-in-means candidate cube and stop. We do NOT
run their `select_direction` (it needs generation + Qwen-specific refusal token ids), so the
steering / orthogonalization hooks below are deliberate stubs we never call.

Formatting note: `_tokenize` mirrors our own src/consequence/acts.py exactly — the model's
canonical chat template with add_generation_prompt=True, then left-padded tokenization. Both
sides of the gate therefore see byte-identical input strings, so the cosine tests the
extraction machinery and not a formatting difference.
"""

import functools

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.model_utils.model_base import ModelBase


def _tokenize(tokenizer, instructions):
    """Raw instruction -> chat-templated, left-padded input tensors. Matches acts.py."""
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ins}], tokenize=False, add_generation_prompt=True
        )
        for ins in instructions
    ]
    return tokenizer(texts, return_tensors="pt", padding=True)


class Qwen25Model(ModelBase):
    def _load_model(self, model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.padding_side = "left"  # generate_directions reads negative token positions
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def _get_tokenize_instructions_fn(self):
        return functools.partial(_tokenize, self.tokenizer)

    def _get_eoi_toks(self):
        # The tokens the chat template appends AFTER the user content — i.e. the
        # "<|im_end|>\n<|im_start|>assistant\n" tail. Their count sets how many trailing
        # positions get a candidate direction; index -1 is the last (final prompt) token.
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "{ins}"}], tokenize=False, add_generation_prompt=True
        )
        suffix = rendered.split("{ins}")[-1]
        return self.tokenizer.encode(suffix, add_special_tokens=False)

    def _get_refusal_toks(self):
        # Only used by select_direction, which we skip. Provide something valid anyway.
        return self.tokenizer.encode("I", add_special_tokens=False)

    def _get_model_block_modules(self):
        return self.model.model.layers  # Qwen2 architecture

    def _get_attn_modules(self):
        return torch.nn.ModuleList([b.self_attn for b in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([b.mlp for b in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction):
        raise NotImplementedError("adapter is extraction-only; steering is not wired")

    def _get_act_add_mod_fn(self, direction, coeff, layer):
        raise NotImplementedError("adapter is extraction-only; steering is not wired")
