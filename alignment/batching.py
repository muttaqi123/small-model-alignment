"""Tokenization and batching for response-only language-model losses.

Every prompt/response pair is tokenized as two separate pieces and then
concatenated, so the prompt length in tokens is exactly
``len(tokenizer.encode(prompt, add_special_tokens=False))`` and the response
labels start at that position regardless of how the tokenizer would merge
characters across the boundary.

Conventions (shared by SFT and DPO):

* the EOS token (when the tokenizer defines one) is appended to the response
  *before* truncation, so it is dropped if the response is cut at
  ``max_length``;
* the concatenated sequence is right-truncated to ``max_length``;
* rows are right-padded; prompt and padding positions get label ``-100`` and
  padding positions get attention mask ``0``;
* row order is always preserved.
"""

from __future__ import annotations

from typing import Sequence

import torch

IGNORE_INDEX = -100


def _encode(tokenizer, text: str) -> list[int]:
    if not text:
        return []
    return list(tokenizer.encode(text, add_special_tokens=False))


def pad_token_id_for(tokenizer) -> int:
    """Padding id: the tokenizer's pad token, else EOS, else 0.

    Padding positions are always masked out explicitly (attention mask and
    ``-100`` labels), so the choice only affects the unused input ids.
    """

    if getattr(tokenizer, "pad_token_id", None) is not None:
        return int(tokenizer.pad_token_id)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return int(tokenizer.eos_token_id)
    return 0


def tokenize_pair(tokenizer, prompt: str, response: str, max_length: int) -> tuple[list[int], list[int]]:
    """Return ``(input_ids, labels)`` for one prompt/response pair, unpadded."""

    prompt_ids = _encode(tokenizer, prompt)
    response_ids = _encode(tokenizer, response)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        response_ids = response_ids + [int(eos_id)]
    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
    return input_ids, labels


def collate(
    examples: Sequence[tuple[list[int], list[int]]],
    pad_id: int,
    width: int | None = None,
) -> dict[str, torch.Tensor]:
    """Right-pad pre-tokenized ``(input_ids, labels)`` rows into tensors.

    ``width`` defaults to the longest row in ``examples``.
    """

    longest = max((len(ids) for ids, _ in examples), default=0)
    width = max(longest if width is None else width, 1)
    input_ids, attention_mask, labels = [], [], []
    for ids, labs in examples:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append(labs + [IGNORE_INDEX] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_lm_batch(
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
    pad_to_max_length: bool = True,
) -> dict[str, torch.Tensor]:
    """Build a response-masked LM batch.

    With ``pad_to_max_length`` every row is padded to exactly ``max_length``
    (the graded adapter behaviour); otherwise rows are padded to the longest
    row in the batch, which is cheaper for training.
    """

    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    examples = [tokenize_pair(tokenizer, p, r, max_length) for p, r in zip(prompts, responses)]
    width = max_length if pad_to_max_length else None
    return collate(examples, pad_token_id_for(tokenizer), width)


def build_sft_batch(tokenizer, prompts, responses, max_length, pad_to_max_length: bool = True):
    """Return ``input_ids``, ``attention_mask`` and response-only ``labels``."""

    return build_lm_batch(tokenizer, prompts, responses, max_length, pad_to_max_length)


def build_dpo_batch(
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    max_length,
    pad_to_max_length: bool = True,
):
    """Return ``{"chosen": batch, "rejected": batch}`` with a shared width.

    Both sides use the identical prompt formatting and are padded to the same
    width so they can be concatenated into a single forward pass.
    """

    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("prompts, chosen_responses and rejected_responses must align")
    chosen = [tokenize_pair(tokenizer, p, r, max_length) for p, r in zip(prompts, chosen_responses)]
    rejected = [tokenize_pair(tokenizer, p, r, max_length) for p, r in zip(prompts, rejected_responses)]
    if pad_to_max_length:
        width = max_length
    else:
        width = max(len(ids) for ids, _ in chosen + rejected) if prompts else 1
    pad_id = pad_token_id_for(tokenizer)
    return {"chosen": collate(chosen, pad_id, width), "rejected": collate(rejected, pad_id, width)}
