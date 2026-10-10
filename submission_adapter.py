"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function should be a thin
wrapper around your own implementation; you do not need to place the training
code itself in this file.

Implementation lives in the ``alignment`` package:

* ``alignment/batching.py`` - tokenization, EOS handling, truncation, padding,
  and ``-100`` masking of prompt/padding labels;
* ``alignment/logprobs.py`` - shifted, response-only summed log-probabilities;
* ``alignment/losses.py`` - SFT loss (token-level mean over response tokens)
  and the DPO objective.

Batches are right-padded to exactly ``max_length``; prompt tokens are encoded
with ``add_special_tokens=False`` and the EOS token is appended to every
response before right-truncation to ``max_length``.
"""

from __future__ import annotations

from alignment import batching, logprobs, losses


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return batching.build_sft_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    return logprobs.response_logprobs(model, tokenizer, prompts, responses, max_length)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    return losses.sft_loss(model, tokenizer, prompts, responses, max_length)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return batching.build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    return losses.dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    return losses.dpo_loss(
        policy_model,
        reference_model,
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        beta,
        max_length,
    )
