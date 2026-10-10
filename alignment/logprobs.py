"""Response-only log-probabilities for causal language models."""

from __future__ import annotations

import torch

from alignment.batching import IGNORE_INDEX, build_lm_batch


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum log p(label_t | x_<t) over unmasked label positions.

    The logits at position ``t`` predict the token at ``t + 1``, so logits are
    shifted left and labels right before gathering. Positions whose label is
    ``-100`` (prompt and padding) contribute nothing. The computation is done
    in float32 even when the model runs under fp16 autocast.

    Returns ``(summed_logps, token_counts)``, each of shape ``(batch,)``.
    """

    shift_labels = labels[:, 1:].to(logits.device)
    mask = shift_labels.ne(IGNORE_INDEX)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    sums = []
    # One row at a time keeps the float32 copy of the (seq, vocab) logits small.
    for row in range(logits.shape[0]):
        row_logits = logits[row, :-1, :].float()
        token_logits = torch.gather(row_logits, dim=-1, index=safe_labels[row].unsqueeze(-1)).squeeze(-1)
        token_logps = (token_logits - torch.logsumexp(row_logits, dim=-1)) * mask[row]
        sums.append(token_logps.sum())
    if not sums:
        return logits.new_zeros(0, dtype=torch.float32), mask.sum(dim=-1)
    return torch.stack(sums), mask.sum(dim=-1)


def batch_logprobs(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``model`` on an LM batch and return per-row summed response logps."""

    device = model_device(model)
    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )
    return sequence_logprobs(outputs.logits, batch["labels"].to(device))


def response_logprobs(model, tokenizer, prompts, responses, max_length) -> torch.Tensor:
    """One summed response-token log-probability per input row."""

    batch = build_lm_batch(tokenizer, prompts, responses, max_length)
    logps, _ = batch_logprobs(model, batch)
    return logps
