"""SFT and DPO objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from alignment.batching import build_dpo_batch, build_lm_batch
from alignment.logprobs import batch_logprobs


def sft_loss_from_batch(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Token-level mean negative log-likelihood over unmasked response tokens.

    Reduction: sum of response-token NLL over the whole batch divided by the
    number of response tokens in the batch (the "mean over all unmasked
    response tokens" option in the manual).
    """

    logps, counts = batch_logprobs(model, batch)
    return -logps.sum() / counts.sum().clamp_min(1)


def sft_loss(model, tokenizer, prompts, responses, max_length) -> torch.Tensor:
    batch = build_lm_batch(tokenizer, prompts, responses, max_length)
    return sft_loss_from_batch(model, batch)


def dpo_loss_from_logps(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Batch mean of ``-log sigmoid(beta * (Δ_policy - Δ_reference))``."""

    device = policy_chosen_logps.device
    policy_gap = policy_chosen_logps.float() - policy_rejected_logps.float().to(device)
    reference_gap = reference_chosen_logps.float().to(device) - reference_rejected_logps.float().to(device)
    return -F.logsigmoid(beta * (policy_gap - reference_gap)).mean()


@torch.no_grad()
def dpo_metrics(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Per-example implicit rewards and preference diagnostics.

    ``r_w = beta * (log pi(y_w|x) - log pi_ref(y_w|x))`` and likewise ``r_l``.
    Accuracy uses the strict comparison ``r_w > r_l`` (ties count as wrong).
    """

    device = policy_chosen_logps.device
    chosen_rewards = beta * (policy_chosen_logps.float() - reference_chosen_logps.float().to(device))
    rejected_rewards = beta * (policy_rejected_logps.float().to(device) - reference_rejected_logps.float().to(device))
    return {
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "correct": (chosen_rewards > rejected_rewards).float(),
        "margins": chosen_rewards - rejected_rewards,
    }


def policy_and_reference_logps(policy_model, reference_model, batch):
    """Score chosen and rejected sequences with both models.

    The reference forward pass runs first under ``torch.no_grad()`` (on its own
    device, so on a two-GPU machine it can overlap with the policy pass). The
    chosen and rejected rows share a width, so each model does one forward
    pass over the concatenated ``[chosen; rejected]`` batch.
    """

    n = batch["chosen"]["input_ids"].shape[0]
    joint = {key: torch.cat([batch["chosen"][key], batch["rejected"][key]], dim=0) for key in batch["chosen"]}
    with torch.no_grad():
        reference_logps, _ = batch_logprobs(reference_model, joint)
    policy_logps, _ = batch_logprobs(policy_model, joint)
    reference_logps = reference_logps.detach().to(policy_logps.device)
    return policy_logps[:n], policy_logps[n:], reference_logps[:n], reference_logps[n:]


def dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
) -> torch.Tensor:
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    logps = policy_and_reference_logps(policy_model, reference_model, batch)
    return dpo_loss_from_logps(*logps, beta)
