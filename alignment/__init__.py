"""Small-model alignment: from-scratch SFT and DPO for Qwen2.5-0.5B."""

from alignment.batching import build_dpo_batch, build_lm_batch, build_sft_batch
from alignment.logprobs import response_logprobs, sequence_logprobs
from alignment.losses import dpo_loss, dpo_loss_from_logps, dpo_metrics, sft_loss
from alignment.prompting import format_prompt

__all__ = [
    "build_dpo_batch",
    "build_lm_batch",
    "build_sft_batch",
    "dpo_loss",
    "dpo_loss_from_logps",
    "dpo_metrics",
    "format_prompt",
    "response_logprobs",
    "sequence_logprobs",
    "sft_loss",
]
