"""Stricter checks of the adapter: truncation, tokenizer quirks, numerics, gradients."""

from __future__ import annotations

import math

import pytest
import torch

import submission_adapter as adapter
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


class LeftPaddingTokenizer(ToyTokenizer):
    padding_side = "left"


def manual_logp(model, tokenizer, response, with_eos=True):
    """Sum of log-softmax(bias)[token] over the response tokens (position-free toy LM)."""

    log_probs = torch.log_softmax(model.logit_bias.detach().float(), dim=-1)
    ids = tokenizer.encode(response) + ([tokenizer.eos_token_id] if with_eos else [])
    return float(sum(log_probs[i] for i in ids))


def test_sft_batch_is_padded_to_max_length_with_exact_masks():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["ab: ", "a: "], ["xyz", "q"], max_length=12)
    assert batch["input_ids"].shape == (2, 12)
    # row 0: 4 prompt tokens, 3 response tokens + eos
    assert batch["labels"][0].tolist() == [-100] * 4 + tok.encode("xyz") + [1] + [-100] * 4
    assert batch["attention_mask"][0].tolist() == [1] * 8 + [0] * 4
    # row 1: 3 prompt tokens, 1 response token + eos
    assert batch["labels"][1].tolist() == [-100] * 3 + tok.encode("q") + [1] + [-100] * 7
    assert batch["input_ids"][1, 5:].eq(tok.pad_token_id).all()
    assert batch["input_ids"].dtype == batch["labels"].dtype == torch.long


def test_eos_is_dropped_when_response_is_truncated():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["P: "], ["abcdef"], max_length=6)
    assert batch["input_ids"].shape == (1, 6)
    assert batch["labels"][0].tolist() == [-100] * 3 + tok.encode("abc")
    assert tok.eos_token_id not in batch["input_ids"][0].tolist()


def test_eos_kept_when_sequence_exactly_fits():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["P: "], ["ab"], max_length=6)
    assert batch["labels"][0].tolist() == [-100] * 3 + tok.encode("ab") + [tok.eos_token_id]
    assert batch["attention_mask"][0].tolist() == [1] * 6


def test_prompt_longer_than_max_length_yields_no_response_tokens():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    batch = adapter.build_sft_batch(tok, ["a very long prompt"], ["resp"], max_length=5)
    assert batch["labels"].eq(-100).all()
    logps = adapter.compute_response_logprobs(model, tok, ["a very long prompt"], ["resp"], max_length=5)
    assert logps.tolist() == [0.0]


def test_missing_pad_token_falls_back_to_eos_and_missing_eos_is_not_appended():
    tok = NoPadTokenizer()
    batch = adapter.build_sft_batch(tok, ["a: ", "abc: "], ["x", "y"], max_length=10)
    assert batch["input_ids"][0, 5:].eq(tok.eos_token_id).all()
    assert batch["labels"][0, 5:].eq(-100).all()

    tok = NoEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["a: "], ["xy"], max_length=10)
    assert batch["labels"][0].tolist() == [-100] * 3 + tok.encode("xy") + [-100] * 5


def test_left_padding_side_does_not_change_scores():
    model = ToyCausalLM(ToyTokenizer().vocab_size, bias_scale=0.9)
    right = adapter.compute_response_logprobs(model, ToyTokenizer(), ["Q: ", "Long Q: "], ["ab", "c"], 16)
    left = adapter.compute_response_logprobs(model, LeftPaddingTokenizer(), ["Q: ", "Long Q: "], ["ab", "c"], 16)
    torch.testing.assert_close(left, right)


def test_response_logprobs_match_manual_values_and_preserve_row_order():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=1.3)
    prompts = ["zz: ", "a: ", "mmmmm: "]
    responses = ["hello", "x", "ok!"]
    logps = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=32)
    expected = torch.tensor([manual_logp(model, tok, r) for r in responses])
    torch.testing.assert_close(logps, expected, rtol=1e-5, atol=1e-5)

    reversed_logps = adapter.compute_response_logprobs(model, tok, prompts[::-1], responses[::-1], max_length=32)
    torch.testing.assert_close(reversed_logps, expected.flip(0), rtol=1e-5, atol=1e-5)


def test_scores_do_not_depend_on_batch_partners_or_padding_width():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.4)
    alone = adapter.compute_response_logprobs(model, tok, ["Q: "], ["abc"], max_length=8)
    batched = adapter.compute_response_logprobs(model, tok, ["Q: ", "Much longer: "], ["abc", "defghij"], 40)
    torch.testing.assert_close(alone[0], batched[0])


def test_sft_loss_is_token_mean_over_response_tokens():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.8)
    prompts, responses = ["A: ", "B: "], ["a", "bcdef"]
    loss = adapter.compute_sft_loss(model, tok, prompts, responses, max_length=20)
    total = sum(manual_logp(model, tok, r) for r in responses)
    n_tokens = sum(len(r) + 1 for r in responses)
    assert loss.ndim == 0
    assert math.isclose(loss.item(), -total / n_tokens, rel_tol=1e-5)


def test_dpo_batch_matches_sft_batches_for_each_side():
    tok = ToyTokenizer()
    prompts, chosen, rejected = ["T: ", "Q: "], ["good", "yes!"], ["bad", "no"]
    batch = adapter.build_dpo_batch(tok, prompts, chosen, rejected, max_length=12)
    torch.testing.assert_close(batch["chosen"]["input_ids"], adapter.build_sft_batch(tok, prompts, chosen, 12)["input_ids"])
    torch.testing.assert_close(batch["rejected"]["labels"], adapter.build_sft_batch(tok, prompts, rejected, 12)["labels"])
    # identical prompt formatting on both sides
    assert torch.equal(batch["chosen"]["input_ids"][:, :3], batch["rejected"]["input_ids"][:, :3])


def test_dpo_loss_from_logps_matches_formula():
    pc = torch.tensor([-1.0, -4.0, -2.5])
    pr = torch.tensor([-3.0, -2.0, -2.5])
    rc = torch.tensor([-2.0, -3.0, -1.0])
    rr = torch.tensor([-2.5, -2.5, -4.0])
    beta = 0.1
    loss = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    z = beta * ((pc - pr) - (rc - rr))
    expected = torch.log1p(torch.exp(-z)).mean()
    torch.testing.assert_close(loss, expected)
    # large margins must stay finite
    big = adapter.compute_dpo_loss_from_logps(torch.tensor([0.0]), torch.tensor([-1e4]), torch.tensor([0.0]), torch.tensor([0.0]), 1.0)
    assert torch.isfinite(big) and big.item() < 1e-6
    big = adapter.compute_dpo_loss_from_logps(torch.tensor([-1e4]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), 1.0)
    assert torch.isfinite(big) and math.isclose(big.item(), 1e4, rel_tol=1e-6)


def test_dpo_loss_value_and_gradient_direction():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    reference = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    prompts, chosen, rejected = ["P: ", "Q: "], ["aa", "a"], ["bb", "b"]
    loss = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, beta=0.5, max_length=10)
    # identical policy and reference => loss is exactly log 2
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))
    loss.backward()
    assert reference.logit_bias.grad is None
    a, b = tok.vocab["a"], tok.vocab["b"]
    # descending the gradient raises the chosen token and lowers the rejected one
    assert policy.logit_bias.grad[a] < 0 < policy.logit_bias.grad[b]
    with torch.no_grad():
        policy.logit_bias -= 1.0 * policy.logit_bias.grad
    new_loss = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, beta=0.5, max_length=10)
    assert new_loss.item() < loss.item()


def test_dpo_loss_matches_manual_logps():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size, bias_scale=1.0)
    reference = ToyCausalLM(tok.vocab_size, bias_scale=-0.7)
    prompts, chosen, rejected = ["P: ", "Long prompt: "], ["xyz", "k"], ["abc", "longer reply"]
    beta = 0.3
    loss = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, beta=beta, max_length=40)
    pc = torch.tensor([manual_logp(policy, tok, r) for r in chosen])
    pr = torch.tensor([manual_logp(policy, tok, r) for r in rejected])
    rc = torch.tensor([manual_logp(reference, tok, r) for r in chosen])
    rr = torch.tensor([manual_logp(reference, tok, r) for r in rejected])
    expected = -torch.nn.functional.logsigmoid(beta * ((pc - pr) - (rc - rr))).mean()
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-5)


def test_real_transformer_scores_are_padding_invariant():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    tok = ToyTokenizer()
    config = transformers.Qwen2Config(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    with torch.no_grad():
        alone = adapter.compute_response_logprobs(model, tok, ["Q: "], ["abc"], max_length=7)
        batched = adapter.compute_response_logprobs(model, tok, ["Q: ", "A much longer prompt: "], ["abc", "z"], 64)
    torch.testing.assert_close(alone[0], batched[0], rtol=1e-4, atol=1e-4)
