"""Prompt template shared by SFT, DPO, and generation.

We use a ChatML-style single-turn template without a system message:

    <|im_start|>user
    {instruction}<|im_end|>
    <|im_start|>assistant
    {response}<|im_end|>

``<|im_start|>`` and ``<|im_end|>`` are single special tokens in the Qwen2.5
vocabulary, so the prompt/response boundary is always a clean token boundary.
For our pipeline the tokenizer's EOS token is set to ``<|im_end|>`` (see
``prepare_tokenizer``), so "append EOS to the response" closes the assistant
turn, and generation stops on the same token the model was trained to emit.
"""

from __future__ import annotations

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"

PROMPT_TEMPLATE = IM_START + "user\n{instruction}" + IM_END + "\n" + IM_START + "assistant\n"


def format_prompt(instruction: str) -> str:
    """Return the model-facing prompt for a single-turn instruction."""

    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def prepare_tokenizer(tokenizer):
    """Configure a Qwen2.5 tokenizer for the ChatML template used everywhere.

    * EOS becomes ``<|im_end|>`` (the end-of-turn token), so responses are
      terminated with it during SFT/DPO and generation stops on it.
    * PAD stays ``<|endoftext|>``, which is therefore distinct from EOS.
    * Right padding is used for training; generation code switches to left
      padding locally.
    """

    vocab = tokenizer.get_vocab()
    if IM_END in vocab:
        tokenizer.eos_token = IM_END
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        if ENDOFTEXT in vocab and ENDOFTEXT != tokenizer.eos_token:
            tokenizer.pad_token = ENDOFTEXT
        elif tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def stop_token_ids(tokenizer) -> list[int]:
    """Token ids that should terminate generation."""

    ids = []
    vocab = tokenizer.get_vocab()
    for token in (IM_END, ENDOFTEXT):
        if token in vocab:
            ids.append(vocab[token])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    return ids
