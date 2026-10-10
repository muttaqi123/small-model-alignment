"""Dataset preparation for SFT and the two from-scratch DPO runs.

All datasets are reduced to single-turn examples:

* SFT: ``HuggingFaceH4/ultrachat_200k`` - first user turn and first assistant
  reply of each conversation (``train_sft`` for training, ``test_sft`` for
  validation).
* HH-RLHF DPO: ``Anthropic/hh-rlhf`` - transcripts with exactly one Human and
  one Assistant turn on both sides, an identical prompt, and different
  responses (``train`` / ``test``).
* UltraFeedback DPO: ``HuggingFaceH4/ultrafeedback_binarized`` - two-message
  conversations with ``score_chosen > score_rejected`` and different responses
  (``train_prefs`` / ``test_prefs``).

Examples are shuffled with a fixed seed, then length-filtered with the training
tokenizer and template so that nothing has to be truncated during training:
SFT keeps examples whose full sequence fits in ``max_length``; DPO keeps pairs
whose formatted prompt fits in ``max_prompt_length`` and whose chosen and
rejected sequences both fit in ``max_length``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Iterable, Iterator

from alignment.prompting import format_prompt

HH_HUMAN = "\n\nHuman:"
HH_ASSISTANT = "\n\nAssistant:"


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_hh_single_turn(transcript: str) -> tuple[str, str] | None:
    """Split a one-turn HH-RLHF transcript into ``(prompt, response)``."""

    if transcript.count(HH_HUMAN) != 1 or transcript.count(HH_ASSISTANT) != 1:
        return None
    before, response = transcript.split(HH_ASSISTANT)
    if before.split(HH_HUMAN)[0].strip():
        return None
    prompt, response = before.split(HH_HUMAN)[1].strip(), response.strip()
    if not prompt or not response:
        return None
    return prompt, response


def iter_ultrachat(split, start_index: int = 0) -> Iterator[dict]:
    for index in range(start_index, len(split)):
        row = split[index]
        messages = row["messages"]
        if len(messages) < 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
            continue
        instruction, response = messages[0]["content"].strip(), messages[1]["content"].strip()
        if instruction and response:
            yield {"id": f"ultrachat-{row['prompt_id']}", "instruction": instruction, "response": response}


def iter_hh(split) -> Iterator[dict]:
    for index in range(len(split)):
        row = split[index]
        chosen, rejected = parse_hh_single_turn(row["chosen"]), parse_hh_single_turn(row["rejected"])
        if chosen is None or rejected is None or chosen[0] != rejected[0] or chosen[1] == rejected[1]:
            continue
        yield {"id": f"hh-{index}", "instruction": chosen[0], "chosen": chosen[1], "rejected": rejected[1]}


def iter_ultrafeedback(split) -> Iterator[dict]:
    for index in range(len(split)):
        row = split[index]
        chosen, rejected = row["chosen"], row["rejected"]
        if len(chosen) != 2 or len(rejected) != 2:
            continue
        if chosen[0]["role"] != "user" or chosen[1]["role"] != "assistant" or rejected[1]["role"] != "assistant":
            continue
        if row["score_chosen"] <= row["score_rejected"]:
            continue
        instruction = row["prompt"].strip()
        good, bad = chosen[1]["content"].strip(), rejected[1]["content"].strip()
        if not instruction or not good or not bad or good == bad:
            continue
        yield {
            "id": f"uf-{row['prompt_id'][:16]}-{index}",
            "instruction": instruction,
            "chosen": good,
            "rejected": bad,
            "score_chosen": row["score_chosen"],
            "score_rejected": row["score_rejected"],
        }


def _length(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def sft_filter(tokenizer, max_length: int) -> Callable[[dict], bool]:
    def keep(row: dict) -> bool:
        prompt_len = _length(tokenizer, format_prompt(row["instruction"]))
        return prompt_len + _length(tokenizer, row["response"]) + 1 <= max_length

    return keep


def dpo_filter(tokenizer, max_length: int, max_prompt_length: int) -> Callable[[dict], bool]:
    def keep(row: dict) -> bool:
        prompt_len = _length(tokenizer, format_prompt(row["instruction"]))
        if prompt_len > max_prompt_length:
            return False
        longest = max(_length(tokenizer, row["chosen"]), _length(tokenizer, row["rejected"]))
        return prompt_len + longest + 1 <= max_length

    return keep


def take_filtered(rows: Iterable[dict], keep: Callable[[dict], bool], limit: int) -> tuple[list[dict], dict]:
    """Take the first ``limit`` rows passing ``keep``; return rows and stats."""

    kept, seen, seen_ids = [], 0, set()
    for row in rows:
        seen += 1
        if row["instruction"] in seen_ids:
            continue
        if keep(row):
            kept.append(row)
            seen_ids.add(row["instruction"])
            if len(kept) >= limit:
                break
    return kept, {"candidates_examined": seen, "kept": len(kept)}


def shuffled(split, seed: int):
    return split.shuffle(seed=seed)


def prepare_sft(tokenizer, out_dir, n_train=30000, n_val=1000, max_length=512, seed=42) -> dict:
    from datasets import load_dataset

    train_split = shuffled(load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft"), seed)
    val_split = shuffled(load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft"), seed)
    keep = sft_filter(tokenizer, max_length)
    train, train_stats = take_filtered(iter_ultrachat(train_split), keep, n_train)
    val, val_stats = take_filtered(iter_ultrachat(val_split), keep, n_val)
    out_dir = Path(out_dir)
    write_jsonl(out_dir / "sft_train.jsonl", train)
    write_jsonl(out_dir / "sft_val.jsonl", val)
    return {"dataset": "HuggingFaceH4/ultrachat_200k", "train": train_stats, "val": val_stats}


def prepare_dpo(name, tokenizer, out_dir, n_train=15000, n_val=1000, max_length=512, max_prompt_length=256, seed=42) -> dict:
    from datasets import load_dataset

    if name == "hh":
        dataset_name, train_name, val_name, iterator = "Anthropic/hh-rlhf", "train", "test", iter_hh
    elif name == "uf":
        dataset_name, train_name, val_name, iterator = (
            "HuggingFaceH4/ultrafeedback_binarized",
            "train_prefs",
            "test_prefs",
            iter_ultrafeedback,
        )
    else:
        raise ValueError(f"unknown preference dataset {name!r}")
    keep = dpo_filter(tokenizer, max_length, max_prompt_length)
    train_split = shuffled(load_dataset(dataset_name, split=train_name), seed)
    val_split = shuffled(load_dataset(dataset_name, split=val_name), seed)
    train, train_stats = take_filtered(iterator(train_split), keep, n_train)
    val, val_stats = take_filtered(iterator(val_split), keep, n_val)
    out_dir = Path(out_dir)
    write_jsonl(out_dir / f"dpo_{name}_train.jsonl", train)
    write_jsonl(out_dir / f"dpo_{name}_val.jsonl", val)
    return {"dataset": dataset_name, "train": train_stats, "val": val_stats}


def sample_alpaca_eval(rows: list[dict], n: int = 300, seed: int = 42) -> list[dict]:
    """Fixed AlpacaEval subset: ``random.Random(seed).sample`` of row indices."""

    indices = random.Random(seed).sample(range(len(rows)), n)
    return [
        {
            "id": f"alpaca-eval-{index:03d}",
            "instruction": rows[index]["instruction"],
            "dataset": "alpaca_eval",
            "source_dataset": rows[index].get("dataset"),
        }
        for index in indices
    ]
