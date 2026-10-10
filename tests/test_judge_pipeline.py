"""Offline tests for pair building and the resumable judge (mock backend)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import judge_pairwise_kaggle as judge

ROOT = Path(__file__).resolve().parents[1]


def make_generations(tmp_path):
    rows = [{"id": f"x{i}", "instruction": f"question {i}", "category": "c1" if i % 2 else "c2"} for i in range(12)]
    dpo = [{**r, "output": f"dpo answer {i}", "generator": "dpo"} for i, r in enumerate(rows)]
    sft = [{**r, "output": f"sft answer {i}", "generator": "sft"} for i, r in enumerate(rows)]
    (tmp_path / "dpo.json").write_text(json.dumps(dpo))
    (tmp_path / "sft.json").write_text(json.dumps(sft))


def build(tmp_path):
    make_generations(tmp_path)
    subprocess.run(
        [
            sys.executable, "scripts/build_pairs.py",
            "--model_path", str(tmp_path / "dpo.json"), "--model_name", "dpo",
            "--baseline_path", str(tmp_path / "sft.json"), "--baseline_name", "sft",
            "--rubric", "general", "--output_path", str(tmp_path / "pairs.jsonl"),
            "--swapped_output_path", str(tmp_path / "pairs_swapped.jsonl"),
        ],
        cwd=ROOT, check=True, capture_output=True,
    )
    return judge.read_jsonl(tmp_path / "pairs.jsonl")


def test_pairs_are_randomised_and_mapping_is_consistent(tmp_path):
    pairs = build(tmp_path)
    assert len(pairs) == 12
    firsts = {p["hidden"]["model_a"] for p in pairs}
    assert firsts == {"dpo", "sft"}
    for p in pairs:
        model_a = p["hidden"]["model_a"]
        assert p["response_a"].startswith(f"{model_a} answer")
    swapped = judge.read_jsonl(tmp_path / "pairs_swapped.jsonl")
    for p, s in zip(pairs, swapped):
        assert s["response_a"] == p["response_b"] and s["hidden"]["model_a"] == p["hidden"]["model_b"]


def test_judge_prompt_never_reveals_model_identity(tmp_path):
    pairs = build(tmp_path)
    for p in pairs:
        p["response_a"] = p["response_b"] = "neutral text"
    prompt = judge.build_prompt(pairs[:5], "general")
    for secret in ("dpo", "sft", "dpo_vs_sft", "x1", "hidden", "target"):
        assert secret not in prompt
    assert "Response A" in prompt and "Response B" in prompt


def test_parse_response_handles_fences_and_bad_labels():
    text = '```json\n{"results": [{"pair": "P2", "winner": "b"}, {"pair": "P1", "winner": "Response A"}, {"pair": "P3", "winner": "maybe"}]}\n```'
    parsed = judge.parse_response(text, 3)
    assert parsed[1][0] == "A" and parsed[2][0] == "B" and parsed[3][0] is None
    assert judge.parse_response("not json at all", 3) == {}


def test_judge_maps_decisions_resumes_and_logs_failures(tmp_path):
    pairs = build(tmp_path)
    out, fail = tmp_path / "judged.jsonl", tmp_path / "failures.jsonl"
    always_a = lambda prompt: json.dumps(  # noqa: E731
        {"results": [{"pair": f"P{i}", "winner": "A"} for i in range(1, prompt.count("=== Pair P") + 1)]}
    )
    rows = judge.judge_pairs(pairs, out, fail, always_a, "test-judge", batch_size=5, sleep_seconds=0, max_retries=3)
    assert len(rows) == 12 and all(r["decision"] == "A" for r in rows)
    for row, pair in zip(rows, pairs):
        assert row["winner_model"] == pair["hidden"]["model_a"]
    summary = judge.summarize(rows)
    assert summary["wins"] == sum(p["hidden"]["model_a"] == "dpo" for p in pairs)
    assert summary["wins"] + summary["losses"] == 12 and summary["judge_picked_A_fraction"] == 1.0

    calls = []
    rows_again = judge.judge_pairs(pairs, out, fail, lambda p: calls.append(p) or "", "test-judge", sleep_seconds=0)
    assert calls == [] and len(rows_again) == 12

    def broken(prompt):
        raise RuntimeError("denied")

    out2 = tmp_path / "judged2.jsonl"
    rows = judge.judge_pairs(pairs[:3], out2, fail, broken, "test-judge", batch_size=5, sleep_seconds=0, max_retries=3)
    assert [r["decision"] for r in rows] == ["failure"] * 3 and all(r["attempts"] == 3 for r in rows)
    assert len(judge.read_jsonl(fail)) == 3
    rows = judge.judge_pairs(pairs[:3], out2, fail, lambda p: '{"results": []}', "test-judge", sleep_seconds=0, max_retries=2)
    assert [r["decision"] for r in rows] == ["failure"] * 3

    def partial(prompt):
        if prompt.count("=== Pair P") == 2:
            return json.dumps({"results": [{"pair": "P1", "winner": "tie"}, {"pair": "P2", "winner": "??"}]})
        return json.dumps({"results": [{"pair": "P1", "winner": "??"}]})

    out3 = tmp_path / "judged3.jsonl"
    rows = judge.judge_pairs(pairs[:2], out3, fail, partial, "test-judge", sleep_seconds=0, max_retries=2)
    assert rows[0]["decision"] == "tie" and rows[1]["decision"] == "unknown"
    assert judge.summarize(rows)["unknown"] == 1


def test_cli_mock_backend_writes_summary(tmp_path):
    build(tmp_path)
    subprocess.run(
        [
            sys.executable, "scripts/judge_pairwise_kaggle.py", "--backend", "mock",
            "--pairs_path", str(tmp_path / "pairs.jsonl"), "--output_path", str(tmp_path / "judged.jsonl"),
            "--sleep_seconds", "5", "--batch_size", "12",
        ],
        cwd=ROOT, check=True, capture_output=True,
    )
    summary = json.loads((tmp_path / "judged.jsonl.summary.json").read_text())
    assert summary["n"] == 12 and summary["wins"] + summary["losses"] + summary["ties"] == 12
