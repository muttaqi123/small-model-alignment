# %% [markdown]
# # Pairwise judging of DPO vs SFT generations (Kaggle Benchmarks task)
#
# Pushed with ``kaggle benchmarks tasks push alignment-pairwise-judge-extra -f kaggle/judge_task.py -d <user>/alignment-judge-inputs-extra``
# and run with ``kaggle benchmarks tasks run alignment-pairwise-judge-extra -m gemini-3.6-flash``.
# The attached dataset holds the saved pairwise JSONL files (built by ``scripts/build_pairs.py``)
# and ``judge_pairwise_kaggle.py``; this task never generates model outputs.
#
# * The primary judge (``PRIMARY_JUDGE``) judges every pair. Any other model (for example the
#   platform default used when the task is first created) judges only the first
#   ``SECOND_JUDGE_SAMPLE`` pairs of each file, which is used to measure judge-judge agreement.
# * Judged files placed under ``judged/`` in the dataset are reused only if they were produced by
#   the same judge model, so reruns skip completed pairs.

# %%
import glob
import json
import sys
from pathlib import Path

import kaggle_benchmarks as kbench

SCRIPT = sorted(glob.glob("/kaggle/input/**/judge_pairwise_kaggle.py", recursive=True))[0]
INPUT_DIR = Path(SCRIPT).parent
sys.path.insert(0, str(INPUT_DIR))
import judge_pairwise_kaggle as judge  # noqa: E402

PRIMARY_JUDGE = "gemini-3.6-flash"
SECOND_JUDGE_SAMPLE = 60
OUT_DIR = Path("/kaggle/working/judgments")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAIR_FILES = sorted(glob.glob(str(INPUT_DIR / "pairs" / "*_pairs.jsonl")))
print("pair files:", [Path(p).name for p in PAIR_FILES])


class Settings:
    batch_size = 5
    sleep_seconds = 5.0
    max_retries = 3
    reuse_judgments = True


def model_label(llm) -> str:
    for attr in ("model", "model_name", "name"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return str(llm)


def restore_previous(label: str) -> None:
    for previous in glob.glob(str(INPUT_DIR / "judged" / "*_judged.jsonl")):
        rows = judge.read_jsonl(previous)
        target = OUT_DIR / Path(previous).name
        if rows and all(r.get("judge_model") == label for r in rows) and not target.exists():
            judge.write_jsonl(target, rows)
            print(f"reusing {len(rows)} judgments from {Path(previous).name}")


# %%
@kbench.task(name="alignment-pairwise-judge-extra")
def alignment_pairwise_judge_extra(llm) -> dict:
    call = judge.make_kbench_caller(llm)
    label = model_label(llm)
    primary = PRIMARY_JUDGE in label
    print(f"judge model: {label} ({'primary: all pairs' if primary else f'secondary: first {SECOND_JUDGE_SAMPLE} pairs per file'})")
    restore_previous(label)
    results = {}
    for path in PAIR_FILES:
        stem = Path(path).name.replace("_pairs.jsonl", "")
        pairs = judge.read_jsonl(path)
        if not primary:
            pairs = pairs[:SECOND_JUDGE_SAMPLE]
            subset = OUT_DIR / f"{stem}_subset_pairs.jsonl"
            judge.write_jsonl(subset, pairs)
            path = subset
        output = OUT_DIR / f"{stem}_judged.jsonl"
        summary = judge.run_file(path, output, OUT_DIR / f"{stem}_failures.jsonl", call, label, Settings)
        results[stem] = {"summary": summary, "rows": judge.read_jsonl(output)}
    (OUT_DIR / "all_results.json").write_text(json.dumps(results, indent=1))
    return results


alignment_pairwise_judge_extra.run(kbench.llm)
