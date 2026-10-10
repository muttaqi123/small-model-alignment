# %% [markdown]
# # Pairwise judging of DPO vs SFT generations (Kaggle Benchmarks task)
#
# Pushed with ``kaggle benchmarks tasks push alignment-pairwise-judge -f kaggle/judge_task.py -d <user>/alignment-judge-inputs``
# and run with ``kaggle benchmarks tasks run alignment-pairwise-judge -m gemini-3.6-flash``.
# The attached dataset holds the saved pairwise JSONL files (built by ``scripts/build_pairs.py``)
# and ``judge_pairwise_kaggle.py``; this task never generates model outputs. Previously
# judged files placed under ``judged/`` in the dataset are reused so reruns skip completed pairs.

# %%
import glob
import json
import shutil
import sys
from pathlib import Path

import kaggle_benchmarks as kbench

SCRIPT = sorted(glob.glob("/kaggle/input/**/judge_pairwise_kaggle.py", recursive=True))[0]
INPUT_DIR = Path(SCRIPT).parent
sys.path.insert(0, str(INPUT_DIR))
import judge_pairwise_kaggle as judge  # noqa: E402

OUT_DIR = Path("/kaggle/working/judgments")
OUT_DIR.mkdir(parents=True, exist_ok=True)
for previous in glob.glob(str(INPUT_DIR / "judged" / "*.jsonl")):
    target = OUT_DIR / Path(previous).name
    if not target.exists():
        shutil.copy(previous, target)
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


# %%
@kbench.task(name="alignment-pairwise-judge")
def alignment_pairwise_judge(llm) -> dict:
    call = judge.make_kbench_caller(llm)
    label = model_label(llm)
    results = {}
    for path in PAIR_FILES:
        stem = Path(path).name.replace("_pairs.jsonl", "")
        output = OUT_DIR / f"{stem}_judged.jsonl"
        summary = judge.run_file(path, output, OUT_DIR / f"{stem}_failures.jsonl", call, label, Settings)
        results[stem] = {"summary": summary, "rows": judge.read_jsonl(output)}
    (OUT_DIR / "all_results.json").write_text(json.dumps(results, indent=1))
    return results


alignment_pairwise_judge.run(kbench.llm)
