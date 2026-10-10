"""Generate the Kaggle notebooks (and kernel-metadata.json) for every stage.

    python kaggle/build_notebooks.py --username <kaggle-user> --commit <git-sha>

Each stage becomes ``kaggle/<slug>/<slug>.ipynb`` and can be pushed with
``kaggle kernels push -p kaggle/<slug>``. Every notebook clones this repository
at a pinned commit, installs dependencies without touching Kaggle's PyTorch,
prints the environment, runs the public tests, and writes all artifacts under
``/kaggle/working/small-model-alignment/outputs/``. Later stages attach the
outputs of earlier ones through ``kernel_sources``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = "https://github.com/muttaqi123/small-model-alignment.git"
WORK = "/kaggle/working/small-model-alignment"

SETUP = r'''
import glob, json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO_URL = "{repo}"
COMMIT = "{commit}"
WORK = Path("{work}")
os.chdir("/kaggle/working")
if not WORK.exists():
    subprocess.run(["git", "clone", "-q", REPO_URL, str(WORK)], check=True)
subprocess.run(["git", "-C", str(WORK), "fetch", "-q", "origin"], check=True)
subprocess.run(["git", "-C", str(WORK), "checkout", "-q", COMMIT], check=True)
os.chdir(WORK)
os.environ["PYTHONPATH"] = str(WORK)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)


def run(cmd, env=None):
    """Run a shell command, stream its output, and fail the notebook on error."""
    print(f"$ {{cmd}}", flush=True)
    merged = {{**os.environ, **(env or {{}})}}
    proc = subprocess.Popen(cmd, shell=True, env=merged, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f"command failed ({{proc.returncode}}): {{cmd}}")


def start(cmd, log_path, env=None):
    """Start a background command (e.g. generation on the second GPU)."""
    print(f"$ {{cmd}} > {{log_path}} &", flush=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "w")
    merged = {{**os.environ, **(env or {{}})}}
    return subprocess.Popen(cmd, shell=True, env=merged, stdout=handle, stderr=subprocess.STDOUT, text=True), log_path


def wait(job):
    proc, log_path = job
    code = proc.wait()
    print(Path(log_path).read_text()[-3000:])
    if code != 0:
        raise RuntimeError(f"background job failed: {{log_path}}")


def archive(zip_path, items, exclude=("trainer", "best", "final", "adapter")):
    """Zip small artifacts (metrics, configs, generations, plots); never weights."""
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in items:
            p = Path(item)
            files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
            for f in files:
                if any(part in exclude for part in f.parts) or f.suffix in (".safetensors", ".bin", ".pt"):
                    continue
                zf.write(f, str(f))
    print(f"{{zip_path}}: {{Path(zip_path).stat().st_size / 1e6:.1f}} MB")


def find_input(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{{pattern}}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"no input matches {{pattern}}")
    print(f"found {{pattern}} -> {{hits[0]}}")
    return hits[0]
'''

INSTALL = r'''
run("nvidia-smi")
run("python -m pip uninstall -y -q torchao || true")
run('python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest matplotlib{extra}')
run("python -m pip install -q -e . --no-deps")
run('python -c "import torch, transformers, datasets; print(\'torch\', torch.__version__, \'| transformers\', transformers.__version__, \'| datasets\', datasets.__version__, \'| cuda\', torch.version.cuda, \'| gpus\', torch.cuda.device_count())"')
run("python -m pytest -q tests/")
'''

GEN = "python -m alignment.generate --batch_size 16 --max_new_tokens 512 --repetition_penalty 1.1"


def cells_pilot():
    return [
        "# Pilot: every stage at tiny scale (paths, versions, memory, throughput)",
        r'''
run("python scripts/prepare_data.py --out_dir outputs/data")
run("python -m alignment.sft --train_path outputs/data/sft_train.jsonl --val_path outputs/data/sft_val.jsonl "
    "--output_dir outputs/pilot/sft --max_train_examples 320 --max_val_examples 64 --eval_every 5")
''',
        r'''
import json
for line in open("outputs/pilot/sft/metrics.jsonl"):
    print(line.strip())
''',
        r'''
run("python -m alignment.dpo --sft_path outputs/pilot/sft/best --train_path outputs/data/dpo_hh_train.jsonl "
    "--val_path outputs/data/dpo_hh_val.jsonl --output_dir outputs/pilot/dpo_hh --max_train_examples 320 "
    "--max_val_examples 64 --eval_every 5")
run("python -m alignment.dpo --sft_path outputs/pilot/sft/best --train_path outputs/data/dpo_uf_train.jsonl "
    "--val_path outputs/data/dpo_uf_val.jsonl --output_dir outputs/pilot/dpo_uf --max_train_examples 96 "
    "--max_val_examples 32 --eval_every 1")
''',
        r'''
run(f"{gen} --model_path Qwen/Qwen2.5-0.5B --eval_path eval_sets/alpaca_eval_300_seed42.json --limit 4 "
    "--output_path outputs/pilot/gen_base.json --generator qwen2.5-0.5b-base --dataset alpaca_eval")
run(f"{gen} --model_path outputs/pilot/sft/best --eval_path eval_sets/hh_rlhf_eval_300_seed42.json --limit 4 "
    "--output_path outputs/pilot/gen_sft.json --generator pilot-sft --device cuda:1")
for f in ("outputs/pilot/gen_base.json", "outputs/pilot/gen_sft.json"):
    for r in json.load(open(f)):
        print("=" * 80, "\n", r["instruction"][:300], "\n---\n", r["output"][:1500])
'''.replace("{gen}", GEN),
        r'''
run("python scripts/train_lora_dpo.py --output_dir outputs/pilot/lora --configs r16_all_lr5e-5 --epochs 1",
    env={"CUDA_VISIBLE_DEVICES": "0"})
run(f"{gen} --model_path HuggingFaceTB/SmolLM2-360M-Instruct --adapter_path outputs/pilot/lora/r16_all_lr5e-5/adapter "
    "--prompt_format chat_template --eval_path eval_sets/persona_style_eval_40.json --limit 4 "
    "--output_path outputs/pilot/gen_lora.json --generator pilot-lora")
for r in json.load(open("outputs/pilot/gen_lora.json")):
    print("=" * 80, "\n", r["instruction"], "\n---\n", r["output"][:800])
'''.replace("{gen}", GEN),
        r'''
run("python scripts/plot_metrics.py --sft outputs/pilot/sft --dpo outputs/pilot/dpo_hh outputs/pilot/dpo_uf --lora outputs/pilot/lora --out_dir outputs/pilot/plots")
shutil.rmtree("outputs/pilot/sft/final", ignore_errors=True)
for d in glob.glob("outputs/pilot/dpo_*/*/"):
    shutil.rmtree(d, ignore_errors=True)
shutil.rmtree("outputs/data", ignore_errors=True)
run("du -sh outputs/pilot/*")
''',
    ]


def cells_sft():
    return [
        "# Stage 1: data preparation, SFT, and base/SFT generations",
        r'''
run("python scripts/prepare_data.py --out_dir outputs/data")
print(open("outputs/data/data_manifest.json").read()[:3000])
''',
        r'''
# Base-model generations run on the second GPU while SFT trains on the first.
base_alpaca = start(f"{gen} --model_path Qwen/Qwen2.5-0.5B --eval_path eval_sets/alpaca_eval_300_seed42.json "
                    "--output_path outputs/generations/alpaca_base.json --generator qwen2.5-0.5b-base --dataset alpaca_eval --device cuda:1",
                    "outputs/logs/gen_alpaca_base.log")
run("python -m alignment.sft --train_path outputs/data/sft_train.jsonl --val_path outputs/data/sft_val.jsonl "
    "--output_dir outputs/sft --eval_every 50")
wait(base_alpaca)
'''.replace("{gen}", GEN),
        r'''
# Sanity check on 3 prompts before the full SFT generations.
run(f"{gen} --model_path outputs/sft/best --eval_path eval_sets/alpaca_eval_300_seed42.json --limit 3 "
    "--output_path outputs/generations/sanity_sft.json --generator qwen2.5-0.5b-sft --dataset alpaca_eval")
for r in json.load(open("outputs/generations/sanity_sft.json")):
    print("=" * 80, "\n", r["instruction"][:300], "\n---\n", r["output"], "\n[finished:", r["finished"], "]")
'''.replace("{gen}", GEN),
        r'''
jobs = [
    start(f"{gen} --model_path outputs/sft/best --eval_path eval_sets/alpaca_eval_300_seed42.json "
          "--output_path outputs/generations/alpaca_sft.json --generator qwen2.5-0.5b-sft --dataset alpaca_eval --device cuda:0",
          "outputs/logs/gen_alpaca_sft.log"),
    start(f"{gen} --model_path outputs/sft/best --eval_path eval_sets/hh_rlhf_eval_300_seed42.json "
          "--output_path outputs/generations/hh_sft.json --generator qwen2.5-0.5b-sft --device cuda:1",
          "outputs/logs/gen_hh_sft.log"),
]
for job in jobs:
    wait(job)
run(f"{gen} --model_path Qwen/Qwen2.5-0.5B --eval_path eval_sets/hh_rlhf_eval_300_seed42.json "
    "--output_path outputs/generations/hh_base.json --generator qwen2.5-0.5b-base")
'''.replace("{gen}", GEN),
        r'''
run("python scripts/plot_metrics.py --sft outputs/sft --out_dir outputs/plots")
shutil.rmtree("outputs/sft/final.tmp", ignore_errors=True)
archive("sft_run_outputs.zip", ["outputs/sft/metrics.jsonl", "outputs/sft/run_config.json", "outputs/sft/summary.json",
        "outputs/generations", "outputs/logs", "outputs/plots", "outputs/data/data_manifest.json"])
run("du -sh outputs/* && ls -la outputs/sft")
''',
    ]


def cells_dpo(name):
    eval_set = "hh_rlhf_eval_300_seed42.json" if name == "hh" else "alpaca_eval_300_seed42.json"
    other_set = "alpaca_eval_300_seed42.json" if name == "hh" else "hh_rlhf_eval_300_seed42.json"
    main_prefix, other_prefix = ("hh", "alpaca") if name == "hh" else ("alpaca", "hh")
    label = f"qwen2.5-0.5b-dpo-{name}"
    return [
        f"# Stage 2: from-scratch DPO on {'HH-RLHF' if name == 'hh' else 'UltraFeedback'} preference pairs",
        r'''
sft_best = str(Path(find_input("outputs/sft/best/config.json")).parent)
data_dir = str(Path(find_input("outputs/data/data_manifest.json")).parent)
Path("outputs/sft").mkdir(parents=True, exist_ok=True)
for f in glob.glob(f"{data_dir}/../sft/*.json*"):
    shutil.copy(f, "outputs/sft/")
print(sft_best, data_dir)
''',
        r'''
run(f"python -m alignment.dpo --sft_path {sft_best} --train_path {data_dir}/dpo_NAME_train.jsonl "
    f"--val_path {data_dir}/dpo_NAME_val.jsonl --output_dir outputs/dpo_NAME --eval_every 25")
'''.replace("NAME", name),
        r'''
run(f"GEN --model_path outputs/dpo_NAME/best --eval_path eval_sets/EVAL --limit 3 "
    "--output_path outputs/generations/sanity_dpo_NAME.json --generator LABEL")
for r in json.load(open("outputs/generations/sanity_dpo_NAME.json")):
    print("=" * 80, "\n", r["instruction"][:300], "\n---\n", r["output"], "\n[finished:", r["finished"], "]")
jobs = [
    start("GEN --model_path outputs/dpo_NAME/best --eval_path eval_sets/EVAL "
          "--output_path outputs/generations/MAIN_dpo_NAME.json --generator LABEL --dataset DATASET --device cuda:0",
          "outputs/logs/gen_MAIN_dpo_NAME.log"),
    start("GEN --model_path outputs/dpo_NAME/best --eval_path eval_sets/OTHER "
          "--output_path outputs/generations/OTHERP_dpo_NAME.json --generator LABEL --dataset OTHERDS --device cuda:1",
          "outputs/logs/gen_OTHERP_dpo_NAME.log"),
]
for job in jobs:
    wait(job)
'''.replace("GEN", GEN)
        .replace("NAME", name)
        .replace("OTHERP", other_prefix)
        .replace("OTHERDS", "alpaca_eval" if other_prefix == "alpaca" else "hh_rlhf_style_eval")
        .replace("OTHER", other_set)
        .replace("MAIN", main_prefix)
        .replace("EVAL", eval_set)
        .replace("DATASET", "alpaca_eval" if main_prefix == "alpaca" else "hh_rlhf_style_eval")
        .replace("LABEL", label),
        r'''
run("python scripts/plot_metrics.py --sft outputs/sft --dpo outputs/dpo_NAME --out_dir outputs/plots_NAME")
archive("dpo_NAME_run_outputs.zip", ["outputs/dpo_NAME", "outputs/generations", "outputs/logs", "outputs/plots_NAME"])
run("du -sh outputs/* && ls -la outputs/dpo_NAME")
'''.replace("NAME", name),
    ]


def cells_lora():
    return [
        "# Stage 3: TRL DPOTrainer + LoRA persona tuning (SmolLM2-360M-Instruct)",
        r'''
run("python scripts/train_lora_dpo.py --output_dir outputs/lora_dpo", env={"CUDA_VISIBLE_DEVICES": "0"})
print(open("outputs/lora_dpo/sweep_summary.json").read()[:6000])
''',
        r'''
base = "HuggingFaceTB/SmolLM2-360M-Instruct"
run(f"{gen} --model_path {base} --prompt_format chat_template --eval_path eval_sets/persona_style_eval_40.json "
    "--output_path outputs/generations/persona_base.json --generator smollm2-360m-instruct")
for name in ("r8_attn_lr5e-5", "r16_all_lr5e-5", "r16_all_lr2e-4"):
    run(f"{gen} --model_path {base} --adapter_path outputs/lora_dpo/{name}/adapter --prompt_format chat_template "
        f"--eval_path eval_sets/persona_style_eval_40.json --output_path outputs/generations/persona_lora_{name}.json "
        f"--generator smollm2-360m-lora-dpo-{name}")
for name in ("persona_base", "persona_lora_r16_all_lr5e-5", "persona_lora_r16_all_lr2e-4"):
    print("#" * 30, name)
    for r in json.load(open(f"outputs/generations/{name}.json"))[::8]:
        print("=" * 80, "\n", r["instruction"], "\n---\n", r["output"][:700])
'''.replace("{gen}", GEN.replace("--max_new_tokens 512", "--max_new_tokens 256")),
        r'''
run("python scripts/plot_metrics.py --lora outputs/lora_dpo --out_dir outputs/plots_lora")
archive("lora_dpo_run_outputs.zip", ["outputs/lora_dpo", "outputs/generations", "outputs/plots_lora"])
run("du -sh outputs/*")
''',
    ]


STAGES = {
    "alignment-00-pilot": {"cells": cells_pilot, "sources": [], "extra": ' "trl>=0.19,<0.24" "peft>=0.15,<0.18"'},
    "alignment-01-sft": {"cells": cells_sft, "sources": [], "extra": ""},
    "alignment-02-dpo-hh": {"cells": lambda: cells_dpo("hh"), "sources": ["alignment-01-sft"], "extra": ""},
    "alignment-03-dpo-uf": {"cells": lambda: cells_dpo("uf"), "sources": ["alignment-01-sft"], "extra": ""},
    "alignment-04-lora-dpo": {"cells": cells_lora, "sources": [], "extra": ' "trl>=0.19,<0.24" "peft>=0.15,<0.18"'},
}


def notebook(cells):
    nb_cells = []
    for i, source in enumerate(cells):
        if i == 0:
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source.strip("\n")})
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--stages", nargs="*", default=list(STAGES))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for slug in args.stages:
        stage = STAGES[slug]
        setup = SETUP.format(repo=args.repo, commit=args.commit, work=WORK)
        install = INSTALL.format(extra=stage["extra"])
        cells = stage["cells"]()
        cells = [cells[0], setup, install, *cells[1:]]
        out = root / slug
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{slug}.ipynb").write_text(json.dumps(notebook(cells), indent=1))
        metadata = {
            "id": f"{args.username}/{slug}",
            "title": slug,
            "code_file": f"{slug}.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "machine_shape": args.accelerator,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [f"{args.username}/{s}" for s in stage["sources"]],
            "model_sources": [],
        }
        (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
