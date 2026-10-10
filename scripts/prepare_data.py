"""Download and preprocess SFT / DPO data into JSONL files.

    python scripts/prepare_data.py --out_dir outputs/data --which sft hh uf

Writes ``sft_{train,val}.jsonl``, ``dpo_hh_{train,val}.jsonl``,
``dpo_uf_{train,val}.jsonl`` and ``data_manifest.json`` (counts, filtering
stats, and the list of example ids for reproducibility).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignment.data import prepare_dpo, prepare_sft, read_jsonl
from alignment.prompting import prepare_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/data")
    parser.add_argument("--which", nargs="+", choices=["sft", "hh", "uf"], default=["sft", "hh", "uf"])
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft_train", type=int, default=30000)
    parser.add_argument("--sft_val", type=int, default=1000)
    parser.add_argument("--dpo_train", type=int, default=15000)
    parser.add_argument("--dpo_val", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = prepare_tokenizer(AutoTokenizer.from_pretrained(args.tokenizer))
    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["settings"] = vars(args)

    if "sft" in args.which:
        manifest["sft"] = prepare_sft(tokenizer, out_dir, args.sft_train, args.sft_val, args.max_length, args.seed)
    for name in ("hh", "uf"):
        if name in args.which:
            manifest[f"dpo_{name}"] = prepare_dpo(
                name, tokenizer, out_dir, args.dpo_train, args.dpo_val, args.max_length, args.max_prompt_length, args.seed
            )

    for path in sorted(out_dir.glob("*.jsonl")):
        manifest.setdefault("files", {})[path.name] = [row["id"] for row in read_jsonl(path)]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=1))


if __name__ == "__main__":
    main()
