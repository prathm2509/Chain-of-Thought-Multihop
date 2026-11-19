"""Ask one model every question, directly and step by step. Runs on a GPU (Colab).

Results are appended to a JSONL file after every batch, and a rerun skips
anything already answered, so a disconnect loses at most one batch.

Pilot:  python run.py --model Qwen/Qwen2.5-1.5B-Instruct --limit 5 --out results/pilot.jsonl
Full:   python run.py --model Qwen/Qwen2.5-1.5B-Instruct --out results/qwen2.5-1.5b.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from prompts import build_prompt, grade

MODES = ("direct", "cot")


def load_items(path: str, limit: int | None) -> list[dict]:
    items = [json.loads(line) for line in open(path, encoding="utf-8")]
    if limit:  # take the first `limit` questions of each hop count
        by_hop: dict[int, list] = {}
        for it in items:
            by_hop.setdefault(it["hops"], []).append(it)
        items = [it for group in by_hop.values() for it in group[:limit]]
    return items


def done_keys(out: Path) -> set[tuple[str, str]]:
    if not out.exists():
        return set()
    keys = set()
    for line in open(out, encoding="utf-8"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # a half-written last line from a crash
        keys.add((rec["mode"], rec["id"]))
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/questions_v1.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="questions per hop count (for a pilot)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256, help="same limit for both modes")
    ap.add_argument("--dtype", default="float16", help="float16 on a T4; bfloat16 on newer GPUs")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto")
    model.eval()
    cfg = model.config
    print(f"{args.model}: {cfg.num_hidden_layers} layers, width {cfg.hidden_size}, dtype {args.dtype}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    skip = done_keys(out)
    items = load_items(args.data, args.limit)
    todo = [(mode, it) for mode in MODES for it in items if (mode, it["id"]) not in skip]
    print(f"{len(todo)} answers to generate ({len(skip)} already done)")

    t_start = time.time()
    for b in range(0, len(todo), args.batch_size):
        batch = todo[b: b + args.batch_size]
        prompts = [
            tok.apply_chat_template([{"role": "user", "content": build_prompt(it, mode)}],
                                    tokenize=False, add_generation_prompt=True)
            for mode, it in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        with open(out, "a", encoding="utf-8") as f:
            for (mode, it), ids in zip(batch, new):
                ids = ids[ids != tok.pad_token_id]
                text = tok.decode(ids, skip_special_tokens=True)
                rec = {
                    "model": args.model, "layers": cfg.num_hidden_layers, "width": cfg.hidden_size,
                    "mode": mode, "id": it["id"], "hops": it["hops"], "answer": it["answer"],
                    "output": text, "n_tokens": int(len(ids)),
                    "hit_limit": int(len(ids)) >= args.max_new_tokens,
                    **grade(text, it["answer"]),
                }
                f.write(json.dumps(rec) + "\n")
        n = b + len(batch)
        rate = n / (time.time() - t_start)
        print(f"  {n}/{len(todo)}  ({rate:.1f} answers/s, ~{(len(todo) - n) / rate / 60:.0f} min left)")
    print(f"Done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
