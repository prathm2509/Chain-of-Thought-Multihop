"""Summarize results: the step-by-step advantage at each hop count, with error bars.

Measures (fixed before any results were seen; see PLAN.md):
  advantage(k)   = step-by-step accuracy - direct accuracy, on the k-hop questions
  error bars     = 95% interval from 5,000 bootstrap resamples of the questions
                   (paired: a resampled question keeps both of its answers)
  growth test    = advantage(largest k) - advantage(1); "real" if its interval excludes 0
  backup check   = share of direct's mistakes that step by step fixed

Run:  python analyze.py results/qwen2.5-1.5b.jsonl results/qwen2.5-3b.jsonl
"""

import argparse
import json
import random
from pathlib import Path

N_BOOT = 5000
SEED = 0


def load(paths: list[str]) -> dict:
    """{model: {hops: {question id: {mode: record}}}}"""
    data: dict = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            data.setdefault(r["model"], {}).setdefault(r["hops"], {}).setdefault(r["id"], {})[r["mode"]] = r
    return data


def paired(questions: dict) -> list[tuple[int, int]]:
    """(direct correct, step-by-step correct) for questions answered both ways."""
    return [(int(q["direct"]["correct"]), int(q["cot"]["correct"]))
            for q in questions.values() if "direct" in q and "cot" in q]


def advantage(pairs: list[tuple[int, int]]) -> float:
    return sum(c - d for d, c in pairs) / len(pairs)


def interval(values: list[float]) -> tuple[float, float]:
    v = sorted(values)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


def boot(pairs, rng, stat=advantage):
    return [stat([rng.choice(pairs) for _ in pairs]) for _ in range(N_BOOT)]


def summarize(data: dict) -> dict:
    rng = random.Random(SEED)
    summary = {}
    for model, by_hop in sorted(data.items()):
        rows = {}
        resamples = {}
        for hops in sorted(by_hop):
            qs = by_hop[hops]
            pairs = paired(qs)
            if not pairs:
                continue
            n = len(pairs)
            acc_d = sum(d for d, _ in pairs) / n
            acc_c = sum(c for _, c in pairs) / n
            fixed = sum(1 for d, c in pairs if not d and c)
            broke = sum(1 for d, c in pairs if d and not c)
            resamples[hops] = (pairs, boot(pairs, rng))
            recs = [r for q in qs.values() for r in q.values()]
            def count(mode, key):
                return sum(1 for r in recs if r["mode"] == mode and r[key])
            def mean_tokens(mode):
                t = [r["n_tokens"] for r in recs if r["mode"] == mode]
                return round(sum(t) / len(t), 1) if t else None
            rows[hops] = {
                "n": n,
                "direct_acc": acc_d,
                "cot_acc": acc_c,
                "advantage": acc_c - acc_d,
                "advantage_ci": interval(resamples[hops][1]),
                "fixed": fixed,
                "broke": broke,
                "share_of_errors_removed": (fixed - broke) / (n - sum(d for d, _ in pairs))
                if n > sum(d for d, _ in pairs) else None,
                "unparsed": {"direct": n - count("direct", "parsed"), "cot": n - count("cot", "parsed")},
                "hit_token_limit": {"direct": count("direct", "hit_limit"), "cot": count("cot", "hit_limit")},
                "mean_tokens": {"direct": mean_tokens("direct"), "cot": mean_tokens("cot")},
            }
        growth = None
        if len(resamples) >= 2:
            lo, hi = min(resamples), max(resamples)
            diffs = [b - a for a, b in zip(resamples[lo][1], resamples[hi][1])]
            point = rows[hi]["advantage"] - rows[lo]["advantage"]
            ci = interval(diffs)
            growth = {"from_hops": lo, "to_hops": hi, "difference": point, "ci": ci,
                      "real": ci[0] > 0 or ci[1] < 0}
        summary[model] = {"by_hops": rows, "growth": growth}
    return summary


def pct(x):
    return "  n/a" if x is None else f"{100 * x:5.1f}"


def print_table(summary: dict) -> None:
    for model, s in summary.items():
        print(f"\n{model}")
        print(" hops | direct | step by step | advantage (95% interval) | fixed/broke | unreadable d/s")
        for hops, r in s["by_hops"].items():
            lo, hi = r["advantage_ci"]
            print(f"  {hops:>3} | {pct(r['direct_acc'])}% | {pct(r['cot_acc'])}%       |"
                  f" {pct(r['advantage'])} pts ({pct(lo)} to {pct(hi)}) |"
                  f" {r['fixed']:>4}/{r['broke']:<4}  | {r['unparsed']['direct']}/{r['unparsed']['cot']}")
        g = s["growth"]
        if g:
            verdict = "real" if g["real"] else "not distinguishable from zero"
            print(f"  Growth from {g['from_hops']} to {g['to_hops']} hops: {pct(g['difference'])} pts "
                  f"({pct(g['ci'][0])} to {pct(g['ci'][1])}) -> {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default="results/summary.json")
    args = ap.parse_args()
    summary = summarize(load(args.results))
    print_table(summary)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
