"""Build the question set: k-hop questions over facts given in the prompt.

Each question gets its own freshly generated world. The prompt shows exactly
N_FACTS facts: the k facts of the answer chain plus distractor facts from the
same world, shuffled. Holding the fact count fixed means a 4-hop prompt is no
longer than a 1-hop prompt, so "more hops" is not confused with "more text".

A question is kept only if, using the facts shown in its prompt, the answer
cannot be reached in fewer than k lookups. Spouse counts in both directions
for that check, since "A's spouse is B" also tells you B's spouse.

Run:  python dataset.py            (writes data/questions_v1.jsonl)
      python dataset.py --show 2   (also prints examples per hop count)
"""

import argparse
import json
import random
from collections import deque
from pathlib import Path

from world import RELATIONS, generate_world, question_text

VERSION = "v1"
HOPS = (1, 2, 3, 4)
PER_HOP = 300
N_FACTS = 24
BASE_SEED = 20261119
WORLD_KW = dict(founder_couples=4, generations=4, sideways_rate=0.8)


def fact_sentence(a: str, rel: str, b: str) -> str:
    return f"{a}'s {rel} is {b}."


def shortest_hops(facts: list[tuple[str, str, str]], start: str, target: str) -> int:
    """Fewest lookups from start to target using only the given facts."""
    edges: dict[str, set[str]] = {}
    for a, rel, b in facts:
        edges.setdefault(a, set()).add(b)
        if rel == "spouse":
            edges.setdefault(b, set()).add(a)
    dist, queue = {start: 0}, deque([start])
    while queue:
        cur = queue.popleft()
        if cur == target:
            return dist[cur]
        for nxt in edges.get(cur, ()):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return -1


def random_chain(world, rng: random.Random, hops: int):
    """A random walk of `hops` defined links that never revisits a person."""
    start = rng.choice(sorted(world.people))
    chain, path = [start], []
    for _ in range(hops):
        options = [rel for rel in RELATIONS
                   if (nxt := world.get(chain[-1], rel)) is not None and nxt not in chain]
        if not options:
            return None
        rel = rng.choice(options)
        path.append(rel)
        chain.append(world.get(chain[-1], rel))
    return chain, path


def make_item(rng: random.Random, hops: int, idx: int, max_tries: int = 500) -> dict:
    for _ in range(max_tries):
        world = generate_world(seed=rng.randrange(2**31), **WORLD_KW)
        walk = random_chain(world, rng, hops)
        if walk is None:
            continue
        chain, path = walk
        chain_facts = list(zip(chain, path, chain[1:]))

        # Distractors: other facts from the same world. A spouse fact is stored
        # both ways in the world, so keep one direction per couple, and never
        # repeat a chain fact in reverse.
        seen = {frozenset((a, b)) for a, rel, b in chain_facts if rel == "spouse"}
        pool = []
        for a, rel, b in world.facts():
            if (a, rel, b) in chain_facts:
                continue
            if rel == "spouse":
                key = frozenset((a, b))
                if key in seen:
                    continue
                seen.add(key)
            pool.append((a, rel, b))
        if len(pool) < N_FACTS - hops:
            continue
        rng.shuffle(pool)
        shown = chain_facts + pool[: N_FACTS - hops]

        if shortest_hops(shown, chain[0], chain[-1]) != hops:
            continue  # a shortcut exists among the shown facts
        rng.shuffle(shown)
        q = {"start": chain[0], "path": path}
        return {
            "id": f"h{hops}-{idx:03d}",
            "hops": hops,
            "question": question_text(q),
            "facts": [fact_sentence(*f) for f in shown],
            "path": path,
            "chain": chain,
            "answer": chain[-1],
        }
    raise RuntimeError(f"could not build a {hops}-hop question after {max_tries} tries")


def build(per_hop: int = PER_HOP) -> list[dict]:
    items = []
    for hops in HOPS:
        # A separate seed per hop count, so changing one set never shifts another.
        rng = random.Random(BASE_SEED + hops)
        items += [make_item(rng, hops, i) for i in range(per_hop)]
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-hop", type=int, default=PER_HOP)
    ap.add_argument("--out", default=f"data/questions_{VERSION}.jsonl")
    ap.add_argument("--show", type=int, default=0, help="print this many examples per hop count")
    args = ap.parse_args()

    items = build(args.per_hop)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"Wrote {len(items)} questions to {out}")

    for hops in HOPS:
        subset = [it for it in items if it["hops"] == hops]
        rels = {}
        for it in subset:
            for r in it["path"]:
                rels[r] = rels.get(r, 0) + 1
        print(f"  {hops} hop{'s' if hops > 1 else ' '}: {len(subset)} questions, links used: {dict(sorted(rels.items()))}")
        for it in subset[: args.show]:
            print(f"    {it['question']}  ->  {it['answer']}   (chain: {' -> '.join(it['chain'])})")


if __name__ == "__main__":
    main()
