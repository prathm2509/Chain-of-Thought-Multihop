"""Synthetic family-tree worlds with multi-hop questions of a guaranteed depth.

A world is a few generations of invented people linked by three functional
relations: mother, father, spouse. Each relation gives at most one answer, so
every question has exactly one correct answer.

A k-hop question follows a chain of k relations, e.g. 3 hops:
    "Who is Ravo's mother's spouse's father?"
In strict mode a question is kept only if the answer can't be reached in fewer
than k hops. Without that filter, "Ravo's mother's spouse" is really just
"Ravo's father" (1 hop wearing a 2-hop costume).

Run:  python world.py --seed 0 --show 3
"""

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass, field

RELATIONS = ("mother", "father", "spouse")

ONSETS = "b d f g k l m n p r s t v z br dr gr kr pl st tr".split()
VOWELS = "a e i o u".split()
CODAS = ["", "", "n", "l", "r", "s"]


@dataclass
class Person:
    name: str
    sex: str  # "f" or "m"
    generation: int
    mother: str | None = None
    father: str | None = None
    spouse: str | None = None


@dataclass
class World:
    people: dict[str, Person] = field(default_factory=dict)

    def get(self, name: str, relation: str) -> str | None:
        return getattr(self.people[name], relation)

    def facts(self) -> list[tuple[str, str, str]]:
        """Every (person, relation, target) edge in the world."""
        return [
            (p.name, rel, getattr(p, rel))
            for p in self.people.values()
            for rel in RELATIONS
            if getattr(p, rel) is not None
        ]


def new_name(rng: random.Random, taken: set[str]) -> str:
    """A fresh invented two-syllable name, so a pretrained model has no priors about it."""
    while True:
        syl = lambda: rng.choice(ONSETS) + rng.choice(VOWELS)
        name = (syl() + syl() + rng.choice(CODAS)).capitalize()
        if name not in taken:
            return name


def generate_world(
    seed: int = 0,
    founder_couples: int = 6,
    generations: int = 4,
    max_children: int = 3,
    marriage_rate: float = 0.8,
) -> World:
    rng = random.Random(seed)
    world = World()

    def new_person(sex: str, gen: int) -> Person:
        p = Person(new_name(rng, world.people.keys()), sex, gen)
        world.people[p.name] = p
        return p

    def marry(a: Person, b: Person) -> None:
        a.spouse, b.spouse = b.name, a.name

    couples = []
    for _ in range(founder_couples):
        f, m = new_person("f", 0), new_person("m", 0)
        marry(f, m)
        couples.append((f, m))

    for gen in range(1, generations):
        # Each couple has children; remember which family each child came from.
        children = []
        for family, (mom, dad) in enumerate(couples):
            for _ in range(rng.randint(1, max_children)):
                c = new_person(rng.choice("fm"), gen)
                c.mother, c.father = mom.name, dad.name
                children.append((family, c))

        # Pair women and men from different families into the next generation's couples.
        # Anyone left without a partner may marry an outsider (a person with no recorded
        # parents), which keeps the population from dying out over many generations.
        rng.shuffle(children)
        unmatched_men = [x for x in children if x[1].sex == "m"]
        couples = []
        for fam, c in children:
            if c.spouse is not None or rng.random() > marriage_rate:
                continue
            if c.sex == "f":
                match = next((x for x in unmatched_men if x[0] != fam), None)
                if match:
                    unmatched_men.remove(match)
                    partner = match[1]
                else:
                    partner = new_person("m", gen)
                couple = (c, partner)
            else:
                if (fam, c) in unmatched_men:
                    unmatched_men.remove((fam, c))
                couple = (new_person("f", gen), c)
            marry(*couple)
            couples.append(couple)
        if not couples:
            break
    return world


def follow(world: World, start: str, path: tuple[str, ...]) -> list[str] | None:
    """People visited along `path`, starting with `start`. None if any hop is undefined."""
    chain = [start]
    for rel in path:
        nxt = world.get(chain[-1], rel)
        if nxt is None:
            return None
        chain.append(nxt)
    return chain


def distances(world: World, start: str) -> dict[str, int]:
    """Shortest number of relation hops from start to everyone reachable (BFS)."""
    dist, queue = {start: 0}, deque([start])
    while queue:
        name = queue.popleft()
        for rel in RELATIONS:
            nxt = world.get(name, rel)
            if nxt is not None and nxt not in dist:
                dist[nxt] = dist[name] + 1
                queue.append(nxt)
    return dist


def all_questions(world: World, max_k: int, strict: bool = True) -> list[dict]:
    """Every valid question up to max_k hops. Strict keeps only true k-hop questions."""
    out = []

    def extend(start: str, path: tuple[str, ...], chain: list[str], dist: dict[str, int]) -> None:
        if path:
            k, answer = len(path), chain[-1]
            if not strict or dist[answer] == k:
                out.append({"start": start, "path": list(path), "chain": chain,
                            "answer": answer, "hops": k})
        if len(path) < max_k:
            for rel in RELATIONS:
                nxt = world.get(chain[-1], rel)
                if nxt is not None:  # undefined hop: no longer path through here exists
                    extend(start, path + (rel,), chain + [nxt], dist)

    for name in world.people:
        extend(name, (), [name], distances(world, name))
    return out


# --- rendering -------------------------------------------------------------

def fact_text(person: str, relation: str, target: str) -> str:
    return f"{person}'s {relation} is {target}."


def question_text(q: dict) -> str:
    # Possessive form reads left to right in hop order: Ravo -> mother -> father.
    return f"Who is {q['start']}'s " + "'s ".join(q["path"]) + "?"


def cot_text(q: dict) -> str:
    """Gold reasoning chain: one stated fact per hop, then the answer."""
    steps = [fact_text(a, rel, b) for a, rel, b in zip(q["chain"], q["path"], q["chain"][1:])]
    return " ".join(steps) + f" So the answer is {q['answer']}."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--founders", type=int, default=6, help="founder couples")
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--max-k", type=int, default=5, help="longest question in hops")
    ap.add_argument("--loose", action="store_true", help="keep questions with shortcuts")
    ap.add_argument("--show", type=int, default=2, help="example questions per hop count")
    ap.add_argument("--out", help="write all questions to this .jsonl file")
    args = ap.parse_args()

    world = generate_world(args.seed, args.founders, args.generations)
    strict_qs = all_questions(world, args.max_k, strict=True)
    loose_qs = all_questions(world, args.max_k, strict=False)
    qs = loose_qs if args.loose else strict_qs

    gens = {}
    for p in world.people.values():
        gens[p.generation] = gens.get(p.generation, 0) + 1
    print(f"World: {len(world.people)} people, {len(world.facts())} facts, "
          f"per generation {dict(sorted(gens.items()))}\n")

    print("hops  true-k-hop  with-shortcuts")
    for k in range(1, args.max_k + 1):
        s = sum(q["hops"] == k for q in strict_qs)
        l = sum(q["hops"] == k for q in loose_qs)
        print(f"{k:>4}  {s:>10}  {l:>14}")

    rng = random.Random(args.seed)
    for k in range(1, args.max_k + 1):
        pool = [q for q in qs if q["hops"] == k]
        for q in rng.sample(pool, min(args.show, len(pool))):
            print(f"\n[{k} hop{'s' * (k > 1)}] {question_text(q)}")
            print(f"   CoT: {cot_text(q)}")

    if args.out:
        with open(args.out, "w") as f:
            for q in qs:
                f.write(json.dumps({**q, "question": question_text(q), "cot": cot_text(q)}) + "\n")
        print(f"\nWrote {len(qs)} questions to {args.out}")


if __name__ == "__main__":
    main()
