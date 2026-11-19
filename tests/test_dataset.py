import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import summarize
from dataset import N_FACTS, build, make_item, shortest_hops
from prompts import build_prompt, extract_answer, grade
from world import REAL_WORDS


def parse_facts(sentences):
    out = []
    for s in sentences:
        left, b = s.rstrip(".").split(" is ")
        a, rel = left.split("'s ")
        out.append((a, rel, b))
    return out


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = build(per_hop=40)

    def test_same_fact_count_at_every_hop(self):
        self.assertTrue(all(len(it["facts"]) == N_FACTS for it in self.items))

    def test_answer_needs_exactly_k_lookups(self):
        for it in self.items:
            facts = parse_facts(it["facts"])
            self.assertEqual(shortest_hops(facts, it["chain"][0], it["answer"]), it["hops"], it["id"])

    def test_chain_is_in_the_prompt(self):
        for it in self.items:
            for a, rel, b in zip(it["chain"], it["path"], it["chain"][1:]):
                self.assertIn(f"{a}'s {rel} is {b}.", it["facts"])

    def test_each_person_has_one_link_of_each_kind(self):
        for it in self.items:
            seen = {}
            for a, rel, b in parse_facts(it["facts"]):
                self.assertEqual(seen.setdefault((a, rel), b), b, it["id"])

    def test_deterministic(self):
        a = make_item(random.Random(5), 3, 0)
        b = make_item(random.Random(5), 3, 0)
        self.assertEqual(a, b)

    def test_no_real_words_as_names(self):
        for it in self.items:
            for name in it["chain"]:
                self.assertNotIn(name.lower(), REAL_WORDS)


class TestGrading(unittest.TestCase):
    def test_takes_the_last_answer_line(self):
        text = "Ravo's mother is Bokon.\nMaybe Answer: Bokon\nBokon's boss is Tilen.\nAnswer: Tilen"
        self.assertEqual(extract_answer(text), "Tilen")

    def test_name_in_reasoning_does_not_count(self):
        # The old experiment's grader credited this; this one must not.
        self.assertFalse(grade("Tuma is involved, but\nAnswer: Plefi", "Tuma")["correct"])

    def test_bold_and_missing(self):
        self.assertEqual(extract_answer("Answer: **Tuma**"), "Tuma")
        self.assertEqual(grade("I think it is Tuma.", "Tuma"), {"pred": None, "parsed": False, "correct": False})

    def test_prompt_contains_facts_and_question(self):
        it = self.items_one = make_item(random.Random(1), 2, 0)
        p = build_prompt(it, "direct")
        self.assertIn(it["question"], p)
        self.assertTrue(all(f in p for f in it["facts"]))
        self.assertIn("step by step", build_prompt(it, "cot"))


class TestAnalysis(unittest.TestCase):
    def test_advantage_and_growth(self):
        data = {"m": {}}
        # 1 hop: both modes always right. 3 hops: direct always wrong, step by step right.
        for hops, d, c in [(1, True, True), (3, False, True)]:
            data["m"][hops] = {f"q{i}": {"direct": {"mode": "direct", "correct": d, "parsed": True, "hit_limit": False, "n_tokens": 3},
                                         "cot": {"mode": "cot", "correct": c, "parsed": True, "hit_limit": False, "n_tokens": 50}}
                               for i in range(20)}
        s = summarize(data)["m"]
        self.assertEqual(s["by_hops"][1]["advantage"], 0)
        self.assertEqual(s["by_hops"][3]["advantage"], 1)
        self.assertTrue(s["growth"]["real"])


if __name__ == "__main__":
    unittest.main()
