import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world import RELATIONS, all_questions, cot_text, distances, follow, generate_world, question_text


class TestWorld(unittest.TestCase):
    def setUp(self):
        self.world = generate_world(seed=0, founder_couples=3, generations=6)
        self.strict = all_questions(self.world, max_k=6, strict=True)

    def test_same_seed_same_world(self):
        again = generate_world(seed=0, founder_couples=3, generations=6)
        self.assertEqual(self.world.facts(), again.facts())

    def test_relations_are_consistent(self):
        for p in self.world.people.values():
            if p.spouse:
                self.assertEqual(self.world.get(p.spouse, "spouse"), p.name)
            if p.mother:
                self.assertEqual(self.world.people[p.mother].sex, "f")
            if p.father:
                self.assertEqual(self.world.people[p.father].sex, "m")

    def test_answer_matches_path(self):
        for q in self.strict:
            self.assertEqual(follow(self.world, q["start"], tuple(q["path"]))[-1], q["answer"])

    def test_strict_questions_have_no_shortcut(self):
        for q in self.strict:
            self.assertEqual(distances(self.world, q["start"])[q["answer"]], q["hops"])

    def test_known_shortcut_is_filtered(self):
        # Your mother's spouse is your father: 1 hop, so it must never appear as a 2-hop question.
        self.assertFalse(any(q["path"] == ["mother", "spouse"] for q in self.strict))

    def test_cot_states_one_fact_per_hop(self):
        for q in self.strict[:50]:
            self.assertEqual(cot_text(q).count(" is "), q["hops"] + 1)  # hops + "the answer is"
            self.assertTrue(question_text(q).startswith(f"Who is {q['start']}'s "))

    def test_relations_cover_all_facts(self):
        self.assertTrue(all(rel in RELATIONS for _, rel, _ in self.world.facts()))


if __name__ == "__main__":
    unittest.main()
