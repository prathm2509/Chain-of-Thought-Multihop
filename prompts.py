"""Prompt wording and answer grading, shared by the run and analysis scripts.

Both modes ask for the same final line, "Answer: <name>", so the same grader
reads both. Grading never looks inside the reasoning: only the name on the
last "Answer:" line counts, which avoids crediting a model that mentions the
right person mid-chain but concludes with someone else.
"""

import re

INTRO = "Here are some facts about a group of people:"

INSTRUCTION = {
    "direct": (
        "Reply with only the final answer, in the form \"Answer: <name>\". "
        "Do not explain."
    ),
    "cot": (
        "Think step by step: write down each fact you use, one per line. "
        "Then give the final answer on the last line, in the form \"Answer: <name>\"."
    ),
}

ANSWER_RE = re.compile(r"Answer:\s*\**\s*([A-Z][a-zA-Z]+)")


def build_prompt(item: dict, mode: str) -> str:
    facts = "\n".join(item["facts"])
    return f"{INTRO}\n{facts}\n\nQuestion: {item['question']}\n{INSTRUCTION[mode]}"


def extract_answer(text: str) -> str | None:
    """The name on the last "Answer:" line, or None if there is none."""
    found = ANSWER_RE.findall(text)
    return found[-1] if found else None


def grade(text: str, gold: str) -> dict:
    pred = extract_answer(text)
    return {"pred": pred, "parsed": pred is not None, "correct": pred == gold}
