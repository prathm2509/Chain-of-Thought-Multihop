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
LONE_NAME_RE = re.compile(r"^\**\s*([A-Z][a-zA-Z]+)\s*\**\.?$")

# Version 2 (2026-11-19, after the 1.5B run): the 1.5B sometimes replied in
# direct mode with just the name ("Ronon") and no "Answer:" prefix. Version 1
# counted those as unreadable, i.e. wrong, which unfairly lowered direct
# accuracy. Version 2 also accepts a last line that is nothing but one name.
# It never looks at the gold answer while reading.
GRADER_VERSION = 2


def build_prompt(item: dict, mode: str) -> str:
    facts = "\n".join(item["facts"])
    return f"{INTRO}\n{facts}\n\nQuestion: {item['question']}\n{INSTRUCTION[mode]}"


def extract_answer(text: str, version: int = GRADER_VERSION) -> str | None:
    """The name on the last "Answer:" line; in version 2, failing that, a last
    line that is a single name. None if neither is there."""
    found = ANSWER_RE.findall(text)
    if found:
        return found[-1]
    if version >= 2:
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if lines and (m := LONE_NAME_RE.match(lines[-1])):
            return m.group(1)
    return None


def grade(text: str, gold: str, version: int = GRADER_VERSION) -> dict:
    pred = extract_answer(text, version)
    return {"pred": pred, "parsed": pred is not None, "correct": pred == gold}
