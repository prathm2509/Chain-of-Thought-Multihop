# Plan (written 2026-11-19, before any results)

## Question

Does step-by-step reasoning (chain-of-thought) help **more** as a question requires chaining **more facts**? And does a deeper model need it later than a shallower one?

## Models

| Model | Layers | Width | Precision |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 28 | 1536 | 16-bit |
| Qwen2.5-3B-Instruct | 36 | 2048 | 16-bit |

The 3B is both deeper and wider, so a difference between them cannot be pinned on depth alone. Qwen2.5-7B-Instruct (28 layers, width 3584) is a possible later check: it has the same depth as the 1.5B but is much wider.

## Questions (`data/questions_v1.jsonl`, built by `dataset.py`)

- 300 questions each at 1, 2, 3 and 4 hops (1,200 in total).
- Each question has its own world of invented people. The prompt lists exactly 24 facts: the chain the answer needs, plus distractor facts from the same world, shuffled.
- Links: mother, father, spouse, boss, mentor, doctor. Each person has at most one of each, so every question has exactly one answer.
- A question is kept only if its answer cannot be reached in fewer lookups using the facts in the prompt.
- Names that are real English words are excluded.

## Answering

Every question is asked twice, using each model's chat format:
- **Direct:** reply with only `Answer: <name>`.
- **Step by step:** write each fact used, one per line, then `Answer: <name>`.

Same limit for both (256 new tokens). Greedy decoding: the most likely next token, every time.

## Grading

Exact match of the name on the last `Answer:` line. Answers with no readable `Answer:` line count as wrong, and are reported separately, as are answers that hit the token limit.

## Measures, fixed now

- **Main:** advantage = step-by-step accuracy minus direct accuracy, at each hop count, per model.
- **Error bars:** 95% intervals from 5,000 paired bootstrap resamples of the questions.
- **Does it grow?** The difference between the 4-hop and the 1-hop advantage. Counted as real only if its 95% interval excludes 0.
- **Backup check:** share of direct's mistakes that step by step fixed, to guard against floor and ceiling effects.

## Changes after seeing results

- **2026-11-19, grader version 2.** In the 1.5B run, direct mode sometimes replied with just the name ("Ronon") instead of "Answer: Ronon". Version 1 counted those as unreadable, so wrong, which lowered direct accuracy and inflated the advantage. Version 2 also accepts a last line consisting of a single name. It does not look at the correct answer while reading. Both versions are reported (`python analyze.py ... --grader 1`).

## Order

1. Pilot: 5 questions per hop count on the 1.5B (checks timing and that answers parse).
2. Full run on the 1.5B, then the 3B.
3. `python analyze.py results/*.jsonl`.
