# Chain-of-Thought Multihop

Does chain-of-thought help more when a question requires chaining more facts together? This experiment asks two Qwen2.5 models (1.5B and 3B) questions that need 1 to 4 chained lookups over facts given in the prompt, answered directly and step by step.

- [PLAN.md](PLAN.md): the design and the measures, fixed before running.
- [PREDICTION.md](PREDICTION.md): predictions, written before any results.

## Files

| File | What it does |
|---|---|
| `world.py` | Generates worlds of invented people and their links |
| `dataset.py` | Builds the 1,200 questions in `data/questions_v1.jsonl` |
| `prompts.py` | Prompt wording and answer grading |
| `run.py` | Asks a model every question, both ways (needs a GPU) |
| `analyze.py` | Accuracy, advantage and error bars per hop count |
| `tests/` | Checks the questions, the grader and the analysis (`python -m unittest discover -s tests`) |

## Running on Google Colab

Choose **Runtime → Change runtime type → T4 GPU**, then run these cells.

```python
# 1. Save results to Google Drive, so a disconnect loses nothing
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/cot-multihop
```

```python
# 2. Get the code
!git clone https://github.com/prathm2509/Chain-of-Thought-Multihop.git
%cd Chain-of-Thought-Multihop
!pip install -q transformers accelerate
```

```python
# 3. Pilot: 5 questions per hop count, about a minute
!python run.py --model Qwen/Qwen2.5-1.5B-Instruct --limit 5 --out /content/drive/MyDrive/cot-multihop/pilot.jsonl
!python analyze.py /content/drive/MyDrive/cot-multihop/pilot.jsonl --out /content/drive/MyDrive/cot-multihop/pilot_summary.json
```

```python
# 4. Full runs (rerun the same line after a disconnect; finished answers are skipped)
!python run.py --model Qwen/Qwen2.5-1.5B-Instruct --out /content/drive/MyDrive/cot-multihop/qwen2.5-1.5b.jsonl
!python run.py --model Qwen/Qwen2.5-3B-Instruct --out /content/drive/MyDrive/cot-multihop/qwen2.5-3b.jsonl
```

```python
# 5. Results
!python analyze.py /content/drive/MyDrive/cot-multihop/qwen2.5-*.jsonl --out /content/drive/MyDrive/cot-multihop/summary.json
```
