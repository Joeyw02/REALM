# REALM: Recursive Relevance Modeling for LLM-based Document Re-Ranking

Code for the EMNLP 2025 paper **“REALM: Recursive Relevance Modeling for LLM-based Document Re-Ranking.”**

---

## Repository Layout

~~~text
.
├── src/
│   ├── models.py          # Flan-T5 / GPT wrappers (tokenization, logits, truncation)
│   ├── algorithm.py       # REALM core: realm(), rating updates, recursive sort, utils
│   ├── main.py            # CLI: load data, run REALM, evaluate NDCG@10 (pytrec_eval)
│   └── run.sh             # convenience launcher (calls main.py)
├── data/
│   ├── retrieve_results_dl19.json
│   ├── qrels_dl19.json
│   ├── retrieve_results_dl20.json
│   ├── qrels_dl20.json
│   └── ....
└── requirements.txt
~~~

---

## Installation

**Python:** 3.9–3.11 recommended.

1) Create a fresh environment.

2) Install the dependencies:

~~~bash
pip install -r requirements.txt
~~~

---

## Models

- **Open-source (local):** `google/flan-t5-large|xl|xxl` (auto-downloaded by 🤗 Transformers).
- **API option:** `gpt-5` (via OpenAI). Set your key:

~~~bash
export OPENAI_API_KEY=sk-...
~~~

**Offline hint:** Pre-download models to a local directory and pass that path to `--model`.

---

## Data Format

- **Retrieve results** (`retrieve_results_*.json`): list of queries and candidate hits

~~~json
[
  {
    "query": "what is llm-based re-ranking?",
    "hits": [
      {"qid": 123, "docid": "D1", "content": "passage text ..."},
      {"qid": 123, "docid": "D2", "content": "another passage ..."}
    ]
  }
]
~~~

- **Qrels** (`qrels_*.json`): relevance labels per query id

~~~json
{
  "123": { "D1": 3, "D9": 2, "D2": 0 }
}
~~~

`main.py` attaches labels when present and evaluates **NDCG@10** using `pytrec_eval`.

---

## Quickstart

### A) Using `run.sh` (recommended)

From the repo root:

~~~bash
cd src
bash run.sh
~~~

Example content of `run.sh`:

~~~bash
cd src
python main.py --dataset dl19 --model google/flan-t5-xl --order bm25
~~~

### B) Direct CLI

~~~bash
cd src
python main.py \
  --dataset dl19 \
  --model google/flan-t5-large \
  --order bm25
~~~

**Arguments**

- `--dataset, -d`: dataset name (e.g., `dl19`, `dl20`)
- `--model, -m`: `google/flan-t5-large|xl|xxl` or `gpt-5`
- `--order, -o`: initial order `bm25`, `random`, or `inverse`

**Output Example**

~~~text
NDCG@10: 0.705

Inference Count: 80.4
Tokens in Prompt: 25823.0
Latency (s): 4.0
Depth: 2.3
~~~

---