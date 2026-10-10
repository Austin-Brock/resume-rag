# resume-rag

A Retrieval-Augmented Generation (RAG) pipeline over a PDF, written from scratch
in ~280 lines of Python. No LangChain, no framework — every stage is a plain
function you can read.

```
PDF ──► extract_pages ──► chunk_text ──► embed ──► ChromaDB
                                                      │
question ──► embed ─────────────► similarity search ──┘
                                        │
                        chunks + question ──► LLM ──► answer
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then paste your real OpenAI key into .env
```

Put a PDF at `resume.pdf` (PDFs are gitignored, so yours stays local).

## Use

Build the vector store:

```bash
python main.py index resume.pdf
```

**Retrieval only — no LLM.** Run this first and read the chunks. If retrieval
is returning the wrong passages, no amount of prompting will fix the answer:

```bash
python main.py search "What databases has Austin worked with?"
```

Then the full pipeline:

```bash
python main.py ask "What experience does Austin have with PostgreSQL?"
python main.py ask "..." --show-chunks    # see the context it used
```

## Experiments

Each of these isolates one variable. Run them and read the difference.

| Command | What it shows |
|---|---|
| `python main.py index resume.pdf --chunk-size 1500 --overlap 200` | Bigger chunks retrieve more surrounding context but dilute the match — the embedding averages over more topics, so precision drops. |
| `python main.py search "..." --top-k 10` vs `--top-k 3` | More context isn't free. Watch the similarity scores fall off; low-ranked chunks are mostly noise the model has to ignore. |
| `python main.py ask "What is Austin's experience with Kubernetes?"` | Ask about something absent from the PDF. A correctly-grounded system says it doesn't know. This is the hallucination boundary. |
| `python main.py ask "..." --no-rag` | Same question, retrieval switched off. The difference between these two outputs *is* what RAG contributes. |
| `python main.py index resume.pdf other.pdf` | Retrieval across documents. Citations show which file each fact came from. |

## Notes on the implementation

- **Cosine distance, not L2.** Chroma defaults to squared L2; OpenAI embeddings
  are meant to be compared by cosine similarity. Set at collection creation in
  `get_collection()` via `{"hnsw:space": "cosine"}` — it cannot be changed
  afterward without rebuilding.
- **Pages are chunked separately** so each chunk keeps an accurate page number.
  That metadata is what makes `(resume.pdf, page 1)` citations possible.
- **`temperature=0`** for reproducibility. When comparing chunk sizes you want
  the only variable to be retrieval, not sampling.
- **Embeddings are batched** 100 at a time in `embed()`; the API preserves input
  order, so results line up with the input list.

## Cost

`text-embedding-3-small` is $0.02 per million tokens. A two-page resume is a few
thousand tokens, so indexing costs a fraction of a cent. `gpt-4o-mini` answers
cost about the same. The whole project runs for well under a penny.

## What's next

- **Hybrid search** — combine vector similarity with BM25 keyword matching.
  Semantic search is weak on exact identifiers (error codes, version numbers,
  proper nouns) because they carry little semantic signal.
- **Agentic RAG** — let the model decide whether it has enough context and issue
  follow-up searches before answering, instead of retrieving exactly once.
