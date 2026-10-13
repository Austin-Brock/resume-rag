# resume-rag

A Retrieval-Augmented Generation (RAG) pipeline over a PDF, written from scratch
in ~300 lines of Python. No LangChain, no framework — every stage is a plain
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
| `python main.py search "..." --top-k 10` vs `--top-k 3` | How well retrieval actually ranks. On a document this short, worse than you'd expect — see below. |
| `python main.py ask "What is Austin's experience with Kubernetes?"` | Ask about something absent from the PDF. A correctly-grounded system says it doesn't know. This is the hallucination boundary. |
| `python main.py ask "..." --no-rag` | Same question, retrieval switched off. The difference between these two outputs *is* what RAG contributes — see below. |
| `python main.py index resume.pdf other.pdf` | Retrieval across documents. Citations show which file each fact came from. |

### What `--no-rag` actually shows

The same question, asked twice. With retrieval:

```
$ python main.py ask "What experience does Austin have with PostgreSQL?"

Austin has experience with PostgreSQL in several ways: he secured multi-tenant
application data using PostgreSQL Row-Level Security, improved database
reliability and deployment safety by implementing SQLAlchemy transaction
management and connection pooling, and automated application testing and
deployment by containerizing PostgreSQL with Docker Compose
(resume.pdf, page 2).
```

Without:

```
$ python main.py ask "What experience does Austin have with PostgreSQL?" --no-rag

Austin has extensive experience with PostgreSQL, having worked with it in
various projects over the past few years. He has utilized PostgreSQL for
database design, optimization, and management... His experience includes
writing complex SQL queries, creating and managing database schemas, and
implementing indexing strategies to improve performance. Additionally, Austin
has experience with PostgreSQL's advanced features, such as JSONB for handling
semi-structured data and using stored procedures for business logic.
```

Every specific in the second answer is invented. JSONB, stored procedures,
indexing strategies, "the past few years" — none of it appears in the PDF, and
the model has no idea who Austin is. It is fluent, confident, and structurally
indistinguishable from the real answer above. That is the failure RAG prevents,
and it is worth seeing once in a case where you know the ground truth, because
in production you won't.

Note that `--no-rag` also swaps the system prompt (see Implementation notes).
Dropping the context while keeping the "answer only from context" instruction
just produces a refusal, which demonstrates nothing.

### What `--top-k` actually shows

Not what you'd hope. The full ranking for a question the PDF clearly answers:

```
$ python main.py search "What databases has Austin worked with?" --top-k 10

1.  page 1  similarity 0.524    <- contact-info header
2.  page 2  similarity 0.461
3.  page 2  similarity 0.431    <- the actual PostgreSQL bullet
4.  page 1  similarity 0.427
5.  page 1  similarity 0.422
6.  page 2  similarity 0.414
7.  page 1  similarity 0.392
8.  page 1  similarity 0.391
9.  page 1  similarity 0.387
10. page 2  similarity 0.369
```

Two things are wrong here, and neither is the one the tutorials warn about.
The scores don't fall off a cliff — they drift down 0.52 to 0.37 with no gap
anywhere, so there is no natural cutoff and a similarity threshold would be
arbitrary. And the top-ranked chunk is the name-and-phone-number header, while
the chunk that actually answers the question ranks third.

The cause is the document. Split a dense two-page resume into 500-character
windows and every chunk looks about equally "resume-shaped" to an embedding
model: same vocabulary, same register, no chunk about one clean topic. Semantic
similarity needs topical variation to rank against, and a resume barely has
any. It still works here only because `--top-k 3` is wide enough to catch the
right chunk at position three and the LLM ignores the other two.

This is the strongest argument for the hybrid search in *What's next* — "databases"
and "PostgreSQL" are exactly the kind of near-keyword match BM25 nails and
cosine similarity smears.

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
- **Two system prompts, deliberately.** `SYSTEM_PROMPT` forbids answering from
  anything but the retrieved context. `--no-rag` switches to
  `UNGROUNDED_SYSTEM_PROMPT`, which drops that instruction, because retrieval
  and the grounding instruction are two separate variables — changing both at
  once measures nothing, and leaving the guardrail on with no context just
  yields "I don't have enough information."

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
