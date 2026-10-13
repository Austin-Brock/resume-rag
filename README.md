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

This is the strongest argument for the hybrid search in *Productionizing this*
— "databases" and "PostgreSQL" are exactly the kind of near-keyword match BM25
nails and cosine similarity smears.

## Why Chroma

A vector database is not strictly required at this scale. With 37 chunks you
could hold the vectors in a NumPy array and brute-force cosine similarity in
about ten lines — and it would be *faster* than an HNSW index, because
approximate nearest-neighbour search only starts paying for itself in the
thousands. Chroma is here for the concepts, not the speed:

- **It matches the project's premise.** The selling point of this repo is that
  every stage is a plain function you can read. A vector store needing Docker, a
  running service and a connection string would move the reader's attention onto
  infrastructure and away from the five stages. `pip install chromadb` plus a
  directory path keeps it where it belongs.
- **It's a real vector database, not a stand-in.** Brute-force NumPy
  demonstrates linear algebra. Chroma demonstrates retrieval: a collection, a
  configurable distance metric, metadata you can filter on, an ANN index, and
  persistence. Those are the ideas that transfer. The scale here just happens to
  be small enough to inspect byte by byte.
- **The embeddings have to outlive the process.** `index`, `search` and `ask`
  are three separate runs. Embedding is the slow step and the only one that
  costs money, so it cannot be redone per query.

### How it's wired

Chroma runs **in-process**. It is a library imported at `main.py:21`, not a
server — no daemon, nothing listening on a port, and no database running once
`main.py` exits. `PersistentClient` reads and writes files under `.chroma/`
(gitignored; rebuild it with `index`):

| Path | What it holds |
|---|---|
| `.chroma/chroma.sqlite3` | chunk text, metadata, the collection registry |
| `.chroma/<segment-id>/*.bin` | the HNSW graph — the vectors themselves |

The split follows the access pattern. Fetching a chunk by id and filtering on
`source` or `page` are relational operations, so they live in SQLite.
Nearest-neighbour search across 1536 dimensions is not something SQL does, so
vectors live in a purpose-built index. A single `collection.query()` touches
both: the index returns ids and distances, SQLite hydrates text and metadata,
and `retrieve()` consumes all three.

### When to use something else

Chroma is the right choice for a local, single-user, read-mostly store and the
wrong one past that. The limits worth naming: one writer, no network access
control, no replication, and an index that lives or dies with a directory on one
machine.

The upgrade path depends on where the data already is. If the application has a
Postgres database, **pgvector** is usually the answer: one extension, vectors as
a real column type, and no second datastore to keep in sync. *Productionizing
this* has the schema and the query. If retrieval is heavy enough to deserve its
own service, a dedicated engine (Qdrant, Weaviate, Pinecone) buys sharding,
filtering at scale, and operational isolation.

Either way the blast radius is small: only `get_collection()` and `retrieve()`
know a vector database exists. Nothing else in `main.py` would change.

## Notes on the implementation

- **No framework, on purpose.** LangChain would compress this to about twenty
  lines — and hide every decision on this list behind a default. The chunking
  strategy, the distance metric, the prompt split and the retrieval cutoff are
  the things actually worth understanding, so they are written out where you can
  read and change them.
- **500-character chunks, 100 of overlap.** Roughly 125 tokens: small enough
  that a chunk is usually about one thing, large enough to hold a complete
  resume bullet. The 20% overlap keeps a bullet that straddles a boundary
  retrievable from at least one chunk. Both are flags rather than constants
  because the right value depends on the document — and on the evidence in
  *What `--top-k` actually shows*, 500 is probably too small for a resume
  specifically. See *Productionizing this*.
- **Deliberately small models.** `text-embedding-3-small` over `-large`: half
  the dimensions, a fraction of the price, and across 37 chunks there is little
  room for the larger model's extra precision to change which chunk ranks first.
  `gpt-4o-mini` because the generation step is extraction and citation from
  supplied text, not reasoning — model capability should track task difficulty,
  not ambition.
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
- **The PDF and the store are both gitignored**, for different reasons.
  `resume.pdf` is a personal document and has no business in a public repo.
  `.chroma/` is a build artifact derived from it — binary, churning, and
  rebuilt by one `index` command. Neither is source.

## Cost

`text-embedding-3-small` is $0.02 per million tokens. A two-page resume is a few
thousand tokens, so indexing costs a fraction of a cent. `gpt-4o-mini` answers
cost about the same. The whole project runs for well under a penny.

## Productionizing this

What this is: a single-user CLI, over one document, on one machine, with the
happy path hand-verified. What it is not: a service. The gap is mostly not the
RAG itself — the five stages would survive largely intact — it is the
measurement and the operations around them. In the order I would actually do
them:

**1. An eval set, before touching anything else.** Every claim in *Experiments*
above was judged by reading output, which is why the `--top-k` finding turned up
by accident instead of by measurement. The fix is unglamorous: ~30 questions
paired with the source pages that should answer them, scored for recall@k on
retrieval and faithfulness on generation, run in CI on every change. Until that
exists, "improving retrieval" is guesswork. It is listed first because every
item below is unverifiable without it.

**2. Chunking that respects the document.** Fixed 500-character windows cut
mid-sentence and mid-bullet, and produce the flat similarity curve documented
above. Splitting on resume structure instead — `PROFESSIONAL EXPERIENCE`,
`EDUCATION`, one chunk per role or per bullet group — should give each chunk a
single topic for the embedding to represent. With step 1 in place this becomes a
measurable claim rather than a plausible one.

**3. Hybrid retrieval.** Dense vectors plus BM25 keyword matching, combined with
reciprocal rank fusion. Embeddings are weak on exact tokens — version numbers,
error codes, proper nouns, `PostgreSQL` — because those carry little semantic
signal, which is exactly the failure visible in the `--top-k` output.

**4. pgvector in place of Chroma.** Chroma is right for a local demo (see *Why
Chroma*) and wrong for a service: one writer, no access control, no replication.
If the application already has Postgres, the whole vector store becomes a table:

```sql
CREATE EXTENSION vector;

CREATE TABLE chunks (
  id        text PRIMARY KEY,
  source    text NOT NULL,
  page      int  NOT NULL,
  text      text NOT NULL,
  embedding vector(1536)
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
```

Retrieval becomes one statement — `<=>` is cosine distance, matching the metric
Chroma is configured with:

```sql
SELECT text, source, page, 1 - (embedding <=> %(q)s) AS similarity
FROM chunks
ORDER BY embedding <=> %(q)s
LIMIT %(k)s;
```

That is the entire port: `get_collection()` and `retrieve()` are rewritten,
roughly thirty lines, and nothing else in `main.py` changes. The payoff is not
speed — it is one datastore instead of two, vector search that can `JOIN` and
`WHERE` against application tables, and indexing that happens in a transaction
so a failed run cannot leave a half-populated index behind.

**5. Indexing that does not start over.** `index` passes `reset=True` and
rebuilds the collection from scratch, which is correct at 37 chunks and absurd
at 37,000. Production needs per-document upserts keyed on content hash, so
re-indexing one changed file re-embeds only that file, and an embedding cache so
unchanged chunks are never paid for twice.

**6. The things that only matter once someone else can reach it.** Retries with
backoff on both API calls. A token budget on the assembled context, which
`top_k × chunk_size` currently leaves unbounded. Structured logging of the
retrieved ids and scores per query, because "why did it answer that?" is
unanswerable after the fact without them. Rate limits and a spend cap if it is
exposed to the internet. A real answer for scanned PDFs, which the indexer
currently warns about and skips rather than OCRs.

**7. Agentic RAG.** Let the model judge whether the retrieved context is
sufficient and issue follow-up searches before answering, instead of retrieving
exactly once. Deliberately last: it multiplies cost and latency per question,
and it is the wrong tool for the single-retrieval failures above. Worth adding
when the eval set shows questions that genuinely need more than one hop.
