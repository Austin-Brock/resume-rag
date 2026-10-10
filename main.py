#!/usr/bin/env python3
"""A minimal RAG pipeline over a PDF, written without a framework.

Every stage is a plain function so you can read exactly what happens:

    PDF -> extract_pages -> chunk_text -> embed -> ChromaDB
    question -> embed -> ChromaDB search -> chunks + question -> LLM -> answer

Subcommands:
    index   Build the vector store from one or more PDFs.
    search  Retrieve chunks only. No LLM. Use this to prove retrieval works.
    ask     Retrieve chunks, then send them to the LLM for an answer.
"""

import argparse
import os
import sys
import textwrap
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
COLLECTION = "documents"
STORE_PATH = ".chroma"

SYSTEM_PROMPT = """You are an assistant answering questions about a software \
engineer's professional experience.

Answer questions using ONLY the provided context. If the context doesn't \
contain the answer, say you don't have enough information. Do not invent \
experience or qualifications.

Each context block is labelled with its source and page. When you use a fact, \
cite it inline like (resume.pdf, page 1)."""


# --- Stage 1: PDF -> text ---------------------------------------------------

def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for pages that actually contain text.

    Pages are kept separate so a chunk can remember which page it came from,
    which is what makes source citations possible later.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((page_number, text))
    return pages


# --- Stage 2: text -> chunks ------------------------------------------------

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into fixed-size chunks that overlap by `overlap` characters.

    The overlap matters: without it, a sentence split across a boundary is
    unretrievable, because neither half carries the whole idea.
    """
    stride = size - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += stride
    return chunks


def build_records(pdf_paths: list[Path], size: int, overlap: int) -> list[dict]:
    """Turn PDFs into chunk records carrying the metadata needed for citations."""
    records = []
    for pdf_path in pdf_paths:
        pages = extract_pages(pdf_path)
        if not pages:
            print(f"  warning: no extractable text in {pdf_path.name} "
                  f"(is it a scanned image?)", file=sys.stderr)
        for page_number, text in pages:
            for i, chunk in enumerate(chunk_text(text, size, overlap)):
                records.append({
                    "id": f"{pdf_path.name}:p{page_number}:c{i}",
                    "text": chunk,
                    "source": pdf_path.name,
                    "page": page_number,
                })
    return records


# --- Stage 3: chunks -> embeddings ------------------------------------------

def embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a list of strings, batched to stay well inside request limits."""
    vectors = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        # The API preserves input order, so response.data lines up with batch.
        vectors.extend(item.embedding for item in response.data)
    return vectors


# --- Stage 4: storage and retrieval -----------------------------------------

def get_collection(reset: bool = False):
    """Open the persistent Chroma collection, optionally rebuilding it."""
    store = chromadb.PersistentClient(path=STORE_PATH)
    if reset:
        try:
            store.delete_collection(COLLECTION)
        except Exception:
            pass  # Nothing to delete on a first run.
    # cosine matches how OpenAI embeddings are meant to be compared;
    # Chroma's default is L2, which is the wrong metric here.
    return store.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def retrieve(openai_client: OpenAI, collection, question: str, k: int) -> list[dict]:
    """Embed the question and return the k most semantically similar chunks."""
    question_vector = embed(openai_client, [question])[0]
    results = collection.query(query_embeddings=[question_vector], n_results=k)

    hits = []
    for text, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            # Chroma returns cosine distance; similarity is easier to read.
            "similarity": 1 - distance,
        })
    return hits


# --- Stage 5: generation ----------------------------------------------------

def answer(openai_client: OpenAI, question: str, hits: list[dict]) -> str:
    """Send the retrieved context plus the question to the LLM."""
    if hits:
        context = "\n\n".join(
            f"[{h['source']}, page {h['page']}]\n{h['text']}" for h in hits
        )
        user_content = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        # The --no-rag experiment: no retrieval, so the model has only its
        # own prior knowledge to draw on. Watch what it does with that.
        user_content = f"Question: {question}"

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# --- CLI --------------------------------------------------------------------

def make_client() -> OpenAI:
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        sys.exit("No OPENAI_API_KEY found. Copy .env.example to .env and add your key.")
    return OpenAI(api_key=key)


def print_hits(hits: list[dict]) -> None:
    for i, hit in enumerate(hits, start=1):
        print(f"\n--- {i}. {hit['source']}, page {hit['page']} "
              f"(similarity {hit['similarity']:.3f}) ---")
        print(textwrap.fill(hit["text"], width=88))


def cmd_index(args) -> None:
    pdf_paths = [Path(p) for p in args.pdfs]
    for path in pdf_paths:
        if not path.exists():
            sys.exit(f"File not found: {path}")

    print(f"Reading {len(pdf_paths)} PDF(s)...")
    records = build_records(pdf_paths, args.chunk_size, args.overlap)
    if not records:
        sys.exit("No text extracted, so there is nothing to index.")
    print(f"Produced {len(records)} chunks "
          f"(size={args.chunk_size}, overlap={args.overlap}).")

    client = make_client()
    print(f"Embedding with {EMBED_MODEL}...")
    vectors = embed(client, [r["text"] for r in records])

    collection = get_collection(reset=True)
    collection.add(
        ids=[r["id"] for r in records],
        documents=[r["text"] for r in records],
        embeddings=vectors,
        metadatas=[{"source": r["source"], "page": r["page"]} for r in records],
    )
    print(f"Stored {collection.count()} chunks in {STORE_PATH}/")


def cmd_search(args) -> None:
    client = make_client()
    collection = get_collection()
    if collection.count() == 0:
        sys.exit("The store is empty. Run: python main.py index resume.pdf")
    hits = retrieve(client, collection, args.question, args.top_k)
    print(f"Top {len(hits)} chunks for: {args.question!r}")
    print_hits(hits)


def cmd_ask(args) -> None:
    client = make_client()

    if args.no_rag:
        print("[--no-rag: skipping retrieval entirely]\n")
        hits = []
    else:
        collection = get_collection()
        if collection.count() == 0:
            sys.exit("The store is empty. Run: python main.py index resume.pdf")
        hits = retrieve(client, collection, args.question, args.top_k)
        if args.show_chunks:
            print(f"Retrieved {len(hits)} chunks:")
            print_hits(hits)
            print()

    print(answer(client, args.question, hits))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A minimal, framework-free RAG pipeline over PDFs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build the vector store from PDFs")
    p_index.add_argument("pdfs", nargs="+", help="one or more PDF paths")
    p_index.add_argument("--chunk-size", type=int, default=500,
                         help="characters per chunk (default: 500)")
    p_index.add_argument("--overlap", type=int, default=100,
                         help="characters of overlap between chunks (default: 100)")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="retrieve chunks only, no LLM")
    p_search.add_argument("question")
    p_search.add_argument("--top-k", type=int, default=3,
                          help="chunks to retrieve (default: 3)")
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="retrieve chunks, then answer with the LLM")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=3,
                       help="chunks to retrieve (default: 3)")
    p_ask.add_argument("--show-chunks", action="store_true",
                       help="print the retrieved context before the answer")
    p_ask.add_argument("--no-rag", action="store_true",
                       help="skip retrieval; ask the model cold")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
