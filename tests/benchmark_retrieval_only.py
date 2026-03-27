"""
tests/benchmark_retrieval_only.py
----------------------------------
Benchmarks retrieval performance only — no LLM calls.
Since retrieval is embedded inside the pipeline functions, we time the full
retrieve step by calling the vectorstore directly, bypassing the LLM.

Outputs:
  - benchmark_retrieval_only.csv
  - Weights & Biases run per retrieval version

Metrics logged per query:
  - retrieval_time_s   : wall-clock seconds for the retrieve() call
  - chunks_returned    : number of Document objects returned
  - unique_sources     : number of distinct source files in the result set
  - version            : v1 / v2 / v3

Usage:
  python -m tests.benchmark_retrieval_only
  WANDB_MODE=disabled python -m tests.benchmark_retrieval_only
"""

import csv
import os
import time
import sys

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# W&B setup
# ---------------------------------------------------------------------------
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _init_wandb(version: str):
    if not _WANDB_AVAILABLE:
        return None
    if os.getenv("WANDB_MODE", "online") == "disabled":
        return None
    return wandb.init(
        project=os.getenv("WANDB_PROJECT", "rag-benchmark"),
        name=f"retrieval-only-{version}",
        group="benchmark_retrieval_only",
        config={
            "version": version,
            "retrieval_k": int(os.getenv("RETRIEVAL_K", "5")),
            "chunk_size": int(os.getenv("CHUNK_SIZE", "300")),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50")),
            "rrf_k": int(os.getenv("RRF_K", "60")),
            "embedder": os.getenv("DEFAULT_EMBEDDER", "instructor"),
            "embed_model": os.getenv("DEFAULT_EMBED_MODEL", "hkunlp/instructor-base"),
        },
        reinit=True,
    )


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from indexer import load_vectorstore
from config import (
    CHROMA_PERSIST_DIR, DEFAULT_EMBEDDER, DEFAULT_EMBED_MODEL, RETRIEVAL_K
)
from retrieval_v2 import generate_queries_simple
from rrf import reciprocal_rank_fusion

QUESTIONS = [
    "What is task decomposition?",
    "How does RAG improve factual accuracy?",
    "Explain goal-oriented behavior in AI systems.",
]

OUTPUT_CSV = "benchmark_retrieval_only.csv"


def unique_sources(docs) -> int:
    sources = set()
    for doc in docs:
        src = getattr(doc, "metadata", {}).get("source", "unknown")
        sources.add(src)
    return len(sources)


def retrieve_v1(question, vs):
    """Basic single-query retrieval."""
    retriever = vs.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    return retriever.invoke(question)


def retrieve_v2(question, vs):
    """Multi-query retrieval with RRF fusion (no LLM for final answer)."""
    queries = generate_queries_simple(question, n=4)
    retriever = vs.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    all_lists = [retriever.invoke(q) for q in queries]
    return reciprocal_rank_fusion(all_lists)


def retrieve_v3(question, vs):
    """Same as v2 retrieval — advanced RAG uses same retrieval, different generation."""
    return retrieve_v2(question, vs)


RETRIEVERS = {
    "v1": retrieve_v1,
    "v2": retrieve_v2,
    "v3": retrieve_v3,
}


def main():
    vs = load_vectorstore(
        CHROMA_PERSIST_DIR,
        embedder=DEFAULT_EMBEDDER,
        embed_model=DEFAULT_EMBED_MODEL,
        init_embedder=True,
    )

    rows = []

    for version, retrieve_fn in RETRIEVERS.items():
        print(f"\n{'='*55}")
        print(f"  Retrieval benchmark — {version.upper()}")
        print(f"{'='*55}")

        run = _init_wandb(version)

        for question in QUESTIONS:
            start = time.perf_counter()
            docs = retrieve_fn(question, vs)
            elapsed = time.perf_counter() - start

            n_chunks = len(docs)
            n_sources = unique_sources(docs)

            print(f"Q : {question[:60]}")
            print(f"    {elapsed:.3f}s | {n_chunks} chunks | {n_sources} sources")

            row = {
                "version": version,
                "question": question,
                "retrieval_time_s": round(elapsed, 4),
                "chunks_returned": n_chunks,
                "unique_sources": n_sources,
            }
            rows.append(row)

            if run is not None:
                wandb.log(row)

        if run is not None:
            version_rows = [r for r in rows if r["version"] == version]
            times = [r["retrieval_time_s"] for r in version_rows]
            wandb.log({
                "mean_retrieval_time_s": sum(times) / len(times),
                "max_retrieval_time_s": max(times),
                "min_retrieval_time_s": min(times),
            })
            wandb.finish()

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()