"""
tests/benchmark_retrieval_only.py
----------------------------------
Benchmarks retrieval performance only — no LLM calls.
Measures latency and a simple recall proxy (number of unique chunks returned).

Outputs:
  - benchmark_retrieval_only.csv  (same as before)
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
# Retrieval wrappers
# ---------------------------------------------------------------------------
from indexer import load_vectorstore
from config import CHROMA_PERSIST_DIR, DEFAULT_EMBEDDER, DEFAULT_EMBED_MODEL, RETRIEVAL_K
from retrieval_v1 import retrieve as retrieve_v1
from retrieval_v2 import retrieve as retrieve_v2
from retrieval_v3 import retrieve as retrieve_v3

RETRIEVERS = {
    "v1": retrieve_v1,
    "v2": retrieve_v2,
    "v3": retrieve_v3,
}

QUESTIONS = [
    "What is task decomposition?",
    "How does RAG improve factual accuracy?",
    "Explain goal-oriented behavior in AI systems.",
    "What are the limitations of transformer models?",
    "How is context window size related to model performance?",
]

OUTPUT_CSV = "benchmark_retrieval_only.csv"


def unique_sources(docs) -> int:
    """Count distinct source filenames across retrieved documents."""
    sources = set()
    for doc in docs:
        src = getattr(doc, "metadata", {}).get("source", "unknown")
        sources.add(src)
    return len(sources)


def main():
    # Load vectorstore once
    vs = load_vectorstore(
        CHROMA_PERSIST_DIR,
        embedder=DEFAULT_EMBEDDER,
        embed_model=DEFAULT_EMBED_MODEL,
        init_embedder=False,
    )

    rows = []

    for version, retrieve_fn in RETRIEVERS.items():
        print(f"\n{'='*55}")
        print(f"  Retrieval benchmark — {version.upper()}")
        print(f"{'='*55}")

        run = _init_wandb(version)

        for question in QUESTIONS:
            start = time.perf_counter()
            docs = retrieve_fn(question, vs, k=RETRIEVAL_K)
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
            # Log per-version summary statistics
            version_rows = [r for r in rows if r["version"] == version]
            times = [r["retrieval_time_s"] for r in version_rows]
            wandb.log({
                "mean_retrieval_time_s": sum(times) / len(times),
                "max_retrieval_time_s": max(times),
                "min_retrieval_time_s": min(times),
            })
            wandb.finish()

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
