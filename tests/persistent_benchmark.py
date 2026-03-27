"""
tests/persistent_benchmark.py
------------------------------
Keeps all models in memory across questions for faster benchmarking.
Writes benchmark_answer_results_persistent.csv.

Metrics logged to W&B per question:
  - query_time_s        : end-to-end wall-clock seconds (retrieval + LLM)
  - answer_length       : character count of the answer
  - answer_word_count   : word count of the answer

Usage:
  python -m tests.persistent_benchmark
  WANDB_MODE=disabled python -m tests.persistent_benchmark
"""

import csv
import os
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# W&B setup
# ---------------------------------------------------------------------------
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _init_wandb():
    if not _WANDB_AVAILABLE:
        return None
    if os.getenv("WANDB_MODE", "online") == "disabled":
        return None
    return wandb.init(
        project=os.getenv("WANDB_PROJECT", "rag-benchmark"),
        name="persistent-benchmark",
        group="persistent_benchmark",
        config={
            "llm_model": os.getenv("LLM_MODEL", "mistral"),
            "retrieval_k": int(os.getenv("RETRIEVAL_K", "5")),
            "chunk_size": int(os.getenv("CHUNK_SIZE", "300")),
            "token_budget": int(os.getenv("TOKEN_BUDGET", "3000")),
            "embedder": os.getenv("DEFAULT_EMBEDDER", "instructor"),
            "embed_model": os.getenv("DEFAULT_EMBED_MODEL", "hkunlp/instructor-base"),
        },
        reinit=True,
    )


# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
from indexer import load_vectorstore
from retrieval_v1 import run_basic_rag
from retrieval_v2 import run_rag_fusion
from retrieval_v3 import run_advanced_rag
from config import (
    CHROMA_PERSIST_DIR, DEFAULT_EMBEDDER, DEFAULT_EMBED_MODEL, LLM_MODEL
)

QUESTIONS = [
    "What is task decomposition?",
    "How does RAG improve factual accuracy?",
    "Explain goal-oriented behavior in AI systems.",
]

VERSIONS = {
    "v1": run_basic_rag,
    "v2": run_rag_fusion,
    "v3": run_advanced_rag,
}

OUTPUT_CSV = "benchmark_answer_results_persistent.csv"


def main():
    run = _init_wandb()

    vs = load_vectorstore(
        CHROMA_PERSIST_DIR,
        embedder=DEFAULT_EMBEDDER,
        embed_model=DEFAULT_EMBED_MODEL,
        init_embedder=True,
    )

    rows = []

    for version, pipeline_fn in VERSIONS.items():
        print(f"\n{'='*60}")
        print(f"  Persistent benchmark — {version.upper()}")
        print(f"{'='*60}")

        for question in QUESTIONS:
            print(f"\nQ: {question}")

            start = time.perf_counter()
            answer = pipeline_fn(question, vs, model=LLM_MODEL)
            elapsed = time.perf_counter() - start

            snippet = str(answer)[:200] + ("…" if len(str(answer)) > 200 else "")
            print(f"Time   : {elapsed:.2f}s")
            print(f"Answer : {snippet}")

            row = {
                "version": version,
                "question": question,
                "query_time_s": round(elapsed, 3),
                "answer_length": len(str(answer)),
                "answer_word_count": len(str(answer).split()),
                "answer_snippet": snippet,
            }
            rows.append(row)

            if run is not None:
                wandb.log({
                    "version": version,
                    "question": question,
                    "query_time_s": row["query_time_s"],
                    "answer_length": row["answer_length"],
                    "answer_word_count": row["answer_word_count"],
                })

    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = ["version", "question", "query_time_s",
                      "answer_length", "answer_word_count", "answer_snippet"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written → {OUTPUT_CSV}")

    if run is not None:
        artifact = wandb.Artifact("benchmark_results", type="dataset")
        artifact.add_file(OUTPUT_CSV)
        run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()