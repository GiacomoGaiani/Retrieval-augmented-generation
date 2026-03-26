"""
tests/test_benchmark.py
-----------------------
Benchmarks v1 / v2 / v3 end-to-end (spawns a subprocess per query so each
version loads its own models cleanly).

Metrics logged to Weights & Biases:
  - query_time_s   : wall-clock seconds for the full pipeline
  - answer_length  : character length of the returned answer
  - version        : pipeline version (v1 / v2 / v3)
  - question       : the question asked

Usage:
  python -m tests.test_benchmark
  WANDB_MODE=disabled python -m tests.test_benchmark   # no W&B
"""

import subprocess
import sys
import time
import os

# ---------------------------------------------------------------------------
# W&B setup — gracefully disabled when WANDB_MODE=disabled or key is missing
# ---------------------------------------------------------------------------
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

def _init_wandb(version: str):
    """Initialise (or re-use) a W&B run for the given pipeline version."""
    if not _WANDB_AVAILABLE:
        return None
    if os.getenv("WANDB_MODE", "online") == "disabled":
        return None
    return wandb.init(
        project=os.getenv("WANDB_PROJECT", "rag-benchmark"),
        name=f"full-pipeline-{version}",
        group="test_benchmark",
        config={
            "version": version,
            "llm_model": os.getenv("LLM_MODEL", "mistral"),
            "retrieval_k": int(os.getenv("RETRIEVAL_K", "5")),
            "chunk_size": int(os.getenv("CHUNK_SIZE", "300")),
        },
        reinit=True,
    )

# ---------------------------------------------------------------------------
# Test questions
# ---------------------------------------------------------------------------
QUESTIONS = [
    "What is task decomposition?",
    "How does RAG improve factual accuracy?",
    "Explain goal-oriented behavior in AI systems.",
]

VERSIONS = ["v1", "v2", "v3"]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_version(version: str, question: str) -> tuple[float, str]:
    """Run main.py in a subprocess; return (elapsed_seconds, answer_snippet)."""
    cmd = [
        sys.executable, "main.py",
        "query",
        "--version", version,
        "--question", question,
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    output = result.stdout.strip()
    # Extract the ANSWER section from the CLI output
    answer = ""
    if "ANSWER:" in output:
        answer = output.split("ANSWER:")[-1].strip()
    elif output:
        answer = output

    return elapsed, answer


def main():
    for version in VERSIONS:
        print(f"\n{'='*60}")
        print(f"  Testing version {version.upper()}")
        print(f"{'='*60}")

        run = _init_wandb(version)

        for question in QUESTIONS:
            print(f"\nQ: {question}")
            elapsed, answer = run_version(version, question)
            snippet = answer[:200] + ("…" if len(answer) > 200 else "")

            print(f"Time : {elapsed:.1f}s")
            print(f"Answer snippet: {snippet}")
            print("-" * 60)

            if run is not None:
                wandb.log({
                    "version": version,
                    "question": question,
                    "query_time_s": elapsed,
                    "answer_length": len(answer),
                })

        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
