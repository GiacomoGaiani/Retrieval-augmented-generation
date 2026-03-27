# Retrieval-Augmented Generation Benchmark

This project benchmarks three versions of a Retrieval-Augmented Generation (RAG) pipeline, from a simple single-query approach to an advanced multi-query pipeline with reranking, token trimming, and source citation.

---

## Pipeline Versions

### Version 1 — Basic RAG
- Uses a **single query** to retrieve the most relevant chunks from a Chroma vector database.
- Simple pipeline: embed question → retrieve top-K chunks → pass context + question to the LLM.
- Fastest but least flexible approach.

### Version 2 — RAG Fusion
- Expands the query into **multiple variations** using the LLM.
- Retrieves results for each variant, then **merges and reranks** using **Reciprocal Rank Fusion (RRF)**.
- Improves recall and factual grounding compared to V1.

### Version 3 — Advanced RAG
- Builds on RAG Fusion with **cross-document reranking**, **token-budget trimming**, **context citation tracking**, and **source attribution** in final answers.
- Produces the most accurate and well-sourced responses, though slower due to multiple retrieval and generation stages.

---

## Project Structure
```
├── main.py                          # CLI entry point
├── indexer.py                       # Builds and loads vectorstore
├── retrieval_v1.py                  # Basic RAG pipeline
├── retrieval_v2.py                  # RAG Fusion (multi-query)
├── retrieval_v3.py                  # Advanced RAG with reranking & citations
├── config.py                        # Global constants and settings (env-driven)
├── rrf.py                           # Reciprocal Rank Fusion implementation
├── trimming.py                      # Token-budget trimming
├── loaders.py                       # Document loaders (PDF, text, web)
├── utils.py                         # Logging and shared utilities
├── Dockerfile                       # Container image definition
├── docker-compose.yml               # Orchestration (containerized or host Ollama)
├── .env.example                     # Environment variable template
├── docs/                            # Your source documents (not tracked)
├── chroma_db/                       # Vector database (auto-created, not tracked)
└── tests/
    ├── test_benchmark.py            # Full pipeline benchmark (with LLM)
    ├── benchmark_retrieval_only.py  # Retrieval-only benchmark (no LLM, fast)
    └── persistent_benchmark.py     # In-memory benchmark (fastest)
```

---

## Requirements

- [Docker](https://www.docker.com/) and Docker Compose
- [Ollama](https://ollama.com/) with a model pulled (default: `mistral`)

No Python environment needed if running via Docker.

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/GiacomoGaiani/Retrieval-augmented-generation.git
cd Retrieval-augmented-generation
```

### 2. Configure environment variables
```bash
cp .env.example .env
```

Open `.env` and fill in your values. At minimum you need:
```
OPENAI_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
LLM_MODEL=mistral
```

### 3. Add your documents

Place `.txt` or `.pdf` files in the `docs/` folder. These will be indexed into the vector database.

---

## Running with Docker

Two modes are supported depending on whether you want Docker to manage Ollama or use an instance already running on your machine.

### Mode A — Ollama inside Docker (default)

Best for a clean, self-contained setup.
```bash
# Start Ollama and pull the model
docker compose up -d ollama
docker compose run --rm ollama-pull

# Index your documents
docker compose run --rm rag main.py index /app/docs

# Run a query
docker compose run --rm rag main.py query --version v1 --question "What is task decomposition?"
```

### Mode B — Ollama already running on your host

Use this if Ollama is already running locally (e.g. `ollama serve` or the Ollama desktop app).
Uses the `host-ollama` profile which routes API calls to `host.docker.internal:11434`.
```bash
# Build the image
docker compose --profile host-ollama build

# Index your documents
docker compose --profile host-ollama run --rm rag-host main.py index /app/docs

# Run a query
docker compose --profile host-ollama run --rm rag-host main.py query --version v1 --question "What is task decomposition?"
```

> **Linux users:** `host.docker.internal` is automatically mapped via `extra_hosts: host-gateway` in the compose file. Requires Docker 20.10+.

---

## Benchmarking

### Retrieval-only benchmark (fast, no LLM calls)

Measures retrieval latency and recall across all three versions. Outputs `benchmark_retrieval_only.csv`.
```bash
# Local
python -m tests.benchmark_retrieval_only

# Docker (Mode B)
docker compose --profile host-ollama run --rm rag-host -m tests.benchmark_retrieval_only
```

### Full pipeline benchmark (includes LLM calls)

Runs each version on a set of questions and measures end-to-end time. Prints answer snippets.
```bash
# Local
python -m tests.test_benchmark

# Docker (Mode B)
docker compose --profile host-ollama run --rm rag-host -m tests.test_benchmark
```

### Persistent benchmark (fastest, models kept in memory)

Runs all questions without reloading models between queries. Outputs `benchmark_answer_results_persistent.csv`.
```bash
# Local
python -m tests.persistent_benchmark

# Docker (Mode B)
docker compose --profile host-ollama run --rm rag-host -m tests.persistent_benchmark
```

---

## Experiment Tracking with Weights & Biases

All three benchmark scripts integrate with [Weights & Biases](https://wandb.ai) for experiment tracking. Each run logs metrics, hyperparameters, and results directly to your W&B dashboard, making it easy to compare pipeline versions across experiments.

### What gets tracked

| Script | W&B group | Metrics logged |
|---|---|---|
| `benchmark_retrieval_only.py` | `benchmark_retrieval_only` | `retrieval_time_s`, `chunks_returned`, `unique_sources`, `mean/max/min_retrieval_time_s` |
| `test_benchmark.py` | `test_benchmark` | `query_time_s`, `answer_length` |
| `persistent_benchmark.py` | `persistent_benchmark` | `query_time_s`, `answer_length`, `answer_word_count` + CSV uploaded as artifact |

Each run also records its full configuration as hyperparameters: `llm_model`, `retrieval_k`, `chunk_size`, `chunk_overlap`, `rrf_k`, `embedder`, `embed_model`.

### Setup

Add your W&B API key and project name to `.env`:
```
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=rag-benchmark
```

W&B is an **optional dependency** — all scripts fall back gracefully if the package is missing or if the key is not set. To explicitly disable tracking without removing the key:
```bash
WANDB_MODE=disabled python -m tests.benchmark_retrieval_only
```

### Docker

The `WANDB_API_KEY` is passed through automatically via the `env_file` in `docker-compose.yml`. No extra steps needed:
```bash
docker compose --profile host-ollama run --rm rag-host -m tests.benchmark_retrieval_only
```
```

---

**Update the `.env.example`** — add these two lines:
```
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=rag-benchmark
```

---

**Update the project structure block** — replace the `tests/` lines with:
```
└── tests/
    ├── test_benchmark.py            # Full pipeline benchmark — logs query_time_s, answer_length to W&B
    ├── benchmark_retrieval_only.py  # Retrieval-only benchmark — logs retrieval_time_s, chunks, sources to W&B
    └── persistent_benchmark.py     # In-memory benchmark — logs query_time_s, answer_word_count, uploads CSV artifact to W&B

---

## Running Locally (without Docker)

### 1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Index documents and run queries
```bash
python main.py index docs/
python main.py query --version v1 --question "What is task decomposition?"
python main.py query --version v2 --question "How does RAG improve factual accuracy?"
python main.py query --version v3 --question "Explain goal-oriented behavior in AI systems."
```

---

## Configuration

All settings are driven by environment variables (via `.env`) with sensible defaults in `config.py`:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `mistral` | Ollama model name |
| `DEFAULT_EMBEDDER` | `instructor` | Embedder type (`instructor`, `hf`, `openai`, `cohere`) |
| `DEFAULT_EMBED_MODEL` | `hkunlp/instructor-base` | Embedding model name |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector database directory |
| `RETRIEVAL_K` | `5` | Number of chunks to retrieve |
| `CHUNK_SIZE` | `300` | Token chunk size for splitting |
| `CHUNK_OVERLAP` | `50` | Token overlap between chunks |
| `RRF_K` | `60` | RRF ranking constant |
| `TOKEN_BUDGET` | `3000` | Max tokens for context trimming (V3) |

---

## Tips

- The first Docker run is slower due to embedding model downloads — subsequent runs use the cached image.
- To switch LLM, change `LLM_MODEL` in `.env` and run `ollama pull <model>` on the host.
- To speed up testing: use `all-MiniLM-L6-v2` as embedder (`DEFAULT_EMBEDDER=hf`), reduce `RETRIEVAL_K` to 3.
- New retrieval strategies (v4, v5, …) can be added by following the same interface pattern as `retrieval_v1.py`.

---

## Version Summary

| Version | Core Idea | Strength | Trade-off |
|---|---|---|---|
| **v1** | Simple retrieval + LLM | Fast, minimal setup | Lower recall |
| **v2** | Multi-query + RRF fusion | Better recall, improved factual grounding | Slightly slower |
| **v3** | Reranking + trimming + citation tracking | Most accurate and sourced | Heaviest compute cost |