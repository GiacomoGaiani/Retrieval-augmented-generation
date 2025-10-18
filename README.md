### Overview

### Version 1 — Basic RAG

- Uses a **single query** to retrieve the most relevant chunks from a Chroma vector database.
- Simple pipeline:
  1. Embed question
  2. Retrieve top-K chunks
  3. Pass context + question to the LLM
- Fastest but least flexible approach.

### Version 2 — RAG Fusion

- Expands the query into **multiple variations** using the LLM.
- Retrieves results for each variant, then **merges and reranks** using **Reciprocal Rank Fusion (RRF)**.
- Improves recall and factual grounding compared to V1.

### Version 3 — Advanced RAG

- Builds on RAG Fusion with:
  - **Cross-document reranking**
  - **Token-budget trimming**
  - **Context citation tracking**
  - **Source attribution in final answers**
- Produces the most accurate and well-sourced responses, though slower due to multiple retrieval and generation stages.

---

## Setup Instructions

### 1. Clone or copy the project

```bash
git clone <your_repo_url>
cd <project_folder>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

Make sure `requirements.txt` includes:

```
langchain
chromadb
sentence-transformers
InstructorEmbedding
tiktoken
ollama
```

Then run:

```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure defaults

Check or create `config.py`.

You can define:

```python
DEFAULT_EMBEDDER = "hf"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "chroma_db"
RETRIEVAL_K = 5
```

---

## Indexing Your Documents

### 1. Place your text or PDF files

Put all documents to be indexed in the `docs/` folder.

### 2. Run the indexer

```bash
python indexer.py
```

This will:

* Load all `.txt` and `.pdf` files
* Split them into chunks
* Embed and store them into a persistent **Chroma** vector database (`chroma_db/`)

### 3. Verify the database

After indexing, you should see a `chroma_db/` folder created in the project root.

---

## Running the RAG Pipelines

You can run any version directly using the CLI interface.

### Version 1 (Basic RAG)

```bash
python main.py query --version v1 --question "What is task decomposition?"
```

### Version 2 (RAG Fusion)

```bash
python main.py query --version v2 --question "How does RAG improve factual accuracy?"
```

### Version 3 (Advanced RAG)

```bash
python main.py query --version v3 --question "Explain goal-oriented behavior in AI systems."
```

Each command will:

* Load the vector store
* Retrieve relevant chunks
* Query the LLM
* Print a structured answer to the console

---

## Benchmarking

Two test scripts are provided:

### 1. `tests/test_benchmark.py`

Runs each version on a set of questions (spawns a new process for each).

* Measures total time per query.
* Prints answer snippets.

Usage:

```bash
python -m tests.test_benchmark
```

Output will show:

```
===== Testing version V1 =====
Q: What is task decomposition?
Time: 42.7s
Answer snippet: Task decomposition is a method...
------------------------------------------------------------
```

### 2. `tests/benchmark_retrieval_only.py`

Benchmarks retrieval performance only (no LLM calls).

Generates a CSV with timings and recall data.

Usage:

```bash
python -m tests.benchmark_retrieval_only
```

CSV output: `benchmark_retrieval_only.csv`

---

## Optional: Persistent Benchmark (Faster Testing)

If you want to test multiple questions  **without reloading models each time** , use:

```bash
python -m tests.persistent_benchmark
```

This keeps all models in memory for much faster results and writes `benchmark_answer_results_persistent.csv`.

---

## Project Structure

```
├── main.py                     # CLI entry point
├── indexer.py                  # Builds and loads vectorstore
├── retrieval_v1.py             # Basic RAG pipeline
├── retrieval_v2.py             # RAG Fusion (multi-query)
├── retrieval_v3.py             # Advanced RAG with reranking & citations
├── config.py                   # Global constants and settings
├── docs/                       # Your source documents
├── chroma_db/                  # Vector database (auto-created)
└── tests/
    ├── test_benchmark.py
    ├── benchmark_retrieval_only.py
    └── persistent_benchmark.py
```

---

## Tips & Notes

* The first run may be slower due to model downloads (embeddings + Ollama model).
* To change the LLM, update the model name in `config.py` (e.g., `"mistral"` → `"llama3"`).
* To speed up indexing or testing:
  * Use smaller embeddings (e.g., `all-MiniLM-L6-v2`)
  * Reduce `RETRIEVAL_K` to 3
  * Limit `num_queries` in RAG Fusion to 2
* You can freely add new retrieval strategies (v4, v5, etc.) following the same interface pattern.

---

## Summary

| Version      | Core Idea                                        | Strength                                  | Trade-off             |
| ------------ | ------------------------------------------------ | ----------------------------------------- | --------------------- |
| **v1** | Simple retrieval + LLM                           | Fast, minimal setup                       | Lower recall          |
| **v2** | Multi-query + RRF fusion                         | Better recall, improved factual grounding | Slightly slower       |
| **v3** | Reranking + context trimming + citation tracking | Most accurate and sourced                 | Heaviest compute cost |
