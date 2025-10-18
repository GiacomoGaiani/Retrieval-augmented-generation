import typer
import os
from typing import List
from loaders import load_blog, load_pdf
from indexer import build_vectorstore, load_vectorstore
from retrieval_v1 import run_basic_rag
from retrieval_v2 import run_rag_fusion
from retrieval_v3 import run_advanced_rag
from config import CHROMA_PERSIST_DIR, DEFAULT_EMBEDDER, DEFAULT_EMBED_MODEL
from utils import set_log_level

app = typer.Typer()

@app.command()
def index(urls: List[str], persist_dir: str = CHROMA_PERSIST_DIR,
          embedder: str = None, embed_model: str = None):
    from config import DEFAULT_EMBEDDER, DEFAULT_EMBED_MODEL
    embedder = embedder or DEFAULT_EMBEDDER
    embed_model = embed_model or DEFAULT_EMBED_MODEL

    docs = []
    for u in urls:
        if os.path.isdir(u):
            for fname in os.listdir(u):
                path = os.path.join(u, fname)
                if fname.lower().endswith(".pdf"):
                    docs.extend(load_pdf(path))
                elif fname.lower().endswith(".txt"):
                    docs.extend(load_blog(path))
        else:
            docs.extend(load_blog(u))
    build_vectorstore(docs, persist_dir=persist_dir, embedder=embedder, embed_model=embed_model)
    typer.echo("Indexing complete.")

@app.command()
def query(
    version: str = "v1",
    question: str = "",
    persist_dir: str = CHROMA_PERSIST_DIR,
    llm_model: str = typer.Option(None, help="Optional override for LLM model"),
    verbose: bool = False,
    debug: bool = False,
):
    from config import LLM_MODEL
    model_to_use = llm_model or LLM_MODEL
    set_log_level(verbose=verbose, debug=debug)

    vs = load_vectorstore(persist_dir, embedder="instructor", embed_model="hkunlp/instructor-base", init_embedder=False)

    if version == "v1":
        res = run_basic_rag(question, vs, model=model_to_use)
    elif version == "v2":
        res = run_rag_fusion(question, vs, model=model_to_use)
    elif version == "v3":
        res = run_advanced_rag(question, vs, model=model_to_use)
    else:
        raise ValueError("version must be v1, v2, or v3")

    typer.echo(f"\nQUESTION:\n{question}\n")
    typer.echo(f"ANSWER:\n{res}\n")

@app.command()
def run(action: str = "index", urls: List[str] = typer.Option(None), version: str = "v1", question: str = ""):
    if action == "index":
        if not urls:
            raise typer.BadParameter("Provide at least one URL to index")
        import asyncio
        asyncio.run(index(urls))
    elif action == "query":
        query(version=version, question=question)
    else:
        raise typer.BadParameter("action must be index or query")

if __name__ == "__main__":
    app()
