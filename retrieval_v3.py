from typing import List, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from rrf import reciprocal_rank_fusion
from trimming import trim_to_token_budget
from config import RETRIEVAL_K, LLM_MODEL
from utils import logger
import json


def generate_queries_json(question: str, n: int = 4, model: Optional[str] = None) -> List[str]:
    """Structured JSON query generation. Returns a list of query strings."""
    model_to_use = model or LLM_MODEL
    logger.info("→ Generating %d JSON-structured queries using '%s'...", n, model_to_use)

    template = "You are a helpful assistant. Produce {n} concise search queries in JSON array for: {question}"
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model=model_to_use, temperature=0)

    try:
        formatted = prompt.format(question=question, n=n)
        response = llm.invoke([HumanMessage(content=formatted)])
        text = response.content if hasattr(response, "content") else str(response)

        try:
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("Not a list")
            queries = parsed
        except Exception:
            queries = [line.strip() for line in text.splitlines() if line.strip()][:n]

        logger.info("→ Generated %d queries.", len(queries))
        logger.debug("→ Example queries: %s", queries)
        return queries

    except Exception as e:
        logger.exception("Query-generation LLM failed: %s", e)
        return [question for _ in range(n)]


def call_llm_with_citations(context_docs: List, question: str, model: Optional[str] = None) -> str:
    """Ask the model to answer and cite sources."""
    model_to_use = model or LLM_MODEL
    logger.info("→ Preparing final context with citations for model '%s'...", model_to_use)

    context_text = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(context_docs, start=1))
    template = (
        "Answer the question using ONLY the provided context. "
        "For each factual claim, cite the source by number.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model=model_to_use, temperature=0)

    try:
        formatted = prompt.format(context=context_text, question=question)
        logger.info("→ Sending context with citations to model '%s'...", model_to_use)
        response = llm.invoke([HumanMessage(content=formatted)])
        logger.info("→ Received final answer with citations.")
        answer = response.content if hasattr(response, "content") else str(response)
        logger.debug("→ Final answer text (first 200 chars): %s...", answer[:200])

        sources = "\n".join(
            f"[{i}] {d.metadata.get('source', 'Unknown source')}"
            for i, d in enumerate(context_docs, start=1)
        )
        return f"{answer.strip()}\n\nSources:\n{sources}"

    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return ""


def run_advanced_rag(question: str, vectorstore, num_queries: int = 4, model: Optional[str] = None) -> str:
    """End-to-end advanced RAG: multi-query -> RRF -> trimming -> LLM with citations."""
    model_to_use = model or LLM_MODEL
    logger.info("→ Starting RAG v3 (Advanced) pipeline with model '%s'", model_to_use)

    queries = generate_queries_json(question, n=num_queries, model=model_to_use)

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    logger.info("→ Retrieving documents for %d queries (top %d each)...", len(queries), RETRIEVAL_K)
    results = [retriever.get_relevant_documents(q) for q in queries]

    logger.info("→ Applying Reciprocal Rank Fusion...")
    fused = reciprocal_rank_fusion(results)
    docs_sorted = [doc for doc, _ in fused]
    logger.info("→ Fused total of %d documents.", len(docs_sorted))
    logger.info("→ Trimming documents to fit token budget...")
    trimmed = trim_to_token_budget(docs_sorted)
    logger.info("→ Trimmed to %d documents.", len(trimmed))
    if len(trimmed) > 0:
        logger.debug("→ Example trimmed doc snippet: %s...", trimmed[0].page_content[:200])

    return call_llm_with_citations(trimmed, question, model=model_to_use)
