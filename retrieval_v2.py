from typing import List, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from rrf import reciprocal_rank_fusion
from config import RRF_K, RETRIEVAL_K, LLM_MODEL
from utils import logger
import json


def generate_queries_simple(question: str, n: int = 4, model: Optional[str] = None) -> List[str]:
    """Generate simple alternative queries using LLM."""
    model_to_use = model or LLM_MODEL
    logger.info("→ Generating %d alternative queries using model '%s'...", n, model_to_use)

    template = (
        "You are a helpful assistant. Generate {n} concise search queries for: {question}\n"
        "Output as a JSON array of strings."
    )
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model=model_to_use, temperature=0)

    try:
        formatted = prompt.format(question=question, n=n)
        response = llm.invoke([HumanMessage(content=formatted)])
        text = response.content if hasattr(response, "content") else str(response)

        try:
            queries = json.loads(text)
            if not isinstance(queries, list):
                raise ValueError("Not a list")
        except Exception:
            queries = [line.strip() for line in text.splitlines() if line.strip()][:n]

        logger.info("→ Generated %d queries.", len(queries))
        logger.debug("→ Example generated queries: %s", queries)
        return queries

    except Exception as e:
        logger.exception("Query generation failed in v2: %s", e)
        return [question for _ in range(n)]


def run_rag_fusion(question: str, vectorstore, num_queries: int = 4, model: Optional[str] = None) -> str:
    """Multi-query retrieval with Reciprocal Rank Fusion."""
    model_to_use = model or LLM_MODEL
    logger.info("→ Starting RAG v2 (Fusion) pipeline with model '%s'", model_to_use)

    queries = generate_queries_simple(question, n=num_queries, model=model_to_use)

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    logger.info("→ Retrieving documents for %d queries (top %d each)...", len(queries), RETRIEVAL_K)
    all_lists = [retriever.get_relevant_documents(q) for q in queries]

    logger.info("→ Applying Reciprocal Rank Fusion (RRF_K=%d)...", RRF_K)
    fused = reciprocal_rank_fusion(all_lists, k=RRF_K)
    top_docs = [doc for doc, _ in fused][:RETRIEVAL_K]
    logger.info("→ Retrieved and fused %d top documents.", len(top_docs))
    if len(top_docs) > 0:
        logger.debug("→ Example fused doc snippet: %s...", top_docs[0].page_content[:200])

    context = "\n\n".join(d.page_content for d in top_docs)
    template = (
        "Answer the question using the context. "
        "If unknown, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(model=model_to_use, temperature=0)
    logger.info("→ Sending fused context to model '%s'...", model_to_use)

    try:
        formatted = prompt.format(context=context, question=question)
        response = llm.invoke([HumanMessage(content=formatted)])
        logger.info("→ Received response from model.")
        answer = response.content if hasattr(response, "content") else str(response)
        logger.debug("→ Final answer text (first 200 chars): %s...", answer[:200])
        return answer
    except Exception as e:
        logger.exception("LLM call failed in v2: %s", e)
        return ""
