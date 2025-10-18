from typing import List, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from config import RETRIEVAL_K, LLM_MODEL
from utils import logger


def run_basic_rag(question: str, vectorstore, model: Optional[str] = None) -> str:
    """Simple retrieval-augmented generation (RAG v1) using Ollama."""
    model_to_use = model or LLM_MODEL
    logger.info("→ Starting RAG v1 pipeline with model '%s'", model_to_use)

    logger.info("→ Retrieving top %d documents for question: %s", RETRIEVAL_K, question)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    docs: List = retriever.get_relevant_documents(question)

    logger.info("→ Retrieved %d documents.", len(docs))
    if len(docs) > 0:
        logger.debug("→ Example doc snippet: %s...", docs[0].page_content[:200])

    context = "\n\n".join(d.page_content for d in docs)

    template = (
        "Answer the question using only the provided context. "
        "If the answer cannot be found, say 'I don't know'.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(model=model_to_use, temperature=0)
    logger.info("→ Sending prompt to model '%s'...", model_to_use)

    try:
        formatted = prompt.format(context=context, question=question)
        response = llm.invoke([HumanMessage(content=formatted)])
        logger.info("→ Received response from model.")
        answer = response.content if hasattr(response, "content") else str(response)

        logger.debug("→ Final answer text (first 200 chars): %s...", answer[:200])
        return answer

    except Exception as e:
        logger.exception("LLM call failed in v1: %s", e)
        return ""
