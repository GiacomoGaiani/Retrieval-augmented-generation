from typing import List
from langchain_core.documents import Document
import tiktoken
from config import TOKEN_BUDGET


def trim_to_token_budget(docs: List[Document], prompt_overhead: int = 500, model_encoding: str = "cl100k_base"):
    """Return a subset of docs whose combined tokens (approx) fit in TOKEN_BUDGET - prompt_overhead."""
    enc = tiktoken.get_encoding(model_encoding)
    budget = TOKEN_BUDGET - prompt_overhead
    kept = []
    total = 0
    for d in docs:
        tokens = len(enc.encode(d.page_content))
        if total + tokens > budget:
            break
        kept.append(d)
        total += tokens
    return kept
