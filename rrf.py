from typing import List, Tuple
from langchain.schema import Document
import json

def reciprocal_rank_fusion(results: List[List[Document]], k=60) -> List[Tuple[Document, float]]:
    """Apply Reciprocal Rank Fusion and return top documents as Document objects."""
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_dict = {"content": doc.page_content, "metadata": doc.metadata}
            doc_str = json.dumps(doc_dict)
            fused_scores[doc_str] = fused_scores.get(doc_str, 0) + 1 / (rank + k)

    reranked_results: List[Tuple[Document, float]] = []

    for doc_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
        doc_dict = json.loads(doc_str)
        doc_obj = Document(page_content=doc_dict["content"], metadata=doc_dict["metadata"])
        reranked_results.append((doc_obj, score))

    return reranked_results
