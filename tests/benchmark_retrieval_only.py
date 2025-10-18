import sys, os
import time
import csv
import warnings
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore", category=DeprecationWarning)

from indexer import load_vectorstore
from config import RETRIEVAL_K, RRF_K, TOKEN_BUDGET
from rrf import reciprocal_rank_fusion
from trimming import trim_to_token_budget

def simple_alternative_queries(q: str, n: int = 4) -> List[str]:
    templates = [
        "{q}",
        "definition of {q}",
        "examples of {q}",
        "why is {q} important"
    ]
    return [t.format(q=q) for t in templates][:n]


questions = [
    "What is task decomposition?",
    "How do AI agents plan their actions?",
    "What is hierarchical reinforcement learning?",
    "Explain the difference between symbolic and sub-symbolic AI.",
    "What is the role of memory in autonomous agents?",
    "How does multi-agent collaboration work?",
    "What is a reactive agent?",
    "Explain goal-oriented behavior in AI systems.",
    "What are the limitations of task decomposition?",
    "How can RAG improve factual accuracy?"
]

def measure_v1(vs, q):
    retriever = vs.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    t0 = time.perf_counter()
    docs = retriever.get_relevant_documents(q)
    t1 = time.perf_counter()
    return {
        "retrieval_time": t1 - t0,
        "num_docs": len(docs)
    }

def measure_v2(vs, q, num_queries=4):
    retriever = vs.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    queries = simple_alternative_queries(q, n=num_queries)
    t0 = time.perf_counter()
    lists = [retriever.get_relevant_documents(qi) for qi in queries]
    t1 = time.perf_counter()
    rrf0 = time.perf_counter()
    fused = reciprocal_rank_fusion(lists, k=RRF_K)
    rrf1 = time.perf_counter()

    top_docs = []
    for item, score in fused:
        if hasattr(item, "page_content"):
            top_docs.append(item)
        elif isinstance(item, dict) and "content" in item:
            class D: pass
            d = D()
            d.page_content = item["content"]
            d.metadata = item.get("metadata", {})
            top_docs.append(d)
        else:
            class D: pass
            d = D()
            d.page_content = str(item)
            d.metadata = {}
            top_docs.append(d)

    return {
        "retrieval_time": t1 - t0,
        "rrf_time": rrf1 - rrf0,
        "num_fused": len(top_docs[:RETRIEVAL_K])
    }

def measure_v3(vs, q, num_queries=4):
    retriever = vs.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    queries = simple_alternative_queries(q, n=num_queries)
    t0 = time.perf_counter()
    lists = [retriever.get_relevant_documents(qi) for qi in queries]
    t1 = time.perf_counter()
    rrf0 = time.perf_counter()
    fused = reciprocal_rank_fusion(lists, k=RRF_K)
    rrf1 = time.perf_counter()
    docs_sorted = [doc for doc, _ in fused]
    trim0 = time.perf_counter()
    trimmed = trim_to_token_budget(docs_sorted)
    trim1 = time.perf_counter()

    count = sum(1 for d in trimmed)
    return {
        "retrieval_time": t1 - t0,
        "rrf_time": rrf1 - rrf0,
        "trimming_time": trim1 - trim0,
        "num_trimmed": count
    }

def run_benchmark():
    print("Loading vectorstore...")
    vs = load_vectorstore()
    rows = []

    for v in ["v1", "v2", "v3"]:
        print(f"\n=== Version {v} ===")
        for q in questions:
            if v == "v1":
                res = measure_v1(vs, q)
            elif v == "v2":
                res = measure_v2(vs, q)
            else:
                res = measure_v3(vs, q)
            row = {"version": v, "question": q, **res}
            rows.append(row)
            print(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open("benchmark_retrieval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\nSaved results to benchmark_retrieval_results.csv")

if __name__ == "__main__":
    run_benchmark()
