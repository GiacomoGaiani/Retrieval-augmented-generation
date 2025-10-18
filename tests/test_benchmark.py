import time
import subprocess

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

versions = ["v1", "v2", "v3"]

for v in versions:
    print(f"\n===== Testing version {v.upper()} =====")
    for q in questions:
        start = time.time()
        result = subprocess.run(
            ["python", "main.py", "query", "--version", v, "--question", q],
            capture_output=True, text=True
        )
        end = time.time()
        duration = round(end - start, 2)
        print(f"Q: {q}\nTime: {duration}s\nAnswer snippet: {result.stdout[:300]}\n{'-'*60}")
