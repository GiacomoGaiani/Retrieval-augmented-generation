import time
import csv
import subprocess

questions = [
    "What is task decomposition?",
    "Explain goal-oriented behavior in AI systems.",
    "How can RAG improve factual accuracy?"
]

versions = ["v1", "v2", "v3"]
results = []

for v in versions:
    print(f"\n===== Testing version {v.upper()} =====")
    for q in questions:
        print(f"→ Query: {q}")
        start = time.time()
        result = subprocess.run(
            ["python", "main.py", "query", "--version", v, "--question", q],
            capture_output=True, text=True
        )
        end = time.time()
        duration = round(end - start, 2)

        output = result.stdout.strip()
        snippet = output[:400].replace("\n", " ")

        print(f"Time: {duration}s\nAnswer snippet: {snippet[:300]}")
        print("-" * 80)

        results.append({
            "version": v,
            "question": q,
            "answer": output,
            "duration": duration
        })

with open("benchmark_answer_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nSaved detailed results to benchmark_answer_results.csv")
