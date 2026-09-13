import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sciharness.rag import SimpleRetriever, EmbeddingRetriever

base_dir = os.path.dirname(__file__)
kb_dir = os.path.join(base_dir, "knowledge_base")

with open(os.path.join(base_dir, "questions.json"), encoding="utf-8") as f:
    questions = json.load(f)

tfidf = SimpleRetriever(kb_dir, top_k=3)
embed = EmbeddingRetriever(kb_dir, top_k=3)

for q in questions:
    print(f"\n{'='*60}\n[{q['id']}] {q['question']}")
    print("--- TF-IDF ---")
    for doc_id, chunk, score in tfidf.search(q["question"]):
        print(f"  score={score:.3f} [{doc_id}] {chunk[:80]}...")
    print("--- Embedding ---")
    for doc_id, chunk, score in embed.search(q["question"]):
        print(f"  score={score:.3f} [{doc_id}] {chunk[:80]}...")
