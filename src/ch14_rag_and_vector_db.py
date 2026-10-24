"""
Chapter 14: Foundations of Generative AI, RAG & Vector Databases
Modern Machine Learning & Data Science Engineering
"""
import os
import json
import math

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

def mock_dense_embedding(text):
    keywords = ["mortgage", "credit", "fico", "dti", "income", "tax", "rate", "points", "underwriting", "down payment"]
    text_lower = text.lower()
    vec = [text_lower.count(kw) + 0.05 for kw in keywords]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]

def demonstrate_vector_rag():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(script_dir, "..", "data", "knowledge_base_rag.json")

    with open(kb_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    print(f"--- 1. Ingesting {len(docs)} Knowledge Base Documents into Vector Index ---")
    indexed_docs = []
    for doc in docs:
        full_text = f"{doc['title']} {doc['content']}"
        embedding = mock_dense_embedding(full_text)
        indexed_docs.append({"doc": doc, "embedding": embedding})
        print(f"Indexed [{doc['doc_id']}]: {doc['title']}")

    user_query = "What is the maximum debt to income ratio allowed?"
    print(f"\n--- 2. Semantic Query: \"{user_query}\" ---")
    query_vec = mock_dense_embedding(user_query)

    scored_results = []
    for item in indexed_docs:
        score = cosine_similarity(query_vec, item["embedding"])
        scored_results.append((score, item["doc"]))

    scored_results.sort(key=lambda x: x[0], reverse=True)

    print("\n--- 3. Top-Ranked Retrieved Knowledge Passages (RAG Context Injection) ---")
    for rank, (score, doc) in enumerate(scored_results[:2], start=1):
        print(f"\n[Rank {rank}] Similarity Score: {score:.4f} | ID: {doc['doc_id']}")
        print(f"Title:   {doc['title']}")
        print(f"Content: {doc['content']}")

if __name__ == "__main__":
    demonstrate_vector_rag()
