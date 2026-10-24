"""
Chapter 7: Word Embeddings & Dense Semantic Vector Analogies
Modern Machine Learning & Data Science Engineering
"""
import numpy as np
import math

def cosine_similarity(u, v):
    dot = np.dot(u, v)
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    return dot / norm if norm > 0 else 0.0

def demonstrate_word_vectors():
    print("=====================================================================")
    print("   WORD EMBEDDINGS & DENSE SEMANTIC VECTOR ARITHMETIC                ")
    print("=====================================================================")

    # Simulated 5-dimensional semantic embeddings:
    # Dimensions: [Royalty, Masculinity, Femininity, Technology, Age]
    vocab_embeddings = {
        "king":    np.array([0.95,  0.88, -0.10, -0.05, 0.60]),
        "queen":   np.array([0.96, -0.12,  0.90, -0.05, 0.55]),
        "man":     np.array([0.05,  0.92, -0.15,  0.02, 0.50]),
        "woman":   np.array([0.05, -0.15,  0.94,  0.02, 0.48]),
        "prince":  np.array([0.88,  0.85, -0.08, -0.02, 0.20]),
        "princess":np.array([0.89, -0.08,  0.88, -0.02, 0.18]),
        "apple":   np.array([0.02,  0.01,  0.01,  0.85, 0.10]),
        "computer":np.array([0.01,  0.02,  0.00,  0.95, 0.12]),
    }

    print("\n1. Inspecting Cosine Similarity between Words:")
    pairs = [("king", "queen"), ("king", "man"), ("man", "woman"), ("king", "computer")]
    for w1, w2 in pairs:
        sim = cosine_similarity(vocab_embeddings[w1], vocab_embeddings[w2])
        print(f"  Similarity({w1:>8}, {w2:>8}) = {sim:.4f}")

    print("\n2. Testing the Classic Vector Analogy: King - Man + Woman = ?")
    target_vector = vocab_embeddings["king"] - vocab_embeddings["man"] + vocab_embeddings["woman"]

    # Rank all candidate words by similarity to target vector
    candidates = {}
    for word, vec in vocab_embeddings.items():
        if word not in ["king", "man", "woman"]: # Exclude input terms
            candidates[word] = cosine_similarity(target_vector, vec)

    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    for rank, (word, score) in enumerate(sorted_candidates[:3], start=1):
        print(f"  Rank {rank}: {word:<10} | Cosine Similarity = {score:.4f}")

    best_match = sorted_candidates[0][0]
    print(f"\nConclusion: Vector math successfully resolved King - Man + Woman = '{best_match}'!")

if __name__ == "__main__":
    demonstrate_word_vectors()
