"""
Chapter 3: Fast Numerical Computing with NumPy
Modern Machine Learning & Data Science Engineering
"""
import numpy as np

def demonstrate_numpy_operations():
    print("--- 1. Vectorized Arithmetic & Normalization ---")
    X = np.array([
        [25, 45000, 680],
        [42, 85000, 740],
        [31, 52000, 620],
        [55, 120000, 790],
        [22, 38000, 590]
    ], dtype=float)

    print("Original Feature Matrix X (first 3 rows):\n", X[:3])

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std

    print("\nFeature Means: ", np.round(mean, 2))
    print("Feature Stds:  ", np.round(std, 2))
    print("Standardized Matrix (Mean=0, Std=1):\n", np.round(X_scaled, 3))

    print("\n--- 2. Linear Algebra: Matrix Multiplication (w^T * x) ---")
    weights = np.array([0.2, 0.5, 0.8])
    bias = -0.1
    z_scores = X_scaled @ weights + bias
    print("Linear Scores z = X @ w + b:\n", np.round(z_scores, 3))

if __name__ == "__main__":
    demonstrate_numpy_operations()
