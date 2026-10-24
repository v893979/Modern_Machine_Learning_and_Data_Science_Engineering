"""
Chapter 10: Advanced Classical Algorithms: SVM, KNN, K-Means & DBSCAN
Modern Machine Learning & Data Science Engineering
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons

def demonstrate_svm_and_clustering():
    print("--- 1. Support Vector Machine (RBF Kernel) ---")
    X_moon, y_moon = make_moons(n_samples=300, noise=0.15, random_state=42)
    svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm.fit(X_moon, y_moon)
    print(f"Trained SVM on Non-Linear Moons. Number of Support Vectors: {len(svm.support_)}")

    print("\n--- 2. K-Means Clustering & Convergence ---")
    X_blobs, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=42)
    kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_blobs)
    print(f"K-Means converged in {kmeans.n_iter_} iterations.")
    print("Final Cluster Centroid Coordinates:\n", np.round(kmeans.cluster_centers_, 3))

    print("\n--- 3. DBSCAN Density-Based Clustering ---")
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    db_labels = dbscan.fit_predict(X_moon)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = list(db_labels).count(-1)
    print(f"DBSCAN discovered {n_clusters} clusters with {n_noise} noise outlier points.")

if __name__ == "__main__":
    demonstrate_svm_and_clustering()
