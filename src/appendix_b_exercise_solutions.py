"""
Appendix B: End-of-Chapter Exercise Solutions & Practice Code
Modern Machine Learning & Data Science Engineering
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

def solution_ch01():
    print("--- Chapter 1 Solution: Tiered Discount Function ---")
    def get_discount(item_count: int) -> float:
        if item_count >= 10:
            return 0.20
        elif item_count >= 5:
            return 0.10
        return 0.0
    print("5 items discount:", get_discount(5))
    print("12 items discount:", get_discount(12))

def solution_ch02():
    print("\n--- Chapter 2 Solution: Temperature List Comprehension ---")
    temperatures = [22.5, 24.0, 19.8, 25.1, 21.3, 26.0, 20.4]
    above_22 = [t for t in temperatures if t > 22.0]
    print("Temperatures above 22°C:", above_22)

def solution_ch03():
    print("\n--- Chapter 3 Solution: NumPy Z-Score Normalization ---")
    data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    z_scores = (data - np.mean(data)) / np.std(data)
    print("Standardized Z-Scores:", np.round(z_scores, 3))

def solution_ch04():
    print("\n--- Chapter 4 Solution: Missing Value Imputation ---")
    df = pd.DataFrame({"age": [25, np.nan, 30, 45, np.nan, 35]})
    df["age_imputed"] = df["age"].fillna(df["age"].median())
    print(df)

def solution_ch08():
    print("\n--- Chapter 8 Solution: Simple Linear Regression ---")
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2.2, 3.9, 6.1, 8.2, 10.1])
    reg = LinearRegression().fit(X, y)
    print(f"Slope (weight): {reg.coef_[0]:.4f}, Intercept: {reg.intercept_:.4f}")

def solution_ch09():
    print("\n--- Chapter 9 Solution: Wine Dataset Random Forest ---")
    wine = load_wine()
    rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    rf.fit(wine.data, wine.target)
    top_feature_idx = np.argmax(rf.feature_importances_)
    print(f"Most Important Feature: {wine.feature_names[top_feature_idx]} ({rf.feature_importances_[top_feature_idx]:.4f})")

def solution_ch10():
    print("\n--- Chapter 10 Solution: K-Means Clustering on Wine ---")
    wine = load_wine()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(wine.data)
    print(f"Cluster Inertia (WCSS): {kmeans.inertia_:.2f}")

if __name__ == "__main__":
    solution_ch01()
    solution_ch02()
    solution_ch03()
    solution_ch04()
    solution_ch08()
    solution_ch09()
    solution_ch10()
