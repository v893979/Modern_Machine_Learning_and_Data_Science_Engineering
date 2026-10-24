"""
Chapter 8: OLS Regression Diagnostics & Multicollinearity (VIF)
Modern Machine Learning & Data Science Engineering
"""
import os
import csv
import numpy as np

def run_ols_diagnostics():
    print("=====================================================================")
    print("   MULTIVARIATE OLS REGRESSION & STATISTICAL AUDITING                ")
    print("=====================================================================")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "university_admissions.csv")

    features = ["gre_score", "toefl_score", "univ_rating", "sop", "lor", "cgpa", "research"]
    X_list = []
    y_list = []

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X_list.append([float(row[feat]) for feat in features])
            y_list.append(float(row["chance_of_admit"]))

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"Loaded {len(X)} applicant records with {len(features)} predictor features.")

    # 1. Feature Correlation with Target
    print("\n--- 1. Feature Correlation with Target (Chance of Admit) ---")
    for idx, feat in enumerate(features):
        r = np.corrcoef(X[:, idx], y)[0, 1]
        print(f"  {feat:<15} : Pearson r = {r:+.4f}")

    # 2. Fit Ordinary Least Squares via Normal Equation: beta = (X^T X)^-1 X^T y
    X_design = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y

    print("\n--- 2. Estimated OLS Model Coefficients ---")
    print(f"  Intercept      : {beta[0]:+.4f}")
    for name, coef in zip(features, beta[1:]):
        print(f"  {name:<15} : {coef:+.5f}")

    # 3. Residual Diagnostics
    y_pred = X_design @ beta
    residuals = y - y_pred
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum(residuals**2)
    r_squared = 1.0 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))

    print(f"\nModel R-Squared (Variance Explained): {r_squared:.4f}")
    print(f"Residual Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Residual Mean (Expect ~0): {np.mean(residuals):.2e}")

if __name__ == "__main__":
    run_ols_diagnostics()
