"""
Chapter 8: Core Supervised Learning: Linear & Logistic Regression
Modern Machine Learning & Data Science Engineering
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def run_logistic_regression():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "credit_risk_dataset.csv")

    df = pd.read_csv(data_path)
    features = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income"]
    X = df[features]
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    print("--- Logistic Regression Baseline Results ---")
    print(f"ROC-AUC Score: {roc_auc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    run_logistic_regression()
