"""
Appendix C Capstone Project: End-to-End Real-World Credit Risk Engine
Modern Machine Learning & Data Science Engineering
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve

def run_capstone_credit_engine():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "credit_risk_dataset.csv")

    print("=====================================================================")
    print("   END-TO-END CAPSTONE: REAL-WORLD CREDIT RISK ASSESSMENT ENGINE     ")
    print("=====================================================================")

    print(f"\nStep 1: Ingesting Data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Raw Ingestion: {len(df):,} applicant profiles.")

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    num_cols = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
    cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]

    print("\nStep 2: Building Scikit-Learn Preprocessing Pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    print("\nStep 3: Training Robust Ensemble Classifier...")
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1))
    ])

    model_pipeline.fit(X_train, y_train)

    print("\nStep 4: Evaluating Out-of-Sample Performance...")
    y_test_probs = model_pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_test_probs)
    print(f"Out-of-Sample Test ROC-AUC: {auc:.4f}")

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    print(f"Optimal Business Decision Threshold: {optimal_threshold:.3f} (Max F1: {f1_scores[best_idx]:.4f})")

    print("\nStep 5: Simulating Live Customer Loan Application...")
    new_applicant = pd.DataFrame([{
        "person_age": 28,
        "person_income": 82000,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 4,
        "loan_intent": "HOMEIMPROVEMENT",
        "loan_grade": "B",
        "loan_amnt": 18000,
        "loan_int_rate": 9.25,
        "loan_percent_income": 0.22,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 6
    }])

    pred_prob = model_pipeline.predict_proba(new_applicant)[0, 1]
    decision = "DECLINE / MANUAL REVIEW" if pred_prob >= optimal_threshold else "AUTOMATIC APPROVAL"

    print(f"Predicted Probability of Default: {pred_prob:.2%}")
    print(f"Automated Decision: {decision}")
    print("=====================================================================")

if __name__ == "__main__":
    run_capstone_credit_engine()
