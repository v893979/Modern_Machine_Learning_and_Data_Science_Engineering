"""
Chapter 9: Tree Ensembles: Random Forest, XGBoost, LightGBM, and CatBoost
Modern Machine Learning & Data Science Engineering
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def run_ensemble_benchmark():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "credit_risk_dataset.csv")

    df = pd.read_csv(data_path)
    df_encoded = pd.get_dummies(df, columns=["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"], drop_first=True)

    X = df_encoded.drop(columns=["loan_status"])
    y = df_encoded["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    print("--- 1. Random Forest Classifier ---")
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    print(f"Random Forest ROC-AUC: {roc_auc_score(y_test, rf_probs):.4f}")

    print("\n--- 2. LightGBM / XGBoost Gradient Boosting ---")
    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
        lgb_model.fit(X_train, y_train)
        lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
        print(f"LightGBM Classifier ROC-AUC: {roc_auc_score(y_test, lgb_probs):.4f}")
    except ImportError:
        print("LightGBM not installed. Install via: pip install lightgbm")

if __name__ == "__main__":
    run_ensemble_benchmark()
