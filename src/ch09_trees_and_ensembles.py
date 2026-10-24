"""
Chapter 9: Decision Trees, Random Forests, Extra Trees, and Gradient Boosting
Modern Machine Learning & Data Science Engineering
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

def out_of_fold_target_encode(df: pd.DataFrame, cat_col: str, target_col: str, smoothing: float = 10.0, n_splits: int = 5) -> pd.DataFrame:
    """
    Out-of-Fold (OOF) Target Encoding with Bayesian smoothing.
    Eliminates data leakage when encoding high-cardinality categorical features.
    
    Formula:
        S_i = (n_i * y_bar_i + m * y_bar_global) / (n_i + m)
    """
    df = df.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    global_mean = df[target_col].mean()
    encoded_col = f"{cat_col}_encoded"
    df[encoded_col] = 0.0

    for train_idx, val_idx in kf.split(df):
        tr = df.iloc[train_idx]
        stats = tr.groupby(cat_col)[target_col].agg(['count', 'mean'])
        smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        df.iloc[val_idx, df.columns.get_loc(encoded_col)] = df.iloc[val_idx][cat_col].map(smooth).fillna(global_mean)

    return df

def run_target_encoding_demo():
    print("=" * 65)
    print("1. OUT-OF-FOLD (OOF) TARGET ENCODING DEMONSTRATION")
    print("=" * 65)
    np.random.seed(42)
    merchants = [f"MERCHANT_{i:04d}" for i in range(1, 51)]
    categories = np.random.choice(merchants, size=500)
    # Give some merchants inherently higher fraud rates
    fraud_prob = {m: (0.65 if int(m.split('_')[1]) % 5 == 0 else 0.08) for m in merchants}
    targets = [np.random.binomial(1, fraud_prob[m]) for m in categories]

    synthetic_df = pd.DataFrame({"merchant_id": categories, "is_fraud": targets})
    print(f"Dataset Shape: {synthetic_df.shape} (High-Cardinality: {synthetic_df['merchant_id'].nunique()} unique merchants)")
    print("Sample Before Encoding:")
    print(synthetic_df.head(4))

    encoded_df = out_of_fold_target_encode(synthetic_df, "merchant_id", "is_fraud", smoothing=10.0)
    print("\nSample After Out-of-Fold Bayesian Target Encoding:")
    print(encoded_df.head(4))
    print(f"Global Base Fraud Rate: {synthetic_df['is_fraud'].mean():.4f}")

def run_ensemble_benchmark():
    print("\n" + "=" * 65)
    print("2. MULTI-MODEL ENSEMBLE BENCHMARK: TREES VS FORESTS VS BOOSTING")
    print("=" * 65)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "credit_risk_dataset.csv")

    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(script_dir), "data", "credit_risk_dataset.csv")

    df = pd.read_csv(data_path)
    df_encoded = pd.get_dummies(df, columns=["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"], drop_first=True)

    X = df_encoded.drop(columns=["loan_status"])
    y = df_encoded["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    models = {
        "1. Single Decision Tree (max_depth=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
        "2. Random Forest (100 trees, max_features='sqrt')": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        "3. Extra Trees (100 extremely randomized trees)": ExtraTreesClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
    }

    try:
        import lightgbm as lgb
        models["4. LightGBM (Leaf-Wise Gradient Boosting)"] = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier
        models["5. CatBoost Classifier"] = CatBoostClassifier(iterations=150, learning_rate=0.05, depth=6, random_seed=42, verbose=0)
    except ImportError:
        pass

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        results.append((name, auc))
        print(f">> {name:<50} ROC-AUC: {auc:.4f}")

    # Top feature importances from Extra Trees
    et = models["3. Extra Trees (100 extremely randomized trees)"]
    importances = pd.Series(et.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
    print("\nTop 5 Predictive Features (Extra Trees MDI):")
    for feat, imp in importances.items():
        print(f"   {feat:<30}: {imp:.4f}")

if __name__ == "__main__":
    run_target_encoding_demo()
    run_ensemble_benchmark()
