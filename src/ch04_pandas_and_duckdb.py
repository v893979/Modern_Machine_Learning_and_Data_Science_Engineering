"""
Chapter 4: Data Manipulation with Pandas, DuckDB Embedded SQL & Polars
Modern Machine Learning & Data Science Engineering
"""
import os
import pandas as pd

def demonstrate_pandas_and_sql():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "credit_risk_dataset.csv")

    print(f"--- 1. Loading Credit Risk Dataset from: {data_path} ---")
    df = pd.read_csv(data_path)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst 3 rows:")
    print(df[["person_age", "person_income", "loan_intent", "loan_amnt", "loan_status"]].head(3))

    print("\n--- 2. GroupBy Aggregations in Pandas ---")
    summary = df.groupby("loan_intent").agg(
        avg_income=("person_income", "mean"),
        avg_loan=("loan_amnt", "mean"),
        default_rate=("loan_status", "mean"),
        applicant_count=("loan_status", "count")
    ).reset_index()
    print(summary.to_string(index=False))

    print("\n--- 3. Executing Embedded SQL Query via DuckDB (if installed) ---")
    try:
        import duckdb
        query = """
        SELECT loan_intent,
               COUNT(*) as total_applicants,
               ROUND(AVG(loan_amnt), 2) as mean_loan,
               ROUND(AVG(loan_status) * 100, 2) as default_rate_pct
        FROM df
        GROUP BY loan_intent
        ORDER BY default_rate_pct DESC
        """
        sql_result = duckdb.query(query).to_df()
        print(sql_result.to_string(index=False))
    except ImportError:
        print("DuckDB not installed. To run embedded SQL: pip install duckdb")

if __name__ == "__main__":
    demonstrate_pandas_and_sql()
