"""
Chapter 5: Exploratory Data Analysis (EDA) & Data Visualization
Modern Machine Learning & Data Science Engineering
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless backend for automation
import matplotlib.pyplot as plt
import seaborn as sns

def load_housing_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "house_prices.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(script_dir), "data", "house_prices.csv")
    return pd.read_csv(data_path)

def audit_data_quality(df: pd.DataFrame):
    print("=" * 65)
    print("1. DATA QUALITY & INTEGRITY AUDIT")
    print("=" * 65)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nSummary Statistics:")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]])

    # Missing Value Audit
    missing = df.isnull().sum()
    print("\nMissing Values per Column:")
    print(missing[missing > 0] if missing.sum() > 0 else "Zero missing values detected.")

    # Physical Boundary Validation
    invalid_sqft = (df["sqft_living"] <= 0).sum()
    invalid_price = (df["price"] <= 0).sum()
    print(f"Boundary Audit: {invalid_sqft} invalid sqft rows, {invalid_price} invalid price rows.")

def detect_outliers_iqr(df: pd.DataFrame, column: str):
    print("\n" + "=" * 65)
    print(f"2. OUTLIER DETECTION (INTERQUARTILE RANGE - IQR): {column.upper()}")
    print("=" * 65)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Q1 (25th percentile): ${Q1:,.2f}")
    print(f"Q3 (75th percentile): ${Q3:,.2f}")
    print(f"IQR:                  ${IQR:,.2f}")
    print(f"Lower Bound:          ${lower_bound:,.2f}")
    print(f"Upper Bound:          ${upper_bound:,.2f}")
    print(f"Detected Outliers:    {len(outliers)} rows ({len(outliers)/len(df)*100:.1f}% of data)")
    return lower_bound, upper_bound

def run_eda_visualizations(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("3. GENERATING STATISTICAL VISUALIZATIONS")
    print("=" * 65)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_plots")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Price Distribution & Skewness
    plt.figure(figsize=(8, 4))
    sns.histplot(df["price"], kde=True, color="#2563eb", bins=25)
    plt.title(f"House Price Distribution (Skewness: {df['price'].skew():.2f})")
    plt.xlabel("Price ($)")
    plt.tight_layout()
    p1 = os.path.join(output_dir, "price_distribution.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f">> Saved: {p1}")

    # 2. Correlation Matrix Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(7, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("Pearson Correlation Matrix Heatmap")
    plt.tight_layout()
    p2 = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f">> Saved: {p2}")

    # 3. Categorical Boxplot: Price by Zip Code
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="zip_code", y="price", data=df)
    plt.title("House Prices by Zip Code (Boxplot)")
    plt.tight_layout()
    p3 = os.path.join(output_dir, "zipcode_boxplot.png")
    plt.savefig(p3, dpi=150)
    plt.close()
    print(f">> Saved: {p3}")

    print("\n[SUCCESS] Chapter 5 EDA & Visualizations Completed Successfully!")

if __name__ == "__main__":
    housing_df = load_housing_data()
    audit_data_quality(housing_df)
    detect_outliers_iqr(housing_df, "price")
    run_eda_visualizations(housing_df)
