# Modern Machine Learning & Data Science Engineering
### Official Companion Code Repository & Datasets

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Welcome to the official source code and dataset repository for **Modern Machine Learning & Data Science Engineering: From First Principles to Real-World Solutions** by Vaibhav Rastogi.

---

## 📚 Repository Overview

This repository contains clean, modular, self-contained Python scripts and realistic sample datasets corresponding directly to each chapter of the textbook. Every script can be executed standalone without complex external cloud infrastructure.

```
├── data/                                 # Sample datasets used across chapters
│   ├── credit_risk_dataset.csv           # 1,000 credit risk profiles (Classification)
│   ├── house_prices.csv                  # 500 housing properties (Regression)
│   ├── customer_churn.csv                # 500 subscription customers (Churn modeling)
│   └── knowledge_base_rag.json           # Domain documents for Vector RAG pipeline
│
├── src/                                  # Chapter-by-chapter runnable implementations
│   ├── ch01_02_python_foundations.py     # Python syntax, collections, comprehensions
│   ├── ch03_numpy_arrays.py              # Vectorized math, standardization, matrix ops
│   ├── ch04_pandas_and_duckdb.py         # Pandas wrangling & DuckDB embedded SQL
│   ├── ch06_business_statistics.py       # t-Tests, ANOVA, Chi-Square, A/B testing
│   ├── ch08_linear_and_logistic_regression.py # OLS & Logistic Regression with Scikit-Learn
│   ├── ch09_trees_and_ensembles.py       # Random Forest, XGBoost & LightGBM
│   ├── ch10_svm_and_clustering.py        # SVM (RBF), K-Means, DBSCAN clustering
│   ├── ch11_pytorch_neural_network.py    # PyTorch MLP deep neural network
│   └── ch14_rag_and_vector_db.py         # Dense embeddings, vector similarity & RAG
│
├── capstone/                             # Comprehensive End-to-End Capstone (Appendix C)
│   └── credit_risk_engine.py             # Production credit scoring pipeline & decisioning
│
├── requirements.txt                      # Pegged Python library dependencies
└── README.md                             # Repository documentation
```

---

## 🚀 Quickstart: Running the Code Locally

### 1. Clone the Repository
```bash
git clone https://github.com/v893979/Modern_Machine_Learning_and_Data_Science_Engineering.git
cd Modern_Machine_Learning_and_Data_Science_Engineering
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Any Script
```bash
# Example 1: Run Logistic Regression Baseline
python src/ch08_linear_and_logistic_regression.py

# Example 2: Run PyTorch Deep Neural Network Training
python src/ch11_pytorch_neural_network.py

# Example 3: Run the Complete Credit Risk Capstone Engine
python capstone/credit_risk_engine.py
```

---

## 📖 Book Curriculum & Chapter Mapping

| Chapter | Title & Focus | Companion Source Code |
| :--- | :--- | :--- |
| **Ch 1–2** | Python Foundations, Lists, Dictionaries & Sets | `src/ch01_02_python_foundations.py` |
| **Ch 3** | Numerical Computing with NumPy & Vectorization | `src/ch03_numpy_arrays.py` |
| **Ch 4** | Data Manipulation with Pandas, DuckDB & Polars | `src/ch04_pandas_and_duckdb.py` |
| **Ch 5** | Exploratory Data Analysis (EDA) & Visualization | `src/ch04_pandas_and_duckdb.py` |
| **Ch 6** | Business Statistics: Probability & Hypothesis Testing | `src/ch06_business_statistics.py` |
| **Ch 7** | Text Processing & NLP Foundations | `src/ch14_rag_and_vector_db.py` |
| **Ch 8** | Core Supervised Learning: Linear & Logistic Regression | `src/ch08_linear_and_logistic_regression.py` |
| **Ch 9** | Trees, Random Forests, XGBoost & LightGBM | `src/ch09_trees_and_ensembles.py` |
| **Ch 10** | SVM Hyperplanes, K-Means & DBSCAN Clustering | `src/ch10_svm_and_clustering.py` |
| **Ch 11** | Deep Learning & Artificial Neural Networks (PyTorch) | `src/ch11_pytorch_neural_network.py` |
| **Ch 12** | Computer Vision & Convolutional Neural Networks (CNNs)| `src/ch11_pytorch_neural_network.py` |
| **Ch 13** | Machine Learning in Practice: Tuning, ONNX & MLOps | `capstone/credit_risk_engine.py` |
| **Ch 14** | Generative AI, Vector Databases & RAG Architectures | `src/ch14_rag_and_vector_db.py` |
| **App C** | Industrial Capstone: Credit Risk Assessment Engine | `capstone/credit_risk_engine.py` |

---

## ✍️ About the Author

**Vaibhav Rastogi** is Founder & Director at Textuva, an independent Enterprise Data Architect based in Melbourne, Australia, and a doctoral researcher at Walsh College of Accountancy and Business Administration focusing on Artificial Intelligence and Large-Scale Data Systems.

- **GitHub:** [@v893979](https://github.com/v893979)
- **ORCID:** [0009-0006-0900-4376](https://orcid.org/0009-0006-0900-4376)
- **Contact:** info@textuva.com

---

## 📄 License
This repository is released under the **MIT License**.
