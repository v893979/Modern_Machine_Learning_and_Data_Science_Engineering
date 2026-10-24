# Modern Machine Learning & Data Science Engineering
### Official Companion Code Repository, Practice Labs & Datasets

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/build-passing-brightgreen)](README.md)

Welcome to the comprehensive official source code, dataset, and hands-on lab repository for **Modern Machine Learning & Data Science Engineering: From First Principles to Real-World Solutions** by Vaibhav Rastogi.

---

## 📚 Repository Architecture

This repository contains clean, modular, production-standard Python scripts and realistic enterprise datasets designed for active learning, experimentation, and industry practice.

```
├── data/                                         # Production-grade enterprise sample datasets
│   ├── credit_risk_dataset.csv                   # 1,000 credit loan profiles (Classification)
│   ├── university_admissions.csv                 # 500 graduate admissions records (Multivariate OLS)
│   ├── hotel_reservations.csv                    # 800 hospitality bookings (Cost-sensitive classification)
│   ├── customer_rfm.csv                          # 600 customer transaction profiles (RFM Segmentation)
│   ├── credit_card_fraud_sample.csv              # 1,000 transaction vectors (Imbalanced deep learning)
│   ├── house_prices.csv                          # 500 residential properties (Regression)
│   ├── customer_churn.csv                        # 500 telecom subscriber logs (Customer churn)
│   └── knowledge_base_rag.json                   # Institutional knowledge manual (Vector RAG)
│
├── src/                                          # Chapter-by-chapter runnable practice implementations
│   ├── ch01_02_python_foundations.py             # Python syntax, collections, comprehensions
│   ├── ch03_numpy_arrays.py                      # Vectorized math, standardization, matrix ops
│   ├── ch04_pandas_and_duckdb.py                 # Pandas wrangling & DuckDB embedded SQL
│   ├── ch06_business_statistics.py               # t-Tests, ANOVA, Chi-Square, A/B testing
│   ├── ch07_word2vec_and_semantic_analogies.py   # Word2Vec vectors & King - Man + Woman = Queen math
│   ├── ch08_linear_and_logistic_regression.py    # OLS & Logistic Regression with Scikit-Learn
│   ├── ch08_ols_diagnostics_admissions.py        # OLS diagnostics, normal equations & Pearson correlations
│   ├── ch09_trees_and_ensembles.py               # Random Forest, XGBoost & LightGBM
│   ├── ch09_hotel_cancellation_cost_sensitive.py # Cost-sensitive business loss & threshold tuning
│   ├── ch10_svm_and_clustering.py                # SVM (RBF), K-Means, DBSCAN clustering
│   ├── ch10_rfm_customer_segmentation.py         # RFM log-transforms & K-Means persona clustering
│   ├── ch11_pytorch_neural_network.py            # PyTorch MLP deep neural network
│   ├── ch12_image_kernels_and_convolution.py     # Pure 2D spatial convolution & Sobel edge filters
│   ├── ch13_fastapi_model_deployment.py          # Production FastAPI REST microservice architecture
│   └── ch14_rag_and_vector_db.py                 # Dense embeddings, vector similarity & RAG pipeline
│
├── capstone/                                     # Comprehensive End-to-End Capstone (Appendix C)
│   └── credit_risk_engine.py                     # Production credit scoring pipeline & decision engine
│
├── requirements.txt                              # Pegged Python library dependencies
└── README.md                                     # Repository documentation
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

### 4. Run Any Practice Script
```bash
# Word2Vec semantic vector analogy math (King - Man + Woman = Queen)
python src/ch07_word2vec_and_semantic_analogies.py

# Multivariate OLS regression diagnostics on university admissions
python src/ch08_ols_diagnostics_admissions.py

# Cost-sensitive classification & threshold tuning on hotel cancellations
python src/ch09_hotel_cancellation_cost_sensitive.py

# Unsupervised RFM customer segmentation clustering
python src/ch10_rfm_customer_segmentation.py

# Computer Vision: Spatial 2D convolution & Sobel edge filtering
python src/ch12_image_kernels_and_convolution.py

# End-to-End Credit Risk Capstone Engine
python capstone/credit_risk_engine.py
```

---

## 📖 Book Curriculum & Practice Mapping

| Chapter | Core Concepts & Analytical Focus | Companion Source Code |
| :--- | :--- | :--- |
| **Ch 1–2** | Python Foundations, Lists, Dicts, Comprehensions & Error Handling | `src/ch01_02_python_foundations.py` |
| **Ch 3** | Numerical Computing with NumPy, Matrix Math & Vectorization | `src/ch03_numpy_arrays.py` |
| **Ch 4** | Data Wrangling with Pandas, DuckDB Embedded SQL & Polars | `src/ch04_pandas_and_duckdb.py` |
| **Ch 5** | Exploratory Data Analysis (EDA), Outliers & Visual Diagnostics | `src/ch04_pandas_and_duckdb.py` |
| **Ch 6** | Business Statistics: Probability, A/B Testing, t-Tests & Chi-Square | `src/ch06_business_statistics.py` |
| **Ch 7** | NLP Foundations, Dense Word Embeddings & Vector Analogies | `src/ch07_word2vec_and_semantic_analogies.py` |
| **Ch 8** | Core Supervised Learning: Linear & Logistic Regression, OLS Auditing | `src/ch08_ols_diagnostics_admissions.py` |
| **Ch 9** | Tree Ensembles, LightGBM, Cost-Sensitive Loss & Threshold Optimization | `src/ch09_hotel_cancellation_cost_sensitive.py` |
| **Ch 10** | SVM Hyperplanes, K-Means & Unsupervised RFM Customer Segmentation | `src/ch10_rfm_customer_segmentation.py` |
| **Ch 11** | Deep Learning, Neural Network Math & PyTorch Architecture | `src/ch11_pytorch_neural_network.py` |
| **Ch 12** | Computer Vision: 2D Spatial Convolution, Sobel Filters & Pooling | `src/ch12_image_kernels_and_convolution.py` |
| **Ch 13** | Machine Learning in Practice: FastAPI REST Serving, Pydantic & MLOps | `src/ch13_fastapi_model_deployment.py` |
| **Ch 14** | Generative AI, Dense Vector Databases & RAG Architectures | `src/ch14_rag_and_vector_db.py` |
| **App C** | Industrial Capstone: End-to-End Real-World Credit Risk Engine | `capstone/credit_risk_engine.py` |

---

## ✍️ About the Author

**Vaibhav Rastogi** is Founder & Director at Textuva, an independent Enterprise Data Architect based in Melbourne, Australia, and a doctoral researcher at Walsh College of Accountancy and Business Administration specializing in Artificial Intelligence and Large-Scale Data Systems.

- **GitHub:** [@v893979](https://github.com/v893979)
- **ORCID:** [0009-0006-0900-4376](https://orcid.org/0009-0006-0900-4376)
- **Contact:** info@textuva.com

---

## 📄 License
This repository is released under the **MIT License**.
