# Modern Machine Learning & Data Science Engineering
### Official Companion Code Repository, Practice Labs, Notebooks & Datasets

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Interactive%20Notebooks-orange?logo=googlecolab)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/build-passing-brightgreen)](README.md)

Welcome to the comprehensive official companion code, dataset, and interactive Jupyter notebook repository for **Modern Machine Learning & Data Science Engineering: From First Principles to Real-World Solutions** by Vaibhav Rastogi.

Every practice lab is available both as a standalone production Python script and as an interactive, zero-setup Jupyter Notebook (`.ipynb`) located directly in `src/` and `notebooks/`. Each notebook includes a 1-click **Open in Google Colab** badge and automated dataset bootstrap code.

---

## ⚡ 1-Click Interactive Google Colab Notebooks

You do not need to install anything locally to start coding. Click any badge below to launch the corresponding chapter directly in Google Colab with free GPU/CPU acceleration:

| Chapter / Project | Topics Covered | 1-Click Colab Launch | Companion Python Script |
| :--- | :--- | :---: | :---: |
| **Ch 01 & 02: Python Foundations** | Lists, Dicts, Sets, Comprehensions, Defensive Code | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/ch01_02_python_foundations.ipynb) | [`src/ch01_02_python_foundations.py`](src/ch01_02_python_foundations.py) |
| **Ch 03: NumPy Arrays** | Vectorization, Broadcasting, 12-Month Store Sales | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/ch03_numpy_arrays.ipynb) | [`src/ch03_numpy_arrays.py`](src/ch03_numpy_arrays.py) |
| **Ch 04: Pandas & DuckDB** | Cleaning Pipelines, Imputation, SQL on DataFrames | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/ch04_pandas_and_duckdb.ipynb) | [`src/ch04_pandas_and_duckdb.py`](src/ch04_pandas_and_duckdb.py) |
| **Ch 05: EDA & Visualization** | Data Audits, IQR Outliers, Skewness, Heatmaps | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/ch05_eda_and_visualization.ipynb) | [`src/ch05_eda_and_visualization.py`](src/ch05_eda_and_visualization.py) |
| **Ch 06: Business Statistics** | Hypothesis Testing, Shapiro-Wilk, ANOVA, Chi-Square | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/ch06_business_statistics.ipynb) | [`src/ch06_business_statistics.py`](src/ch06_business_statistics.py) |
| **Ch 07: NLP & Embeddings** | TF-IDF, Sentiment Classifier, Cosine Similarity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/ch07_word2vec_and_semantic_analogies.ipynb) | [`src/ch07_word2vec_and_semantic_analogies.py`](src/ch07_word2vec_and_semantic_analogies.py) |
| **Ch 08: Ridge, Lasso & Logistic** | OLS, Ridge (L2), Lasso (L1), StandardScaler, Churn | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/ch08_linear_and_logistic_regression.ipynb) | [`src/ch08_linear_and_logistic_regression.py`](src/ch08_linear_and_logistic_regression.py) |
| **Ch 08: OLS Diagnostics** | Statsmodels, VIF Multicollinearity, Econometrics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/ch08_ols_diagnostics_admissions.ipynb) | [`src/ch08_ols_diagnostics_admissions.py`](src/ch08_ols_diagnostics_admissions.py) |
| **Ch 09: Trees, Forests & Extra Trees** | Out-of-Fold Target Encoding, Extra Trees, LightGBM | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/ch09_trees_and_ensembles.ipynb) | [`src/ch09_trees_and_ensembles.py`](src/ch09_trees_and_ensembles.py) |
| **Ch 09: Cost-Sensitive Tuning** | Hotel Cancellation Cost Curve ($150 FN vs $60 FP) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/ch09_hotel_cancellation_cost_sensitive.ipynb) | [`src/ch09_hotel_cancellation_cost_sensitive.py`](src/ch09_hotel_cancellation_cost_sensitive.py) |
| **Ch 10: SVM & PCA** | RBF Kernels, Naive Bayes Spam Classifier, PCA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/ch10_svm_and_clustering.ipynb) | [`src/ch10_svm_and_clustering.py`](src/ch10_svm_and_clustering.py) |
| **Ch 10: RFM Segmentation** | RFM Feature Engineering, K-Means, Silhouette Scores | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/ch10_rfm_customer_segmentation.ipynb) | [`src/ch10_rfm_customer_segmentation.py`](src/ch10_rfm_customer_segmentation.py) |
| **Ch 11: Deep Learning PyTorch** | Custom MLP, Dataset/DataLoader, Adam, Loss Curves | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/ch11_pytorch_neural_network.ipynb) | [`src/ch11_pytorch_neural_network.py`](src/ch11_pytorch_neural_network.py) |
| **Ch 12: Computer Vision** | 2D Spatial Convolutions, Sobel, PyTorch Conv2D | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/ch12_image_kernels_and_convolution.ipynb) | [`src/ch12_image_kernels_and_convolution.py`](src/ch12_image_kernels_and_convolution.py) |
| **Ch 13: MLOps with FastAPI** | FastAPI REST Endpoints, Pydantic v2, ONNX Runtime | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/ch13_fastapi_model_deployment.ipynb) | [`src/ch13_fastapi_model_deployment.py`](src/ch13_fastapi_model_deployment.py) |
| **Ch 14: GenAI & RAG** | In-Memory Vector DB, Cosine Retrieval, 80% Rule | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/ch14_rag_and_vector_db.ipynb) | [`src/ch14_rag_and_vector_db.py`](src/ch14_rag_and_vector_db.py) |
| **Appendix A: Math & Stats** | Matrix Math, Eigenvalues, SymPy Calculus, Monty Hall | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/appendix_a_math_and_stats.ipynb) | [`src/appendix_a_math_and_stats.py`](src/appendix_a_math_and_stats.py) |
| **Appendix B: Exercise Solutions** | Runnable Solutions for All End-of-Chapter Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/appendix_b_exercise_solutions.ipynb) | [`src/appendix_b_exercise_solutions.py`](src/appendix_b_exercise_solutions.py) |
| **Appendix C: Full Capstone** | 6-Phase Industrial Credit Risk Engine + Pipeline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/notebooks/capstone_credit_risk_engine.ipynb) | [`capstone/credit_risk_engine.py`](capstone/credit_risk_engine.py) |

---

## 🏛️ Interactive Capstone Streamlit Application

The full end-to-end industrial Credit Risk Capstone Project includes an interactive Streamlit Web Dashboard:
```bash
# Run the interactive loan officer decision interface
streamlit run capstone/app_loan_officer.py
```
This application allows loan officers to:
1. Input applicant financial attributes via interactive sliders and form controls.
2. Generate real-time risk scores and automated Approve / Deny decisions using calibrated probability thresholds.
3. Inspect interactive SHAP waterfall explanations to understand exactly which features contributed to the decision.

---

## 📁 Repository Directory Structure

```text
├── capstone/
│   ├── credit_risk_engine.py           # 6-phase industrial credit risk engine
│   ├── credit_risk_engine.ipynb        # Interactive capstone notebook
│   └── app_loan_officer.py             # Streamlit web application
├── data/
│   ├── credit_card_fraud_sample.csv    # Fraud detection benchmark dataset
│   ├── credit_risk_dataset.csv         # 1,200 loan applications for capstone
│   ├── customer_churn.csv              # Telecom customer churn dataset
│   ├── customer_rfm.csv                # E-commerce transaction histories
│   ├── hotel_reservations.csv          # Booking cancellations & cost modeling
│   ├── house_prices.csv                # King County housing sales dataset
│   ├── knowledge_base_rag.json         # RAG enterprise knowledge base
│   └── university_admissions.csv       # Graduate admissions data for OLS
├── notebooks/                          # 19 standalone Jupyter Notebooks with Colab badges
├── src/                                # Standalone runnable Python scripts and synced notebooks
│   ├── ch01_02_python_foundations.py
│   ├── ch03_numpy_arrays.py
│   ├── ch04_pandas_and_duckdb.py
│   ├── ch05_eda_and_visualization.py
│   ├── ch06_business_statistics.py
│   ├── ch07_word2vec_and_semantic_analogies.py
│   ├── ch08_linear_and_logistic_regression.py
│   ├── ch08_ols_diagnostics_admissions.py
│   ├── ch09_trees_and_ensembles.py
│   ├── ch09_hotel_cancellation_cost_sensitive.py
│   ├── ch10_svm_and_clustering.py
│   ├── ch10_rfm_customer_segmentation.py
│   ├── ch11_pytorch_neural_network.py
│   ├── ch12_image_kernels_and_convolution.py
│   ├── ch13_fastapi_model_deployment.py
│   ├── ch14_rag_and_vector_db.py
│   ├── appendix_a_math_and_stats.py
│   └── appendix_b_exercise_solutions.py
├── requirements.txt                    # Python environment dependencies
└── README.md                           # Documentation & quick start guide
```

---

## 🚀 Local Installation & Setup

If you prefer to run the scripts locally on your machine:

```bash
# 1. Clone the repository
git clone https://github.com/v893979/Modern_Machine_Learning_and_Data_Science_Engineering.git
cd Modern_Machine_Learning_and_Data_Science_Engineering

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # On Linux/macOS
venv\Scripts\activate          # On Windows

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Run any chapter script
python src/ch09_trees_and_ensembles.py
```

---

## 👨‍💻 About the Author

**Vaibhav Rastogi** is an Enterprise Data Architect and educator with over two decades of experience designing enterprise data platforms, machine learning architectures, and high-throughput analytical engines. He has founded Textuva.
