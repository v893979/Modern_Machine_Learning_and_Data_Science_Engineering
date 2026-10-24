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

You do not need to install anything locally to start coding. Click any badge below to launch an interactive, pre-configured sandbox in Google Colab:

| Chapter / Topic | Interactive Lab | Google Colab Launcher |
| :--- | :--- | :---: |
| **Ch 1–2: Python Foundations** | Core syntax, comprehensions, DTI metrics & debt analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/src/ch01_02_python_foundations.ipynb) |
| **Ch 3: NumPy Arrays & Vectorization** | Vectorized math, Z-score standardization & $z = \mathbf{X}\mathbf{w} + b$ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/src/ch03_numpy_arrays.ipynb) |
| **Ch 4: Pandas & DuckDB SQL** | High-performance wrangling & embedded DuckDB analytical SQL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/src/ch04_pandas_and_duckdb.ipynb) |
| **Ch 6: Business Statistics** | Parametric t-tests, ANOVA, Chi-Square & A/B test decisioning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/src/ch06_business_statistics.ipynb) |
| **Ch 7: Word2Vec & Vector Analogies** | Dense semantics, cosine similarity & $\mathbf{v}_{\text{King}} - \mathbf{v}_{\text{Man}} + \mathbf{v}_{\text{Woman}} = \mathbf{v}_{\text{Queen}}$ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/src/ch07_word2vec_and_semantic_analogies.ipynb) |
| **Ch 8: Logistic Regression Baseline** | Classification report, ROC-AUC curve & probability calibration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/src/ch08_linear_and_logistic_regression.ipynb) |
| **Ch 8: OLS Diagnostics & Regression** | Normal equations $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$, residuals & $R^2$ auditing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/src/ch08_ols_diagnostics_admissions.ipynb) |
| **Ch 9: Trees & Ensemble Benchmarks** | Random Forest vs LightGBM on credit underwriting data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/src/ch09_trees_and_ensembles.ipynb) |
| **Ch 9: Cost-Sensitive Loss & Thresholds** | Asymmetric hospitality cancellation loss & expected profit optimization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/src/ch09_hotel_cancellation_cost_sensitive.ipynb) |
| **Ch 10: SVM, K-Means & DBSCAN** | RBF kernel boundaries, centroid convergence & density clustering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/src/ch10_svm_and_clustering.ipynb) |
| **Ch 10: RFM Customer Segmentation** | Log feature scaling, inertia elbow heuristic & K-Means personas | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/src/ch10_rfm_customer_segmentation.ipynb) |
| **Ch 11: PyTorch Deep Learning** | Custom PyTorch `nn.Module` MLP, BCEWithLogitsLoss & Backprop | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/src/ch11_pytorch_neural_network.ipynb) |
| **Ch 12: Convolutions & Image Kernels** | Spatial 2D convolution from scratch & horizontal/vertical Sobel filters | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/src/ch12_image_kernels_and_convolution.ipynb) |
| **Ch 13: FastAPI Microservices & MLOps** | Strict Pydantic v2 schemas, health probes & production REST serving | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/src/ch13_fastapi_model_deployment.ipynb) |
| **Ch 14: RAG Architectures & Vector DB** | Dense vector similarity search, context injection & hallucination control | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/src/ch14_rag_and_vector_db.ipynb) |
| **Appendix C: Enterprise Capstone Engine** | Full production credit risk underwriting pipeline & loan decision engine | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/v893979/Modern_Machine_Learning_and_Data_Science_Engineering/blob/main/capstone/credit_risk_engine.ipynb) |

---

## 📚 Repository Architecture

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
├── src/                                          # Runnable practice implementations (.py & .ipynb)
│   ├── ch01_02_python_foundations.ipynb / .py   # Python syntax, collections, comprehensions
│   ├── ch03_numpy_arrays.ipynb / .py            # Vectorized math, standardization, matrix ops
│   ├── ch04_pandas_and_duckdb.ipynb / .py       # Pandas wrangling & DuckDB embedded SQL
│   ├── ch06_business_statistics.ipynb / .py     # t-Tests, ANOVA, Chi-Square, A/B testing
│   ├── ch07_word2vec_and_semantic_analogies.ipynb / .py # Word2Vec vectors & King - Man + Woman = Queen math
│   ├── ch08_linear_and_logistic_regression.ipynb / .py  # OLS & Logistic Regression with Scikit-Learn
│   ├── ch08_ols_diagnostics_admissions.ipynb / .py      # OLS diagnostics, normal equations & Pearson correlations
│   ├── ch09_hotel_cancellation_cost_sensitive.ipynb / .py # Cost-sensitive business loss & threshold tuning
│   ├── ch09_trees_and_ensembles.ipynb / .py     # Random Forest, XGBoost & LightGBM
│   ├── ch10_rfm_customer_segmentation.ipynb / .py # RFM log-transforms & K-Means persona clustering
│   ├── ch10_svm_and_clustering.ipynb / .py      # SVM (RBF), K-Means, DBSCAN clustering
│   ├── ch11_pytorch_neural_network.ipynb / .py  # PyTorch MLP deep neural network
│   ├── ch12_image_kernels_and_convolution.ipynb / .py # Pure 2D spatial convolution & Sobel edge filters
│   ├── ch13_fastapi_model_deployment.ipynb / .py # Production FastAPI REST microservice architecture
│   └── ch14_rag_and_vector_db.ipynb / .py       # Dense embeddings, vector similarity & RAG pipeline
│
├── capstone/                                     # Comprehensive End-to-End Capstone (Appendix C)
│   └── credit_risk_engine.ipynb / .py           # Production credit scoring pipeline & decision engine
│
├── notebooks/                                    # Dedicated Jupyter Notebooks mirror for Colab
│   └── ...                                       # 16 matching .ipynb notebooks
│
├── requirements.txt                              # Pegged Python library dependencies
└── README.md                                     # Repository documentation
```

---

## 🚀 Running the Code Locally

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

### 4. Run Scripts or Launch Jupyter Locally
```bash
# Run standalone script
python src/ch07_word2vec_and_semantic_analogies.py

# Or start Jupyter Lab to inspect notebooks locally
jupyter lab
```

---

## 📖 Book Curriculum & Practice Mapping

| Chapter | Core Concepts & Analytical Focus | Python Script | Interactive Notebook |
| :--- | :--- | :--- | :--- |
| **Ch 1–2** | Python Foundations, Lists, Dicts, Comprehensions & Error Handling | [`src/ch01_02_python_foundations.py`](src/ch01_02_python_foundations.py) | [`src/ch01_02_python_foundations.ipynb`](src/ch01_02_python_foundations.ipynb) |
| **Ch 3** | Numerical Computing with NumPy, Matrix Math & Vectorization | [`src/ch03_numpy_arrays.py`](src/ch03_numpy_arrays.py) | [`src/ch03_numpy_arrays.ipynb`](src/ch03_numpy_arrays.ipynb) |
| **Ch 4** | Data Wrangling with Pandas, DuckDB Embedded SQL & Polars | [`src/ch04_pandas_and_duckdb.py`](src/ch04_pandas_and_duckdb.py) | [`src/ch04_pandas_and_duckdb.ipynb`](src/ch04_pandas_and_duckdb.ipynb) |
| **Ch 5** | Exploratory Data Analysis (EDA), Outliers & Visual Diagnostics | [`src/ch04_pandas_and_duckdb.py`](src/ch04_pandas_and_duckdb.py) | [`src/ch04_pandas_and_duckdb.ipynb`](src/ch04_pandas_and_duckdb.ipynb) |
| **Ch 6** | Business Statistics: Probability, A/B Testing, t-Tests & Chi-Square | [`src/ch06_business_statistics.py`](src/ch06_business_statistics.py) | [`src/ch06_business_statistics.ipynb`](src/ch06_business_statistics.ipynb) |
| **Ch 7** | NLP Foundations, Dense Word Embeddings & Vector Analogies | [`src/ch07_word2vec_and_semantic_analogies.py`](src/ch07_word2vec_and_semantic_analogies.py) | [`src/ch07_word2vec_and_semantic_analogies.ipynb`](src/ch07_word2vec_and_semantic_analogies.ipynb) |
| **Ch 8** | Core Supervised Learning: Linear & Logistic Regression, OLS Auditing | [`src/ch08_ols_diagnostics_admissions.py`](src/ch08_ols_diagnostics_admissions.py) | [`src/ch08_ols_diagnostics_admissions.ipynb`](src/ch08_ols_diagnostics_admissions.ipynb) |
| **Ch 9** | Tree Ensembles, LightGBM, Cost-Sensitive Loss & Threshold Optimization | [`src/ch09_hotel_cancellation_cost_sensitive.py`](src/ch09_hotel_cancellation_cost_sensitive.py) | [`src/ch09_hotel_cancellation_cost_sensitive.ipynb`](src/ch09_hotel_cancellation_cost_sensitive.ipynb) |
| **Ch 10** | SVM Hyperplanes, K-Means & Unsupervised RFM Customer Segmentation | [`src/ch10_rfm_customer_segmentation.py`](src/ch10_rfm_customer_segmentation.py) | [`src/ch10_rfm_customer_segmentation.ipynb`](src/ch10_rfm_customer_segmentation.ipynb) |
| **Ch 11** | Deep Learning, Neural Network Math & PyTorch Architecture | [`src/ch11_pytorch_neural_network.py`](src/ch11_pytorch_neural_network.py) | [`src/ch11_pytorch_neural_network.ipynb`](src/ch11_pytorch_neural_network.ipynb) |
| **Ch 12** | Computer Vision: 2D Spatial Convolution, Sobel Filters & Pooling | [`src/ch12_image_kernels_and_convolution.py`](src/ch12_image_kernels_and_convolution.py) | [`src/ch12_image_kernels_and_convolution.ipynb`](src/ch12_image_kernels_and_convolution.ipynb) |
| **Ch 13** | Machine Learning in Practice: FastAPI REST Serving, Pydantic & MLOps | [`src/ch13_fastapi_model_deployment.py`](src/ch13_fastapi_model_deployment.py) | [`src/ch13_fastapi_model_deployment.ipynb`](src/ch13_fastapi_model_deployment.ipynb) |
| **Ch 14** | Generative AI, Dense Vector Databases & RAG Architectures | [`src/ch14_rag_and_vector_db.py`](src/ch14_rag_and_vector_db.py) | [`src/ch14_rag_and_vector_db.ipynb`](src/ch14_rag_and_vector_db.ipynb) |
| **App C** | Industrial Capstone: End-to-End Real-World Credit Risk Engine | [`capstone/credit_risk_engine.py`](capstone/credit_risk_engine.py) | [`capstone/credit_risk_engine.ipynb`](capstone/credit_risk_engine.ipynb) |

---

## ✍️ About the Author

**Vaibhav Rastogi** is Founder & Director at Textuva, an independent Enterprise Data Architect based in Melbourne, Australia, and a doctoral researcher at Walsh College of Accountancy and Business Administration specializing in Artificial Intelligence and Large-Scale Data Systems.

- **GitHub:** [@v893979](https://github.com/v893979)
- **ORCID:** [0009-0006-0900-4376](https://orcid.org/0009-0006-0900-4376)
- **Contact:** info@textuva.com

---

## 📄 License
This repository is released under the **MIT License**.
