# ⚛️ Customer Segmentation & Retention Analysis
### Customer Segmentation Portfolio — Data Science & Machine Learning

---

## Project Overview

An end-to-end machine learning system that applies **DBSCAN density-based clustering** to segment customers into distinct behavioural profiles, enabling targeted retention strategies. The system is deployed as an interactive **Streamlit dashboard** inside a **Docker** container.

---

## Architecture

```
customer_segmentation/
│
├── data/
│   └── generate_data.py          # Synthetic 3000-customer dataset (RFM + demographics)
│
├── utils/
│   └── preprocessing.py          # Custom sklearn transformers:
│                                 #   CategoricalEncoder, OutlierClipper,
│                                 #   RFMFeatureEngineer, MedianImputer
│
├── models/
│   └── pipeline.py               # sklearn Pipeline: StandardScaler → PCA → DBSCAN
│                                 # + hyperparameter tuning + evaluation metrics
│
├── streamlit_app/
│   └── app.py                    # 6-tab Streamlit dashboard
│
├── tests/
│   └── test_all.py               # 20+ pytest unit & integration tests
│
├── train.py                      # Full orchestration (data → tune → fit → save)
├── Dockerfile                    # Containerised deployment
├── docker-compose.yml            # Multi-service compose
└── requirements.txt
```

---

## Pipeline Steps

### 1 · Data Preprocessing
| Transformer | What it does |
|---|---|
| `CategoricalEncoder` | LabelEncodes `preferred_channel`, `region` |
| `SimpleImputer(median)` | Fills NaN values with feature medians |
| `OutlierClipper(IQR×1.5)` | Caps extreme values to [Q1−1.5·IQR, Q3+1.5·IQR] |
| `RFMFeatureEngineer` | Derives `recency_score`, `monetary_score`, `clv_estimate`, `engagement_ratio`, `churn_risk_score` |

### 2 · StandardScaler
Zero-mean unit-variance normalisation. Critical for DBSCAN: the algorithm is distance-based, so unscaled high-magnitude features (e.g. `annual_income`) would dominate the metric.

### 3 · Pipeline
```python
Pipeline([
    ("scaler",  StandardScaler()),
    ("pca",     PCA(n_components=8)),
    ("dbscan",  DBSCAN(eps=ε*, min_samples=10)),
])
```
Encapsulates all transforms — no data leakage, single `fit_transform` call, trivial serialisation.

### 4 · Hyperparameter Tuning
Grid search over `eps × min_samples` scored by **Silhouette Index**:
- Auto-generates `eps` candidates via **k-NN distance elbow** heuristic
- Returns full results DataFrame + best configuration
- Visualises the hyperparameter space as a scatter heatmap

### 5 · DBSCAN Clustering
**Why DBSCAN over K-Means?**
- No need to predefine number of clusters
- Discovers arbitrary-shaped clusters
- Marks low-density outliers as noise (label = −1)
- Robust to the curse of dimensionality (after PCA)

**Evaluation Metrics:** Silhouette Score · Davies-Bouldin Index · Calinski-Harabasz Score

### 6 · Streamlit Dashboard
6-tab interactive interface: Overview · Data Explorer · Train & Tune · Cluster Analysis · Predict · Retention Intel

### 7 · Docker
```bash
docker compose up --build          # Full build + train + serve
docker compose --profile train up  # Train only
```

---

## Quick Start

```bash
# Local
pip install -r requirements.txt
python train.py
streamlit run streamlit_app/app.py

# Docker (recommended)
docker compose up --build
# Open: http://localhost:8501

# Tests
pytest tests/ -v
```

---

## Customer Segments

| Segment | Description | Retention Strategy |
|---|---|---|
| Champions | High AOV, high frequency, loyal | VIP rewards, referral programme |
| Loyal High-Value | Long tenure, high CLV | Personalised upsell, account manager |
| At-Risk Churners | High recency gap, declining frequency | Win-back campaign, 15-20% discount |
| Budget Shoppers | Low AOV, high frequency | Bundle deals, loyalty points |
| New Prospects | Low loyalty_years | Onboarding drip, first-purchase offer |
| Occasional Buyers | Sporadic behaviour | Flash-sale alerts, seasonal nudges |
| Noise / Outliers | Atypical – possible fraud | Manual review, data enrichment |

---

## Tech Stack
`Python 3.11` · `scikit-learn` · `pandas` · `numpy` · `Streamlit` · `Docker` · `matplotlib` · `joblib` · `pytest`
