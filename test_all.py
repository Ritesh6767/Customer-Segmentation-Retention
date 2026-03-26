"""
tests/test_all.py
=================
Unit & integration tests for the full pipeline.
Run:  pytest tests/ -v --tb=short
"""

import os, sys, json
import numpy as np
import pandas as pd
import pytest
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utils.preprocessing import (
    load_and_preprocess, OutlierClipper, RFMFeatureEngineer,
    CategoricalEncoder, NUMERIC_FEATURES
)
from models.pipeline import (
    build_pipeline, fit_and_evaluate, cluster_profile,
    tune_hyperparameters, estimate_epsilon,
    save_pipeline, load_pipeline,
)


# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def csv_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    path = str(tmp / "test.csv")

    np.random.seed(7)
    n = 400
    df = pd.DataFrame({
        "customer_id":             [f"C{i}" for i in range(n)],
        "age":                     np.random.randint(18, 70, n).astype(float),
        "annual_income":           np.random.normal(50000, 15000, n),
        "spending_score":          np.random.randint(1, 101, n).astype(float),
        "purchase_frequency":      np.random.poisson(8, n).clip(1, 40).astype(float),
        "avg_order_value":         np.random.exponential(120, n).clip(5, 900),
        "days_since_last_purchase":np.random.exponential(30, n).clip(1, 300),
        "total_orders":            np.random.poisson(20, n).clip(1, 150).astype(float),
        "return_rate":             np.random.beta(2, 10, n),
        "loyalty_years":           np.random.uniform(0, 8, n),
        "email_open_rate":         np.random.beta(3, 7, n),
        "support_tickets":         np.random.poisson(1.5, n).clip(0, 20).astype(float),
        "promo_usage_rate":        np.random.beta(2, 5, n),
        "preferred_channel":       np.random.choice(["online","mobile","store"], n),
        "region":                  np.random.choice(["North","South","East","West","Central"], n),
    })
    df.loc[np.random.choice(df.index, 15), "age"] = np.nan
    df.loc[np.random.choice(df.index, 10), "annual_income"] = np.nan
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def preprocessed(csv_path):
    return load_and_preprocess(csv_path)


# ─────────────────────────────────────────────────────────
# Preprocessing Tests
# ─────────────────────────────────────────────────────────

class TestPreprocessing:

    def test_returns_correct_types(self, preprocessed):
        raw, X, feats = preprocessed
        assert isinstance(raw, pd.DataFrame)
        assert isinstance(X, np.ndarray)
        assert isinstance(feats, list)

    def test_no_nans_after_preprocessing(self, preprocessed):
        _, X, _ = preprocessed
        assert not np.isnan(X).any(), "NaN values remain after preprocessing"

    def test_rfm_features_present(self, preprocessed):
        _, _, feats = preprocessed
        for f in ["recency_score", "monetary_score", "clv_estimate", "engagement_ratio"]:
            assert f in feats, f"Missing RFM feature: {f}"

    def test_shape_reasonable(self, preprocessed):
        _, X, feats = preprocessed
        assert X.shape[1] == len(feats)
        assert X.shape[0] == 400

    def test_outlier_clipper(self):
        data = pd.DataFrame({"a": [1.0, 2.0, 3.0, 9999.0], "b": [5.0, 5.0, 5.0, 5.0]})
        clipper = OutlierClipper(factor=1.5, numeric_cols=["a"])
        clipped = clipper.fit_transform(data)
        assert float(clipped["a"].max()) < 9999.0

    def test_rfm_engineer_adds_cols(self):
        df = pd.DataFrame({
            "days_since_last_purchase": [10.0, 30.0],
            "purchase_frequency": [5.0, 2.0],
            "avg_order_value": [100.0, 200.0],
            "loyalty_years": [3.0, 1.0],
            "return_rate": [0.05, 0.1],
        })
        rfm = RFMFeatureEngineer()
        out = rfm.fit_transform(df)
        for col in ["recency_score", "monetary_score", "clv_estimate"]:
            assert col in out.columns

    def test_categorical_encoder(self):
        df = pd.DataFrame({"preferred_channel": ["online","mobile","store"],
                           "region": ["North","East","South"]})
        enc = CategoricalEncoder(cat_cols=["preferred_channel","region"])
        out = enc.fit_transform(df)
        assert out["preferred_channel"].dtype in [np.int32, np.int64, int]


# ─────────────────────────────────────────────────────────
# Pipeline Tests
# ─────────────────────────────────────────────────────────

class TestPipeline:

    def test_pipeline_steps_with_pca(self):
        p = build_pipeline(use_pca=True)
        names = [s[0] for s in p.steps]
        assert names == ["scaler", "pca", "dbscan"]

    def test_pipeline_steps_no_pca(self):
        p = build_pipeline(use_pca=False)
        names = [s[0] for s in p.steps]
        assert names == ["scaler", "dbscan"]

    def test_fit_evaluate_returns_dict(self, preprocessed, tmp_path):
        _, X, feats = preprocessed
        result = fit_and_evaluate(
            X, feats, eps=1.2, min_samples=5,
            use_pca=True, n_components=4,
            output_dir=str(tmp_path),
        )
        for key in ["pipeline", "labels", "metrics", "X_transformed"]:
            assert key in result

    def test_labels_length_matches_input(self, preprocessed, tmp_path):
        _, X, feats = preprocessed
        result = fit_and_evaluate(X, feats, eps=1.2, min_samples=5,
                                   use_pca=True, n_components=4,
                                   output_dir=str(tmp_path))
        assert len(result["labels"]) == X.shape[0]

    def test_metrics_keys(self, preprocessed, tmp_path):
        _, X, feats = preprocessed
        result = fit_and_evaluate(X, feats, eps=1.2, min_samples=5,
                                   use_pca=True, n_components=4,
                                   output_dir=str(tmp_path))
        for k in ["n_clusters", "n_noise", "noise_pct"]:
            assert k in result["metrics"]

    def test_save_load_pipeline(self, preprocessed, tmp_path):
        _, X, feats = preprocessed
        result = fit_and_evaluate(X, feats, eps=1.2, min_samples=5,
                                   use_pca=True, n_components=4,
                                   output_dir=str(tmp_path))
        model_path = str(tmp_path / "test_pipe.pkl")
        save_pipeline(result["pipeline"], model_path)
        loaded = load_pipeline(model_path)
        assert loaded is not None

    def test_cluster_profile_returns_dataframe(self, preprocessed, tmp_path):
        raw, X, feats = preprocessed
        result = fit_and_evaluate(X, feats, eps=1.2, min_samples=5,
                                   use_pca=True, n_components=4,
                                   output_dir=str(tmp_path))
        num_cols = [c for c in NUMERIC_FEATURES if c in raw.columns]
        profile  = cluster_profile(raw, result["labels"], num_cols)
        assert isinstance(profile, pd.DataFrame)


# ─────────────────────────────────────────────────────────
# Hyperparameter Tuning Tests
# ─────────────────────────────────────────────────────────

class TestTuning:

    def test_tuning_returns_dataframe(self, preprocessed):
        _, X, _ = preprocessed
        df = tune_hyperparameters(
            X,
            eps_values=[0.5, 1.0, 1.5],
            min_samples_values=[5, 10],
            use_pca=True,
            n_components=4,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6   # 3 eps × 2 min_samples

    def test_tuning_has_required_columns(self, preprocessed):
        _, X, _ = preprocessed
        df = tune_hyperparameters(X, eps_values=[1.0], min_samples_values=[5],
                                   use_pca=True, n_components=4)
        for col in ["eps","min_samples","n_clusters","silhouette_score"]:
            assert col in df.columns

    def test_epsilon_estimation_returns_float(self, preprocessed):
        from sklearn.preprocessing import StandardScaler
        _, X, _ = preprocessed
        X_s = StandardScaler().fit_transform(X)
        eps = estimate_epsilon(X_s, k=5)
        assert isinstance(eps, float)
        assert eps > 0
