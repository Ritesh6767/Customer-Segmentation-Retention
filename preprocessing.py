"""
preprocessing.py
================
Production-grade preprocessing for customer segmentation:

  1. CategoricalEncoder     - LabelEncodes categorical features
  2. MedianImputer          - Fills NaNs with feature medians
  3. OutlierClipper         - IQR-based outlier capping
  4. RFMFeatureEngineer     - Derives recency/frequency/monetary signals + CLV
  5. load_and_preprocess()  - Orchestrates all steps & returns (raw_df, X, features)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Feature groups
# ─────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "age", "annual_income", "spending_score",
    "purchase_frequency", "avg_order_value",
    "days_since_last_purchase", "total_orders",
    "return_rate", "loyalty_years",
    "email_open_rate", "support_tickets", "promo_usage_rate",
]
CATEGORICAL_FEATURES = ["preferred_channel", "region"]
RFM_FEATURES         = ["recency_score", "monetary_score", "clv_estimate", "engagement_ratio", "churn_risk_score"]


# ─────────────────────────────────────────────────────────
# Custom Transformers
# ─────────────────────────────────────────────────────────

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Label-encodes a fixed list of categorical columns."""

    def __init__(self, cat_cols: list):
        self.cat_cols = cat_cols
        self._encoders = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in self.cat_cols:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str).fillna("unknown"))
                self._encoders[col] = le
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for col, le in self._encoders.items():
            if col in X.columns:
                known = set(le.classes_)
                X[col] = X[col].astype(str).apply(
                    lambda v: v if v in known else le.classes_[0]
                )
                X[col] = le.transform(X[col])
        return X


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clips numeric columns to [Q1 − k·IQR, Q3 + k·IQR]."""

    def __init__(self, factor: float = 1.5, numeric_cols: list = None):
        self.factor      = factor
        self.numeric_cols = numeric_cols
        self._bounds     = {}

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.numeric_cols or X.select_dtypes(include=np.number).columns.tolist()
        for col in cols:
            if col in X.columns:
                q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
                iqr = q3 - q1
                self._bounds[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in self._bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lo, hi)
        return X


class RFMFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives composite RFM features and a churn-risk proxy."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        eps = 1e-9

        if "days_since_last_purchase" in X.columns:
            X["recency_score"] = 1.0 / (X["days_since_last_purchase"] + 1)

        if {"avg_order_value", "purchase_frequency"}.issubset(X.columns):
            X["monetary_score"] = X["avg_order_value"] * X["purchase_frequency"]

        if {"monetary_score", "loyalty_years"}.issubset(X.columns):
            X["clv_estimate"] = X["monetary_score"] * (X["loyalty_years"] + eps)

        if {"purchase_frequency", "days_since_last_purchase"}.issubset(X.columns):
            X["engagement_ratio"] = X["purchase_frequency"] / (X["days_since_last_purchase"] + 1)

        # Churn risk proxy: high recency gap + high return rate + low freq
        if {"days_since_last_purchase", "return_rate", "purchase_frequency"}.issubset(X.columns):
            X["churn_risk_score"] = (
                (X["days_since_last_purchase"] / (X["days_since_last_purchase"].max() + eps))
                + X["return_rate"]
                - (X["purchase_frequency"] / (X["purchase_frequency"].max() + eps))
            ).clip(0, 2)

        logger.info(f"[RFM] Features added: {RFM_FEATURES}")
        return X


# ─────────────────────────────────────────────────────────
# Master preprocessing function
# ─────────────────────────────────────────────────────────

def load_and_preprocess(filepath: str) -> tuple:
    """
    Full preprocessing pipeline:
      load CSV → encode categoricals → impute → clip outliers → RFM engineering
    Returns
    -------
    raw_df        : pd.DataFrame  - original data + cluster column placeholder
    X             : np.ndarray    - preprocessed feature matrix
    feature_names : list[str]     - ordered feature names matching X columns
    """
    logger.info(f"[Preprocessing] Loading: {filepath}")
    df     = pd.read_csv(filepath)
    raw_df = df.copy()
    logger.info(f"  Shape={df.shape}  |  Missing={df.isnull().sum().sum()}")

    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X_df = df[feature_cols].copy()

    # Step 1 – Encode categoricals
    cat_enc = CategoricalEncoder(cat_cols=CATEGORICAL_FEATURES)
    X_df    = cat_enc.fit_transform(X_df)
    logger.info("[Preprocessing] Step 1: Categorical encoding ✓")

    # Step 2 – Impute missing values
    imputer  = SimpleImputer(strategy="median")
    X_arr    = imputer.fit_transform(X_df)
    X_df     = pd.DataFrame(X_arr, columns=feature_cols)
    logger.info("[Preprocessing] Step 2: Median imputation ✓")

    # Step 3 – Clip outliers
    clipper = OutlierClipper(factor=1.5, numeric_cols=NUMERIC_FEATURES)
    X_df    = clipper.fit_transform(X_df)
    logger.info("[Preprocessing] Step 3: Outlier clipping ✓")

    # Step 4 – RFM feature engineering
    rfm_eng = RFMFeatureEngineer()
    X_df    = rfm_eng.fit_transform(X_df)
    logger.info("[Preprocessing] Step 4: RFM engineering ✓")

    all_features  = feature_cols + [f for f in RFM_FEATURES if f in X_df.columns]
    X_final       = X_df[all_features].values

    logger.info(f"[Preprocessing] Final matrix: {X_final.shape} | Features: {len(all_features)}")
    return raw_df, X_final, all_features


if __name__ == "__main__":
    from data.generate_data import generate_customers
    generate_customers()
    raw, X, feats = load_and_preprocess("data/customers_raw.csv")
    print(f"Output shape: {X.shape}")
    print(f"Features ({len(feats)}): {feats}")
