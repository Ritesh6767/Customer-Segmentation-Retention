"""
train.py  –  End-to-End Training Orchestrator
==============================================
Steps:
  1. Generate / load data
  2. Preprocess (encode, impute, clip, RFM)
  3. Hyperparameter tuning (eps × min_samples grid)
  4. Fit best pipeline  (StandardScaler → PCA → DBSCAN)
  5. Evaluate & profile clusters
  6. Persist artefacts
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate_data        import generate_customers
from utils.preprocessing       import load_and_preprocess, NUMERIC_FEATURES
from models.pipeline import (
    tune_hyperparameters,
    fit_and_evaluate,
    cluster_profile,
    save_pipeline,
    SEGMENT_MAP,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR     = "outputs"
MODEL_PATH     = "models_saved/pipeline.pkl"
METRICS_PATH   = "outputs/metrics.json"
TUNING_PATH    = "outputs/tuning_results.csv"
PROFILES_PATH  = "outputs/cluster_profiles.json"
DATA_PATH      = "data/customers_raw.csv"


def main(
    run_tuning:     bool = True,
    use_pca:        bool = True,
    n_components:   int  = 8,
    min_samples:    int  = 10,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("models_saved", exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────
    logger.info("━"*55)
    logger.info("STEP 1 │ Data Generation")
    logger.info("━"*55)
    if not os.path.exists(DATA_PATH):
        generate_customers(save_path=DATA_PATH)
    else:
        logger.info(f"Using existing dataset: {DATA_PATH}")

    # ── 2. Preprocessing ─────────────────────────────────
    logger.info("━"*55)
    logger.info("STEP 2 │ Preprocessing")
    logger.info("━"*55)
    raw_df, X, feature_names = load_and_preprocess(DATA_PATH)

    # ── 3. Hyperparameter Tuning ─────────────────────────
    best_eps = None
    if run_tuning:
        logger.info("━"*55)
        logger.info("STEP 3 │ Hyperparameter Tuning")
        logger.info("━"*55)
        tuning_df = tune_hyperparameters(
            X,
            use_pca=use_pca,
            n_components=n_components,
        )
        tuning_df.to_csv(TUNING_PATH, index=False)
        logger.info(f"Tuning results saved → {TUNING_PATH}")

        valid = tuning_df.dropna(subset=["silhouette_score"]).sort_values(
            "silhouette_score", ascending=False
        )
        if len(valid):
            best       = valid.iloc[0]
            best_eps   = float(best["eps"])
            min_samples= int(best["min_samples"])
            logger.info(
                f"Best config: eps={best_eps}, min_samples={min_samples}, "
                f"clusters={int(best['n_clusters'])}, "
                f"silhouette={best['silhouette_score']}"
            )
    else:
        logger.info("STEP 3 │ Skipping tuning (run_tuning=False)")

    # ── 4. Fit Pipeline ───────────────────────────────────
    logger.info("━"*55)
    logger.info("STEP 4 │ Fitting Pipeline  (Scaler → PCA → DBSCAN)")
    logger.info("━"*55)
    result = fit_and_evaluate(
        X,
        feature_names=feature_names,
        eps=best_eps,
        min_samples=min_samples,
        use_pca=use_pca,
        n_components=n_components,
        output_dir=OUTPUT_DIR,
    )
    pipeline = result["pipeline"]
    labels   = result["labels"]
    metrics  = result["metrics"]

    # ── 5. Cluster Profiles ───────────────────────────────
    logger.info("━"*55)
    logger.info("STEP 5 │ Cluster Profiling")
    logger.info("━"*55)
    numeric_cols = [c for c in NUMERIC_FEATURES if c in raw_df.columns]
    prof_df      = cluster_profile(raw_df, labels, numeric_cols)
    prof_dict    = prof_df.to_dict(orient="index")
    prof_dict    = {str(k): v for k, v in prof_dict.items()}

    # ── 6. Save ───────────────────────────────────────────
    logger.info("━"*55)
    logger.info("STEP 6 │ Saving Artefacts")
    logger.info("━"*55)

    # Attach X_transformed for inference k-NN lookup
    pipeline._X_t_train = result["X_transformed"]

    save_pipeline(pipeline, MODEL_PATH)

    # Metrics JSON – serialise numpy types
    clean_metrics = {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in metrics.items()
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(clean_metrics, f, indent=2)
    logger.info(f"Metrics → {METRICS_PATH}")

    with open(PROFILES_PATH, "w") as f:
        json.dump(prof_dict, f, indent=2, default=str)
    logger.info(f"Profiles → {PROFILES_PATH}")

    # Labelled CSV
    raw_df["cluster"]      = labels
    raw_df["segment_name"] = [SEGMENT_MAP.get(l, (f"Cluster {l}", ""))[0] for l in labels]
    raw_df.to_csv("outputs/customers_clustered.csv", index=False)
    logger.info("Labelled data → outputs/customers_clustered.csv")

    # ── Summary ───────────────────────────────────────────
    logger.info("━"*55)
    logger.info("TRAINING COMPLETE")
    logger.info("━"*55)
    for k, v in clean_metrics.items():
        logger.info(f"  {k:<32}: {v}")

    return result


if __name__ == "__main__":
    main(run_tuning=True)
