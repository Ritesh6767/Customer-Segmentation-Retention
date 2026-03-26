"""
pipeline.py
===========
Constructs the full scikit-learn Pipeline:
    StandardScaler → PCA → DBSCAN

Includes:
  - Automatic ε estimation via k-NN elbow heuristic
  - Grid-search-style hyperparameter tuning across (eps, min_samples) combos
  - Silhouette, Davies-Bouldin, Calinski-Harabasz evaluation
  - Cluster profiling & segment naming
  - save / load helpers
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import logging
import itertools
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SEGMENT_MAP = {
    -1: ("Noise / Outliers",      "#6c757d"),
    0:  ("Budget Shoppers",       "#4CC9F0"),
    1:  ("Loyal High-Value",      "#F72585"),
    2:  ("At-Risk Churners",      "#FF6B6B"),
    3:  ("Champions",             "#06D6A0"),
    4:  ("New Prospects",         "#FFD166"),
    5:  ("Occasional Buyers",     "#A78BFA"),
    6:  ("Premium Inactives",     "#FB8500"),
    7:  ("Deal Hunters",          "#3A86FF"),
}


# ─────────────────────────────────────────────────────────
# ε Estimation
# ─────────────────────────────────────────────────────────

def estimate_epsilon(X_scaled: np.ndarray, k: int = 5, plot_path: str = None) -> float:
    """k-NN distance elbow method to estimate DBSCAN ε."""
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_dist = np.sort(distances[:, k - 1])[::-1]

    # Second derivative elbow
    d2      = np.diff(np.diff(k_dist))
    elbow   = int(np.argmax(d2)) + 1
    eps_val = float(k_dist[elbow])

    if plot_path:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(k_dist, color="#4CC9F0", lw=2)
        ax.axvline(elbow, color="#F72585", ls="--", label=f"Elbow  ε ≈ {eps_val:.3f}")
        ax.set_title(f"k-NN Distance Plot  (k={k})", fontweight="bold")
        ax.set_xlabel("Points sorted by distance")
        ax.set_ylabel(f"{k}-NN distance")
        ax.legend(); ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"[Epsilon] Plot saved → {plot_path}")

    logger.info(f"[Epsilon] Estimated ε = {eps_val:.4f}")
    return eps_val


# ─────────────────────────────────────────────────────────
# Pipeline Builder
# ─────────────────────────────────────────────────────────

def build_pipeline(
    eps:          float = 0.5,
    min_samples:  int   = 5,
    use_pca:      bool  = True,
    n_components: int   = 8,
) -> Pipeline:
    """
    Builds sklearn Pipeline:
        StandardScaler → [PCA] → DBSCAN
    """
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_components, random_state=42)))
    steps.append(("dbscan", DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)))

    pipe = Pipeline(steps)
    logger.info(f"[Pipeline] Steps: {[s[0] for s in steps]}")
    return pipe


# ─────────────────────────────────────────────────────────
# Hyperparameter Tuning
# ─────────────────────────────────────────────────────────

def tune_hyperparameters(
    X:            np.ndarray,
    eps_values:   list = None,
    min_samples_values: list = None,
    use_pca:      bool = True,
    n_components: int  = 8,
    top_n:        int  = 5,
) -> pd.DataFrame:
    """
    Grid search over (eps, min_samples) combinations.
    Scores each by Silhouette Score (higher = better).

    Returns a DataFrame of all results sorted by silhouette score.
    """
    # Pre-transform once (scaler + PCA) for speed
    transform_pipe = Pipeline([("scaler", StandardScaler())])
    if use_pca:
        transform_pipe.steps.append(("pca", PCA(n_components=n_components, random_state=42)))
    X_t = transform_pipe.fit_transform(X)

    # Auto-generate eps range if not provided
    if eps_values is None:
        base_eps = estimate_epsilon(X_t, k=5)
        eps_values = [round(base_eps * m, 3) for m in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]]

    if min_samples_values is None:
        min_samples_values = [3, 5, 8, 10, 15, 20]

    combos  = list(itertools.product(eps_values, min_samples_values))
    results = []

    logger.info(f"[Tuning] Testing {len(combos)} (eps, min_samples) combinations …")

    for eps, ms in combos:
        db     = DBSCAN(eps=eps, min_samples=ms, metric="euclidean", n_jobs=-1)
        labels = db.fit_predict(X_t)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = int(np.sum(labels == -1))
        noise_pct  = round(100 * n_noise / len(labels), 2)

        sil, db_score, ch_score = None, None, None
        if n_clusters >= 2:
            mask = labels != -1
            if mask.sum() > n_clusters:
                try:
                    sil      = round(silhouette_score(X_t[mask], labels[mask]), 4)
                    db_score = round(davies_bouldin_score(X_t[mask], labels[mask]), 4)
                    ch_score = round(calinski_harabasz_score(X_t[mask], labels[mask]), 2)
                except Exception:
                    pass

        results.append({
            "eps":                    eps,
            "min_samples":            ms,
            "n_clusters":             n_clusters,
            "n_noise":                n_noise,
            "noise_pct":              noise_pct,
            "silhouette_score":       sil,
            "davies_bouldin_score":   db_score,
            "calinski_harabasz_score": ch_score,
        })

    df_results = pd.DataFrame(results)
    df_valid   = df_results.dropna(subset=["silhouette_score"]).sort_values(
        "silhouette_score", ascending=False
    )

    logger.info(f"[Tuning] {len(df_valid)} valid configurations found.")
    if len(df_valid):
        best = df_valid.iloc[0]
        logger.info(
            f"[Tuning] Best → eps={best.eps}, min_samples={best.min_samples}, "
            f"clusters={best.n_clusters}, silhouette={best.silhouette_score}"
        )

    return df_results


# ─────────────────────────────────────────────────────────
# Fit & Evaluate
# ─────────────────────────────────────────────────────────

def fit_and_evaluate(
    X:            np.ndarray,
    feature_names: list,
    eps:          float = None,
    min_samples:  int   = 10,
    use_pca:      bool  = True,
    n_components: int   = 8,
    output_dir:   str   = "outputs",
) -> dict:
    """
    Fits the pipeline and evaluates cluster quality.
    Returns dict with pipeline, labels, metrics, plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Transform first to estimate epsilon
    tmp = Pipeline([("scaler", StandardScaler())])
    if use_pca:
        tmp.steps.append(("pca", PCA(n_components=n_components, random_state=42)))
    X_t = tmp.fit_transform(X)

    if eps is None:
        eps = estimate_epsilon(X_t, k=min_samples,
                               plot_path=f"{output_dir}/epsilon_elbow.png")

    # Build & apply full pipeline
    pipeline   = build_pipeline(eps=eps, min_samples=min_samples,
                                use_pca=use_pca, n_components=n_components)
    transform  = Pipeline(pipeline.steps[:-1])
    X_t        = transform.fit_transform(X)
    dbscan     = pipeline.named_steps["dbscan"]
    labels     = dbscan.fit_predict(X_t)

    # Store for inference
    pipeline._transform  = transform
    pipeline._dbscan     = dbscan
    pipeline._X_t        = X_t
    pipeline._labels     = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int(np.sum(labels == -1))

    metrics = {
        "eps": round(eps, 4),
        "min_samples": min_samples,
        "n_clusters":  n_clusters,
        "n_noise":     n_noise,
        "noise_pct":   round(100 * n_noise / len(labels), 2),
        "total_points": len(labels),
    }

    if n_clusters >= 2:
        mask = labels != -1
        if mask.sum() > n_clusters:
            metrics["silhouette_score"]        = round(silhouette_score(X_t[mask], labels[mask]), 4)
            metrics["davies_bouldin_score"]    = round(davies_bouldin_score(X_t[mask], labels[mask]), 4)
            metrics["calinski_harabasz_score"] = round(calinski_harabasz_score(X_t[mask], labels[mask]), 2)

    logger.info(f"[Eval] Clusters={n_clusters} | Noise={metrics['noise_pct']}% | "
                f"Silhouette={metrics.get('silhouette_score', 'N/A')}")

    _plot_clusters(X_t, labels, n_clusters, output_dir)

    return {
        "pipeline":      pipeline,
        "labels":        labels,
        "metrics":       metrics,
        "X_transformed": X_t,
        "eps":           eps,
    }


def _plot_clusters(X_t, labels, n_clusters, output_dir):
    xy = X_t[:, :2]
    cmap = plt.cm.get_cmap("tab10", max(n_clusters, 1))

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    for lbl in sorted(set(labels)):
        m     = labels == lbl
        color = "#555" if lbl == -1 else cmap(lbl)
        name  = "Noise" if lbl == -1 else SEGMENT_MAP.get(lbl, (f"Cluster {lbl}", ""))[0]
        ax.scatter(xy[m, 0], xy[m, 1], c=[color], s=14, alpha=0.75,
                   label=name, edgecolors="none")

    ax.set_title(f"DBSCAN Clustering  ·  {n_clusters} Segments",
                 color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("PCA 1", color="#aaa"); ax.set_ylabel("PCA 2", color="#aaa")
    ax.tick_params(colors="#555"); ax.legend(fontsize=8, labelcolor="white",
                                              facecolor="#161b22", edgecolor="#30363d")
    ax.grid(alpha=0.08, color="white")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_plot.png", dpi=150)
    plt.close()
    logger.info(f"[Plot] Cluster plot → {output_dir}/cluster_plot.png")


# ─────────────────────────────────────────────────────────
# Cluster Profile
# ─────────────────────────────────────────────────────────

def cluster_profile(raw_df: pd.DataFrame, labels: np.ndarray, numeric_cols: list) -> pd.DataFrame:
    df = raw_df.copy()
    df["cluster"] = labels
    df = df[df["cluster"] != -1]
    profile = df.groupby("cluster")[numeric_cols].mean().round(2)
    profile["segment_name"] = profile.index.map(lambda i: SEGMENT_MAP.get(i, (f"Cluster {i}", ""))[0])
    profile["size"]          = df.groupby("cluster").size()
    return profile


# ─────────────────────────────────────────────────────────
# Persist
# ─────────────────────────────────────────────────────────

def save_pipeline(pipeline, path: str = "models_saved/pipeline.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"[Save] Pipeline → {path}")


def load_pipeline(path: str = "models_saved/pipeline.pkl"):
    p = joblib.load(path)
    return p
    logger.info(f"[Load] Pipeline ← {path}")
    return p
