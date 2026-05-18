"""
streamlit_app/app.py
====================
Streamlit dashboard for Customer Segmentation & Retention.

Tabs:
  1. 🏠 Overview         – project summary & pipeline diagram
  2. 📊 Data Explorer    – raw/processed data stats & distributions
  3. ⚙️  Train & Tune    – run hyperparameter tuning + training
  4. 🔬 Cluster Analysis – cluster plots, profiles, segment details
  5. 🔮 Predict          – single-customer prediction form
  6. 📈 Retention Intel  – actionable recommendations per segment
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import streamlit as st
import joblib

# ── path setup ───────────────────────────────────────────
ROOT = os.path.abspath(os.path.dirname(__file__))  # root of this project
sys.path.insert(0, ROOT)

from utils.preprocessing import load_and_preprocess, NUMERIC_FEATURES
from models.pipeline import (
    tune_hyperparameters,
    fit_and_evaluate,
    cluster_profile,
    save_pipeline,
    estimate_epsilon,
    SEGMENT_MAP,
)

try:
    from data.generate_data import generate_customers
except (ModuleNotFoundError, ImportError):
    from generate_data import generate_customers

# ── constants ────────────────────────────────────────────
DATA_PATH     = os.path.join(ROOT, "data/customers_raw.csv")
MODEL_PATH    = os.path.join(ROOT, "models_saved/pipeline.pkl")
METRICS_PATH  = os.path.join(ROOT, "outputs/metrics.json")
TUNING_PATH   = os.path.join(ROOT, "outputs/tuning_results.csv")
CLUSTER_CSV   = os.path.join(ROOT, "outputs/customers_clustered.csv")
OUTPUT_DIR    = os.path.join(ROOT, "outputs")

SEGMENT_COLORS = {
    -1: "#6c757d", 0: "#4CC9F0", 1: "#F72585", 2: "#FF6B6B",
     3: "#06D6A0", 4: "#FFD166", 5: "#A78BFA", 6: "#FB8500", 7: "#3A86FF",
}

RETENTION_ACTIONS = {
    "Champions":            ["Exclusive VIP rewards & early-access launches",
                             "Referral ambassador programme",
                             "Premium loyalty tier upgrade"],
    "Loyal High-Value":     ["Personalised upsell recommendations",
                             "Dedicated account manager outreach",
                             "Annual loyalty celebration offers"],
    "At-Risk Churners":     ["Automated win-back email sequence",
                             "Targeted discount (15–20 %)",
                             "Satisfaction survey + free gift"],
    "Budget Shoppers":      ["Bundle deals & combo offers",
                             "Loyalty point double-up events",
                             "Free-shipping threshold incentives"],
    "New Prospects":        ["Onboarding drip campaign",
                             "First-purchase welcome discount",
                             "Product discovery quizzes"],
    "Occasional Buyers":    ["Seasonal re-engagement campaigns",
                             "Limited-time flash-sale alerts",
                             "Category-based personalised nudges"],
    "Premium Inactives":    ["Reactivation offer (high-value bespoke)",
                             "Personalised catalogue based on past AOV",
                             "Exclusive member preview events"],
    "Deal Hunters":         ["Proactive promo notifications",
                             "Price-drop alerts on wishlist items",
                             "Gamified challenge with discount reward"],
    "Noise / Outliers":     ["Manual review for fraud / data anomalies",
                             "Data enrichment to reclassify",
                             "Exclude from automated campaigns"],
}

# ─────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Customer Segmentation & Retention",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

  /* Global Styles & Premium Dark Mode */
  html, body, [class*="css"] { 
    font-family: 'Outfit', sans-serif; 
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    color: #f8fafc; 
  }
  
  /* Hide Streamlit elements */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}

  h1, h2, h3, h4 { 
    font-family: 'Space Grotesk', sans-serif; 
    letter-spacing: -0.02em; 
    background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  /* Glassmorphism Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px; 
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    border-radius: 16px; 
    padding: 8px;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }
  .stTabs [data-baseweb="tab"] {
    color: #94a3b8; 
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px; 
    font-weight: 600; 
    border-radius: 12px; 
    padding: 10px 20px;
    transition: all 0.3s ease;
  }
  .stTabs [data-baseweb="tab"]:hover {
    color: #f8fafc;
    background: rgba(255, 255, 255, 0.05);
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important; 
    color: white !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
  }

  /* Metric Cards */
  .metric-card {
    background: rgba(30, 41, 59, 0.7); 
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px; 
    padding: 24px;
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
  }
  .metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    border-color: rgba(99, 102, 241, 0.3);
  }
  .metric-card .val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.5rem; 
    font-weight: 700; 
    color: #38bdf8;
    text-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
  }
  .metric-card .lbl {
    font-size: 13px; 
    color: #94a3b8; 
    text-transform: uppercase;
    letter-spacing: 0.1em; 
    margin-top: 8px;
    font-weight: 500;
  }

  /* Segment Badges */
  .seg-badge {
    display: inline-block; 
    padding: 6px 16px;
    border-radius: 24px; 
    font-family: 'Space Grotesk', sans-serif;
    font-size: 14px; 
    font-weight: 600; 
    letter-spacing: 0.02em;
    margin: 4px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    transition: transform 0.2s ease;
  }
  .seg-badge:hover {
    transform: scale(1.05);
  }

  /* Pipeline Steps */
  .pipeline-step {
    background: rgba(30, 41, 59, 0.5); 
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-left: 4px solid #8b5cf6;
    border-radius: 12px; 
    padding: 16px 20px;
    margin: 10px 0;
    transition: background 0.3s ease;
  }
  .pipeline-step:hover {
    background: rgba(30, 41, 59, 0.8);
    border-left-color: #38bdf8;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%); 
    color: white;
    border: none; 
    border-radius: 12px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600; 
    font-size: 16px;
    letter-spacing: 0.02em; 
    padding: 12px 28px;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    transition: all 0.3s ease;
  }
  .stButton > button:hover { 
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
    color: white;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
  }
  .sidebar-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 24px; 
    font-weight: 700;
    background: linear-gradient(to right, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center; 
    padding: 16px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08); 
    margin-bottom: 20px;
  }

  /* Action Items */
  .action-item {
    background: rgba(30, 41, 59, 0.6); 
    border-left: 3px solid #10b981;
    padding: 12px 18px; 
    border-radius: 0 10px 10px 0;
    margin: 6px 0; 
    font-size: 15px;
    transition: transform 0.2s ease;
  }
  .action-item:hover {
    transform: translateX(4px);
    background: rgba(30, 41, 59, 0.9);
  }

  code, .mono { font-family: 'Space Grotesk', monospace !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────
def _load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def _load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


def _load_tuning():
    if os.path.exists(TUNING_PATH):
        return pd.read_csv(TUNING_PATH)
    return None


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">📊 Customer SegML</div>', unsafe_allow_html=True)

    st.markdown("**Data**")
    if st.button("🔄 Generate Fresh Dataset"):
        generate_customers(save_path=DATA_PATH)
        st.success("Dataset generated!")

    st.markdown("---")
    st.markdown("**Model Status**")
    model_exists   = os.path.exists(MODEL_PATH)
    metrics_exists = os.path.exists(METRICS_PATH)

    st.markdown(f"{'🟢' if model_exists else '🔴'} Pipeline: {'Loaded' if model_exists else 'Not trained'}")
    st.markdown(f"{'🟢' if metrics_exists else '🔴'} Metrics: {'Available' if metrics_exists else 'None'}")
    if metrics_exists:
        m = _load_metrics()
        st.markdown(f"Clusters: **{m.get('n_clusters', '–')}**")
        st.markdown(f"Silhouette: **{m.get('silhouette_score', '–')}**")

    st.markdown("---")
    st.caption("Customer Segmentation & Retention\nCERN Internship Portfolio Project")


# ─────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Overview",
    "📊 Data Explorer",
    "⚙️ Train & Tune",
    "🔬 Cluster Analysis",
    "🔮 Predict",
    "📈 Retention Intel",
])


# ══════════════════════════════════════════════════════════
# TAB 1 – Overview
# ══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## ⚛️  Customer Segmentation & Retention Analysis")
    st.markdown(
        "DBSCAN-based unsupervised segmentation pipeline identifying distinct "
        "customer behavioural profiles for targeted retention strategies."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### ML Pipeline")
        steps = [
            ("1", "Data Generation", "3 000 synthetic customers · RFM + demographic features · injected NaN & outliers"),
            ("2", "Preprocessing",   "Categorical encoding · Median imputation · IQR outlier clipping · RFM feature engineering"),
            ("3", "StandardScaler",  "Zero-mean, unit-variance normalisation – essential for distance-based clustering"),
            ("4", "PCA",             "Dimensionality reduction to 8 components – mitigates curse of dimensionality"),
            ("5", "Hyperparameter Tuning", "Grid search over (ε, min_samples) scored by Silhouette Index"),
            ("6", "DBSCAN",          "Density-Based Spatial Clustering – no predefined k · handles noise · arbitrary shapes"),
            ("7", "Streamlit App",   "Interactive dashboard for training, analysis, and single-customer prediction"),
            ("8", "Docker",          "Containerised deployment – reproducible environment, one-command start"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
              <span style="color:#1f6feb;font-family:'Share Tech Mono',monospace;font-weight:700">[{num}]</span>
              <span style="font-family:'Rajdhani',sans-serif;font-size:16px;font-weight:700;margin:0 8px">{title}</span>
              <span style="color:#8b949e;font-size:13px">{desc}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("### Segment Catalogue")
        for cid, (name, color) in SEGMENT_MAP.items():
            if cid != -1:
                st.markdown(
                    f'<span class="seg-badge" style="background:{color}20;color:{color};border:1px solid {color}40">{name}</span>',
                    unsafe_allow_html=True,
                )

        st.markdown("### Tech Stack")
        for tech in ["Python 3.11", "scikit-learn", "pandas / numpy",
                     "Streamlit", "Docker", "matplotlib", "joblib"]:
            st.markdown(f"• `{tech}`")


# ══════════════════════════════════════════════════════════
# TAB 2 – Data Explorer
# ══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 📊 Data Explorer")

    if not os.path.exists(DATA_PATH):
        st.warning("No dataset found. Generate one from the sidebar first.")
    else:
        df = pd.read_csv(DATA_PATH)

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val in [
            (c1, "Customers",   f"{len(df):,}"),
            (c2, "Features",    str(df.shape[1])),
            (c3, "Missing",     str(df.isnull().sum().sum())),
            (c4, "Numeric cols",str(df.select_dtypes(include=np.number).shape[1])),
        ]:
            col.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{label}</div></div>',
                         unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Raw Data Sample")
        st.dataframe(df.head(50), use_container_width=True)

        st.markdown("### Descriptive Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)

        st.markdown("### Missing Value Map")
        miss = df.isnull().sum().reset_index()
        miss.columns = ["Feature", "Missing Count"]
        miss["Missing %"] = (miss["Missing Count"] / len(df) * 100).round(2)
        miss = miss[miss["Missing Count"] > 0]
        if len(miss):
            st.dataframe(miss, use_container_width=True)
        else:
            st.success("No missing values detected.")

        st.markdown("### Distribution Plots")
        num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        cols     = st.multiselect("Select features to plot", num_cols, default=num_cols[:4])

        if cols:
            fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
            fig.patch.set_facecolor("#0d1117")
            if len(cols) == 1: axes = [axes]
            for ax, col_name in zip(axes, cols):
                ax.set_facecolor("#161b22")
                ax.hist(df[col_name].dropna(), bins=40, color="#4CC9F0", alpha=0.85, edgecolor="none")
                ax.set_title(col_name, color="white", fontsize=10)
                ax.tick_params(colors="#555")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════════════════
# TAB 3 – Train & Tune
# ══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## ⚙️  Train & Hyperparameter Tune")

    with st.expander("Pipeline Configuration", expanded=True):
        cc1, cc2, cc3 = st.columns(3)
        use_pca      = cc1.toggle("Use PCA", value=True)
        n_components = cc2.slider("PCA Components", 2, 15, 8)
        min_samples  = cc3.slider("DBSCAN min_samples", 3, 30, 10)

        run_tuning = st.toggle("Run Hyperparameter Tuning (grid search)", value=True)

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("🚀 Train Model"):
            if not os.path.exists(DATA_PATH):
                st.error("Generate a dataset first.")
            else:
                with st.spinner("Preprocessing …"):
                    raw_df, X, features = load_and_preprocess(DATA_PATH)

                best_eps   = None
                best_ms    = min_samples

                if run_tuning:
                    with st.spinner("Hyperparameter tuning …"):
                        tuning_df = tune_hyperparameters(
                            X, use_pca=use_pca, n_components=n_components
                        )
                        tuning_df.to_csv(TUNING_PATH, index=False)
                        valid = tuning_df.dropna(subset=["silhouette_score"]).sort_values(
                            "silhouette_score", ascending=False
                        )
                        if len(valid):
                            best_eps = float(valid.iloc[0]["eps"])
                            best_ms  = int(valid.iloc[0]["min_samples"])

                with st.spinner("Fitting pipeline …"):
                    result = fit_and_evaluate(
                        X, feature_names=features,
                        eps=best_eps, min_samples=best_ms,
                        use_pca=use_pca, n_components=n_components,
                        output_dir=OUTPUT_DIR,
                    )
                    pipeline = result["pipeline"]
                    pipeline._X_t_train = result["X_transformed"]
                    save_pipeline(pipeline, MODEL_PATH)

                    m = result["metrics"]
                    m_clean = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                               for k, v in m.items()}
                    with open(METRICS_PATH, "w") as f:
                        json.dump(m_clean, f, indent=2)

                    # cluster profiles
                    num_cols = [c for c in NUMERIC_FEATURES if c in raw_df.columns]
                    prof_df  = cluster_profile(raw_df, result["labels"], num_cols)

                st.success(f"✅ Training complete! {m.get('n_clusters')} clusters found.")
                st.json(m_clean)

    with col_b:
        tuning_df = _load_tuning()
        if tuning_df is not None:
            st.markdown("### Tuning Results")
            valid = tuning_df.dropna(subset=["silhouette_score"]).sort_values(
                "silhouette_score", ascending=False
            ).head(20)
            st.dataframe(valid, use_container_width=True)

            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#161b22")
            sc = ax.scatter(
                tuning_df["eps"].dropna(),
                tuning_df["min_samples"].dropna(),
                c=tuning_df["silhouette_score"].fillna(-1),
                cmap="plasma", s=90, alpha=0.85, edgecolors="none",
            )
            plt.colorbar(sc, ax=ax, label="Silhouette")
            ax.set_xlabel("eps", color="#aaa"); ax.set_ylabel("min_samples", color="#aaa")
            ax.set_title("Hyperparameter Space", color="white", fontweight="bold")
            ax.tick_params(colors="#555")
            plt.tight_layout()
            st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════
# TAB 4 – Cluster Analysis
# ══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 🔬 Cluster Analysis")

    metrics = _load_metrics()
    if not metrics:
        st.info("Train the model first (⚙️ Train & Tune tab).")
    else:
        c1, c2, c3, c4 = st.columns(4)
        for col, lbl, key in [
            (c1, "Clusters",    "n_clusters"),
            (c2, "Silhouette",  "silhouette_score"),
            (c3, "Noise %",     "noise_pct"),
            (c4, "DB Index",    "davies_bouldin_score"),
        ]:
            val = metrics.get(key, "–")
            col.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                         unsafe_allow_html=True)

        st.markdown("---")

        plot_path = os.path.join(OUTPUT_DIR, "cluster_plot.png")
        elbow_path = os.path.join(OUTPUT_DIR, "epsilon_elbow.png")

        col_p, col_e = st.columns(2)
        if os.path.exists(plot_path):
            col_p.image(plot_path, caption="DBSCAN Cluster Plot (PCA 2D projection)", use_container_width=True)
        if os.path.exists(elbow_path):
            col_e.image(elbow_path, caption="k-NN Distance Elbow (ε estimation)", use_container_width=True)

        if os.path.exists(CLUSTER_CSV):
            st.markdown("### Cluster Profiles")
            clustered = pd.read_csv(CLUSTER_CSV)
            num_cols  = [c for c in NUMERIC_FEATURES if c in clustered.columns]
            profile   = clustered[clustered["cluster"] != -1].groupby("segment_name")[num_cols].mean().round(2)
            st.dataframe(profile, use_container_width=True)

            st.markdown("### Segment Size Distribution")
            seg_counts = clustered["segment_name"].value_counts()
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#161b22")
            bars = ax.barh(seg_counts.index, seg_counts.values,
                           color=["#4CC9F0","#F72585","#06D6A0","#FFD166","#A78BFA","#FF6B6B","#FB8500","#3A86FF","#6c757d"])
            ax.set_xlabel("Count", color="#aaa"); ax.tick_params(colors="#aaa")
            ax.set_title("Customers per Segment", color="white", fontweight="bold")
            for bar in bars:
                ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                        str(int(bar.get_width())), va="center", color="#aaa", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════
# TAB 5 – Predict
# ══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 🔮 Single-Customer Prediction")

    pipeline = _load_model()
    if pipeline is None:
        st.warning("No trained model found. Go to ⚙️ Train & Tune first.")
    else:
        st.markdown("Enter customer features to assign them to a segment:")

        with st.form("predict_form"):
            r1c1, r1c2, r1c3 = st.columns(3)
            age               = r1c1.number_input("Age",                      18, 100, 35)
            annual_income     = r1c2.number_input("Annual Income ($)",         0, 500000, 65000, step=1000)
            spending_score    = r1c3.slider("Spending Score",                  1, 100, 60)

            r2c1, r2c2, r2c3 = st.columns(3)
            purchase_freq     = r2c1.number_input("Purchase Frequency",        1, 100, 10)
            avg_order         = r2c2.number_input("Avg Order Value ($)",        1.0, 2000.0, 145.0)
            days_since        = r2c3.number_input("Days Since Last Purchase",   1, 500, 15)

            r3c1, r3c2, r3c3 = st.columns(3)
            total_orders      = r3c1.number_input("Total Orders",              1, 500, 30)
            return_rate       = r3c2.slider("Return Rate",                     0.0, 1.0, 0.05)
            loyalty_years     = r3c3.slider("Loyalty Years",                   0.0, 15.0, 3.0)

            r4c1, r4c2, r4c3 = st.columns(3)
            email_open        = r4c1.slider("Email Open Rate",                 0.0, 1.0, 0.3)
            support_tix       = r4c2.number_input("Support Tickets",           0, 50, 1)
            promo_rate        = r4c3.slider("Promo Usage Rate",                0.0, 1.0, 0.25)

            r5c1, r5c2, _ = st.columns(3)
            channel           = r5c1.selectbox("Preferred Channel",            ["online", "mobile", "store"])
            region            = r5c2.selectbox("Region",                       ["North","South","East","West","Central"])

            submitted = st.form_submit_button("🔮 Predict Segment")

        if submitted:
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from sklearn.neighbors import NearestNeighbors

            chan_enc = LabelEncoder().fit(["mobile","online","store"])
            reg_enc  = LabelEncoder().fit(["Central","East","North","South","West"])

            row = {
                "age": age, "annual_income": annual_income, "spending_score": spending_score,
                "purchase_frequency": purchase_freq, "avg_order_value": avg_order,
                "days_since_last_purchase": days_since, "total_orders": total_orders,
                "return_rate": return_rate, "loyalty_years": loyalty_years,
                "email_open_rate": email_open, "support_tickets": support_tix,
                "promo_usage_rate": promo_rate,
                "preferred_channel": int(chan_enc.transform([channel])[0]),
                "region": int(reg_enc.transform([region])[0]),
            }

            # RFM
            row["recency_score"]    = 1.0 / (row["days_since_last_purchase"] + 1)
            row["monetary_score"]   = row["avg_order_value"] * row["purchase_frequency"]
            row["clv_estimate"]     = row["monetary_score"] * (row["loyalty_years"] + 1e-9)
            row["engagement_ratio"] = row["purchase_frequency"] / (row["days_since_last_purchase"] + 1)
            row["churn_risk_score"] = 0.5  # placeholder

            feat_order = [
                "age","annual_income","spending_score","purchase_frequency","avg_order_value",
                "days_since_last_purchase","total_orders","return_rate","loyalty_years",
                "email_open_rate","support_tickets","promo_usage_rate","preferred_channel","region",
                "recency_score","monetary_score","clv_estimate","engagement_ratio","churn_risk_score",
            ]
            X_in = np.array([[row.get(f, 0) for f in feat_order]], dtype=float)

            try:
                X_t  = pipeline._transform.transform(X_in)
                core = pipeline._X_t_train[pipeline._dbscan.core_sample_indices_]
                core_labels = pipeline._labels[pipeline._dbscan.core_sample_indices_]

                nbrs = NearestNeighbors(n_neighbors=1).fit(core)
                dist, idx = nbrs.kneighbors(X_t)

                dist_val = float(dist[0][0])
                is_noise = dist_val > pipeline._dbscan.eps
                lbl      = int(core_labels[idx[0][0]]) if not is_noise else -1
                seg_name, seg_color = SEGMENT_MAP.get(lbl, (f"Cluster {lbl}", "#888"))
                confidence = max(0.0, 1.0 - dist_val / (pipeline._dbscan.eps + 1e-9))

                st.markdown("---")
                st.markdown(
                    f'<div style="text-align:center;padding:30px;background:#161b22;border-radius:16px;'
                    f'border:2px solid {seg_color}40">'
                    f'<div style="font-family:Rajdhani,sans-serif;font-size:14px;color:#8b949e;letter-spacing:.1em;'
                    f'text-transform:uppercase">Assigned Segment</div>'
                    f'<div style="font-family:Rajdhani,sans-serif;font-size:36px;font-weight:700;color:{seg_color};'
                    f'margin:10px 0">{seg_name}</div>'
                    f'<div style="color:#aaa;font-size:14px">Confidence: {confidence:.1%} &nbsp;|&nbsp; '
                    f'Noise: {"Yes" if is_noise else "No"}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Prediction error: {e}")


# ══════════════════════════════════════════════════════════
# TAB 6 – Retention Intel
# ══════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## 📈 Retention Strategy Intelligence")
    st.markdown("Actionable recommendations for each customer segment:")

    for seg_name, actions in RETENTION_ACTIONS.items():
        # Find color
        color = next((c for _, (n, c) in SEGMENT_MAP.items() if n == seg_name), "#4CC9F0")
        with st.expander(f"**{seg_name}**", expanded=False):
            for act in actions:
                st.markdown(f'<div class="action-item">✦ {act}</div>', unsafe_allow_html=True)

    if os.path.exists(CLUSTER_CSV):
        st.markdown("---")
        st.markdown("### Segment Overview Table")
        df_c = pd.read_csv(CLUSTER_CSV)
        summary = df_c[df_c["cluster"] != -1].groupby("segment_name").agg(
            Count           = ("customer_id", "count"),
            Avg_Income      = ("annual_income", "mean"),
            Avg_Orders      = ("total_orders", "mean"),
            Avg_AOV         = ("avg_order_value", "mean"),
            Avg_Loyalty_Yrs = ("loyalty_years", "mean"),
        ).round(1)
        st.dataframe(summary, use_container_width=True)
