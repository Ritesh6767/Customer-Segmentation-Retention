"""
generate_data.py
================
Generates a realistic synthetic customer dataset with:
- RFM (Recency, Frequency, Monetary) behaviour signals
- Demographic features
- Controlled missing values & outliers for preprocessing demo
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 3000


def generate_customers(n: int = N, save_path: str = "data/customers_raw.csv") -> pd.DataFrame:
    data = {
        "customer_id": [f"CUST_{i:05d}" for i in range(1, n + 1)],

        # Demographics
        "age":                      np.random.randint(18, 75, n).astype(float),
        "annual_income":            np.random.normal(58000, 22000, n).clip(12000, 250000),
        "region":                   np.random.choice(["North", "South", "East", "West", "Central"], n),
        "preferred_channel":        np.random.choice(["online", "mobile", "store"], n, p=[0.5, 0.3, 0.2]),

        # Behavioural / RFM
        "days_since_last_purchase": np.random.exponential(35, n).clip(1, 400),
        "purchase_frequency":       np.random.poisson(9, n).clip(1, 60).astype(float),
        "avg_order_value":          np.random.exponential(130, n).clip(8, 1200),
        "total_orders":             np.random.poisson(28, n).clip(1, 250).astype(float),
        "return_rate":              np.random.beta(2, 12, n),
        "loyalty_years":            np.random.uniform(0, 12, n),
        "spending_score":           np.random.randint(1, 101, n).astype(float),

        # Engagement
        "email_open_rate":          np.random.beta(3, 7, n),
        "support_tickets":          np.random.poisson(1.5, n).clip(0, 20).astype(float),
        "promo_usage_rate":         np.random.beta(2, 5, n),
    }

    df = pd.DataFrame(data)

    # ── Inject realistic missingness ──
    for col, frac in [("age", 0.04), ("annual_income", 0.03),
                      ("avg_order_value", 0.02), ("email_open_rate", 0.05)]:
        idx = np.random.choice(df.index, int(frac * n), replace=False)
        df.loc[idx, col] = np.nan

    # ── Inject outliers ──
    outlier_idx = np.random.choice(df.index, 30, replace=False)
    df.loc[outlier_idx, "annual_income"] *= 8
    df.loc[outlier_idx[:15], "avg_order_value"] *= 12

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[DataGen] Saved {df.shape[0]} rows × {df.shape[1]} cols → {save_path}")
    print(f"[DataGen] Missing values: {df.isnull().sum().sum()} total")
    return df


if __name__ == "__main__":
    generate_customers()
