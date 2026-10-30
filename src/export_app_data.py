import shutil
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from llm_insights import name_segments

DB = Path("data/processed/retail.duckdb")
MODEL_PATH = Path("models/xgb_churn.joblib")
SEGMENTS = Path("data/processed/segments.parquet")
REPORTS = Path("reports")
OUT = Path("app/data")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute("SELECT * FROM features").df()
    con.close()

    # Same transforms as train.py so predictions match the trained model
    gap = df["avg_gap_days"].where(df["avg_gap_days"] > 0)
    df["overdue_ratio"] = df["recency_days"] / gap
    model_in = df.copy()
    for col in ["frequency", "monetary", "total_items"]:
        model_in[col] = np.log1p(model_in[col])

    bundle = joblib.load(MODEL_PATH)
    df["churn_risk"] = bundle["model"].predict_proba(model_in[bundle["features"]])[:, 1]

    # Segments, named by behavior
    profile = name_segments(pd.read_csv(REPORTS / "segment_profile.csv", index_col="segment"))
    seg = pd.read_parquet(SEGMENTS)
    df = df.merge(seg, on="customer_id")
    df["segment_name"] = df["segment"].map(profile["name"])

    keep = [
        "customer_id", "segment_name", "recency_days", "frequency", "monetary",
        "avg_order_value", "n_products", "tenure_days", "churned", "churn_risk",
    ]
    df[keep].round(3).to_parquet(OUT / "customers.parquet", index=False)
    profile.reset_index().to_csv(OUT / "segments.csv", index=False)

    shutil.copy(REPORTS / "shap_importance.csv", OUT / "shap_importance.csv")
    shutil.copy(REPORTS / "retention_plan.md", OUT / "retention_plan.md")
    for img in ["shap_importance.png", "shap_beeswarm.png", "shap_overdue_ratio.png"]:
        shutil.copy(REPORTS / img, OUT / img)

    print(f"Exported {len(df)} customers to {OUT}/")


if __name__ == "__main__":
    main()