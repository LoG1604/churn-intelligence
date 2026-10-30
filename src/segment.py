from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DB = Path("data/processed/retail.duckdb")
OUT = Path("data/processed/segments.parquet")
PROFILE = Path("reports/segment_profile.csv")


def main():
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute("SELECT * FROM features").df()
    con.close()

    X = pd.DataFrame(
        {
            "recency": df["recency_days"],
            "frequency": np.log1p(df["frequency"]),
            "monetary": np.log1p(df["monetary"]),
        }
    )
    Xs = StandardScaler().fit_transform(X)

    # Try k = 3..6 and keep the best silhouette score
    best_k, best_score = None, -1
    for k in range(3, 7):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xs)
        score = silhouette_score(Xs, labels)
        print(f"k={k}  silhouette={score:.3f}")
        if score > best_score:
            best_k, best_score = k, score
    print(f"Using k={best_k}\n")

    df["segment"] = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(Xs)

    profile = (
        df.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_orders=("frequency", "mean"),
            avg_spend=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
            churn_rate=("churned", "mean"),
        )
        .round(2)
    )
    profile["revenue_share_pct"] = (100 * profile["total_revenue"] / profile["total_revenue"].sum()).round(1)
    profile["customer_share_pct"] = (100 * profile["customers"] / profile["customers"].sum()).round(1)
    profile = profile.sort_values("total_revenue", ascending=False)

    PROFILE.parent.mkdir(exist_ok=True)
    profile.to_csv(PROFILE)
    df[["customer_id", "segment"]].to_parquet(OUT, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(profile)


if __name__ == "__main__":
    main()