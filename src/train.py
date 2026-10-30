from pathlib import Path
import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DB = Path("data/processed/retail.duckdb")
MODEL_PATH = Path("models/xgb_churn.joblib")
TEST_PATH = Path("data/processed/test_set.parquet")

FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "n_products",
    "total_items",
    "tenure_days",
    "orders_last_90d",
    "is_uk",
    "avg_gap_days",
    "overdue_ratio",
]
TARGET = "churned"


def load_data() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute("SELECT * FROM features").df()
    con.close()

    # How overdue is the customer vs their own usual buying rhythm
    gap = df["avg_gap_days"].where(df["avg_gap_days"] > 0)
    df["overdue_ratio"] = df["recency_days"] / gap

    # Log-scale the heavy-tailed columns so the linear baseline is fair
    for col in ["frequency", "monetary", "total_items"]:
        df[col] = np.log1p(df[col])
    return df


def main():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    baseline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5,
        eval_metric="auc",
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_cv = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="roc_auc")
    xgb_cv = cross_val_score(xgb, X_train, y_train, cv=cv, scoring="roc_auc")

    baseline.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    base_auc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])
    proba = xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, proba)
    preds = (proba >= 0.5).astype(int)

    print(f"LogReg  CV AUC: {base_cv.mean():.3f} +/- {base_cv.std():.3f} | test AUC: {base_auc:.3f}")
    print(f"XGBoost CV AUC: {xgb_cv.mean():.3f} +/- {xgb_cv.std():.3f} | test AUC: {xgb_auc:.3f}")
    print()
    print(classification_report(y_test, preds, target_names=["retained", "churned"]))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": xgb, "features": FEATURES}, MODEL_PATH)

    test_out = X_test.copy()
    test_out[TARGET] = y_test.values
    test_out["churn_proba"] = proba
    test_out.to_parquet(TEST_PATH)
    print(f"Saved model to {MODEL_PATH} and test set to {TEST_PATH}")


if __name__ == "__main__":
    main()