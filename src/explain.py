from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

MODEL_PATH = Path("models/xgb_churn.joblib")
TEST_PATH = Path("data/processed/test_set.parquet")
REPORTS = Path("reports")


def save(name: str):
    plt.tight_layout()
    plt.savefig(REPORTS / name, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    REPORTS.mkdir(exist_ok=True)

    bundle = joblib.load(MODEL_PATH)
    model, features = bundle["model"], bundle["features"]
    X = pd.read_parquet(TEST_PATH)[features]

    explainer = shap.TreeExplainer(model)
    sv = explainer(X)

    # 1. Global importance
    shap.plots.bar(sv, show=False)
    save("shap_importance.png")

    # 2. Direction of effect (each dot is a customer)
    shap.plots.beeswarm(sv, show=False)
    save("shap_beeswarm.png")

    # 3. Overdue ratio effect
    shap.plots.scatter(sv[:, "overdue_ratio"], show=False)
    save("shap_overdue_ratio.png")

    # 4. Table version for the dashboard and LLM layer later
    importance = (
        pd.DataFrame({"feature": features, "mean_abs_shap": abs(sv.values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(REPORTS / "shap_importance.csv", index=False)
    print(importance)
    print("Saved plots to reports/")


if __name__ == "__main__":
    main()