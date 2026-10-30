import os
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors

PROFILE = Path("reports/segment_profile.csv")
SHAP_CSV = Path("reports/shap_importance.csv")
OUT = Path("reports/retention_plan.md")


def name_segments(profile: pd.DataFrame) -> pd.DataFrame:
    """Name segments by behavior, since KMeans ids are arbitrary."""
    profile = profile.copy()
    names = {}
    names[profile["total_revenue"].idxmax()] = "Champions"
    names[profile["churn_rate"].idxmax()] = "Lapsed"
    for seg in profile.index:
        names.setdefault(seg, "Occasional Buyers")
    profile["name"] = profile.index.map(names)
    return profile


def build_prompt(profile: pd.DataFrame, shap_df: pd.DataFrame) -> str:
    cols = [
        "name", "customers", "customer_share_pct", "revenue_share_pct",
        "avg_recency_days", "avg_orders", "avg_spend", "churn_rate",
    ]
    seg_table = profile[cols].to_markdown(index=False)
    top_drivers = ", ".join(shap_df["feature"].head(4))

    return f"""You are a retention analyst for a UK online gift retailer.
Below are customer segments built from purchase history, plus the top drivers
of churn from an XGBoost model. churn_rate is the share of customers with no
purchase in the following 90 days.

SEGMENTS:
{seg_table}

TOP CHURN DRIVERS (most important first): {top_drivers}

For each segment write:
1. One sentence describing who they are, using only the numbers above.
2. Two specific retention actions (email, offer, timing).
3. One metric to track.

Rules: use only the numbers provided, do not invent statistics, keep the
whole answer under 350 words, use plain language a manager can act on."""


def generate_with_retry(client, models: list[str], prompt: str, tries: int = 4) -> str:
    """Try each model in order. Retry on 429/503 with exponential backoff."""
    for model in models:
        for attempt in range(tries):
            try:
                resp = client.models.generate_content(model=model, contents=prompt)
                print(f"Generated with {model}")
                return resp.text
            except errors.APIError as e:
                if e.code not in (429, 500, 503):
                    raise
                wait = 2 ** attempt * 3  # 3s, 6s, 12s, 24s
                print(f"{model} returned {e.code}, retrying in {wait}s ({attempt + 1}/{tries})")
                time.sleep(wait)
        print(f"{model} still busy, switching model")
    raise RuntimeError("All models are busy. Try again in a few minutes.")


def main():
    load_dotenv()
    profile = name_segments(pd.read_csv(PROFILE, index_col="segment"))
    shap_df = pd.read_csv(SHAP_CSV)
    prompt = build_prompt(profile, shap_df)

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("No GEMINI_API_KEY found. Prompt that would be sent:\n")
        print(prompt)
        return

    models = [
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite"),
    ]
    client = genai.Client(api_key=key)
    text = generate_with_retry(client, models, prompt)

    OUT.write_text(text)
    print(text)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()