from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA = Path(__file__).parent / "data"
ORDER = ["Champions", "Occasional Buyers", "Lapsed"]

# Consistent colors everywhere: Champions = safe, Lapsed = risk
COLORS = {
    "Champions": "#2E8B57",
    "Occasional Buyers": "#F4A300",
    "Lapsed": "#C0392B",
}

# Short, fixed actions per segment, matching the retention plan tab
ACTIONS = {
    "Champions": "Loyalty perk, early access",
    "Occasional Buyers": "Nudge toward 3rd order",
    "Lapsed": "Low-cost win-back only",
}

st.set_page_config(page_title="Churn Intelligence", layout="wide")


@st.cache_data
def load():
    customers = pd.read_parquet(DATA / "customers.parquet")
    segments = pd.read_csv(DATA / "segments.csv").set_index("name").loc[ORDER]
    shap_df = pd.read_csv(DATA / "shap_importance.csv")
    plan = (DATA / "retention_plan.md").read_text()
    return customers, segments, shap_df, plan


customers, segments, shap_df, plan = load()

st.title("Customer Churn Intelligence")
st.markdown(
    "**What this does:** segments customers by purchase behavior, predicts who's "
    "likely to churn, explains why with SHAP, and turns that into a retention plan "
    "a manager can act on."
)
st.caption(
    "UK online retailer, Dec 2010 to Dec 2011. Features come from purchases before "
    "2011-09-10. Churned means no purchase in the following 90 days."
)
st.divider()

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(customers):,}")
c2.metric("Past revenue", f"{customers['monetary'].sum() / 1e6:.2f}M")
c3.metric("Churn rate", f"{customers['churned'].mean():.0%}")
top = segments.loc["Champions"]
c4.metric("Champions: share of revenue", f"{top['revenue_share_pct']:.0f}%")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Segment explorer", "Why customers churn", "Retention plan"]
)

with tab1:
    left, right = st.columns(2)

    share = segments.reset_index().melt(
        id_vars="name",
        value_vars=["customer_share_pct", "revenue_share_pct"],
        var_name="metric",
        value_name="pct",
    )
    share["metric"] = share["metric"].map(
        {"customer_share_pct": "% of customers", "revenue_share_pct": "% of past revenue"}
    )
    fig = px.bar(
        share, x="metric", y="pct", color="name", barmode="group",
        category_orders={"name": ORDER}, color_discrete_map=COLORS,
        labels={"metric": "", "pct": "%", "name": "Segment"},
        title="Customers vs revenue by segment",
    )
    left.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        segments.reset_index(), x="name", y="churn_rate", color="name",
        category_orders={"name": ORDER}, color_discrete_map=COLORS,
        labels={"name": "", "churn_rate": "Churn rate"},
        title="Churn rate by segment", text_auto=".0%",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(showlegend=False)
    right.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment profile")
    view = segments[
        ["customers", "customer_share_pct", "revenue_share_pct",
         "avg_recency_days", "avg_orders", "avg_spend", "churn_rate"]
    ]
    st.dataframe(view, use_container_width=True)

with tab2:
    name = st.selectbox("Segment", ORDER)
    seg = segments.loc[name]
    sub = customers[customers["segment_name"] == name]

    a, b, c, d = st.columns(4)
    a.metric("Customers", f"{int(seg['customers']):,}")
    b.metric("Avg orders", f"{seg['avg_orders']:.1f}")
    c.metric("Avg days since last order", f"{seg['avg_recency_days']:.0f}")
    d.metric("Churn rate", f"{seg['churn_rate']:.0%}")

    st.info(f"**Suggested action for {name}:** {ACTIONS[name]}")

    st.subheader("Highest churn risk in this segment")
    st.caption(
        "churn_risk is the model's predicted probability. Customers were part of "
        "training, so treat it as a ranking, not a forecast."
    )
    risky = (
        sub.sort_values("churn_risk", ascending=False)
        .head(20)[["customer_id", "recency_days", "frequency", "monetary", "churn_risk"]]
        .copy()
    )
    risky["suggested_action"] = ACTIONS[name]
    st.dataframe(risky, use_container_width=True, hide_index=True)

    fig = px.histogram(
        sub, x="churn_risk", nbins=30, title="Churn risk distribution",
        labels={"churn_risk": "Predicted churn risk"},
        color_discrete_sequence=[COLORS[name]],
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.write(
        "Top drivers from the XGBoost model, explained with SHAP. "
        "Overdue ratio is recency divided by the customer's usual gap between orders."
    )
    left, right = st.columns(2)
    left.image(str(DATA / "shap_importance.png"), caption="Average impact on churn prediction")
    right.image(str(DATA / "shap_beeswarm.png"), caption="Direction of impact (red = high value)")
    st.image(str(DATA / "shap_overdue_ratio.png"), caption="Risk rises fast until about 8x overdue, then flattens", width=500)

with tab4:
    st.markdown(plan)
    st.caption("Generated by an LLM from the segment table and top churn drivers. Reviewed manually.")