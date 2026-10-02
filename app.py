import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import load_and_clean_data
from association_rules import mine_rules
from clustering import build_customer_features, cluster_customers

st.set_page_config(page_title="SMart Data Mining", page_icon="🛒", layout="wide")

st.title("🛒 SMart — Data Mining System")
st.caption("Customer purchase pattern mining using Apriori and K-Means")

@st.cache_data
def get_data():
    return load_and_clean_data("data/transactions.csv")

df = get_data()

st.sidebar.header("Controls")
min_support = st.sidebar.slider("Minimum Support", 0.01, 0.15, 0.03, 0.01)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.10, 0.95, 0.25, 0.05)
n_clusters = st.sidebar.slider("Customer Clusters", 2, 6, 4)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🔗 Association Rules", "👥 Customer Segmentation", "💡 Insights"
])

with tab1:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", df["TransactionID"].nunique())
    c2.metric("Customers", df["CustomerID"].nunique())
    c3.metric("Products", df["Product"].nunique())
    c4.metric("Rows", len(df))

    st.markdown("### Top Products")
    top = df.groupby("Product")["Quantity"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top)

    st.markdown("### Sample Cleaned Data")
    st.dataframe(df.head(20), use_container_width=True)

with tab2:
    st.subheader("Market Basket Analysis — Apriori")
    basket, frequent, rules = mine_rules(df, min_support, min_confidence)

    c1, c2 = st.columns(2)
    c1.metric("Frequent Itemsets", len(frequent))
    c2.metric("Association Rules", len(rules))

    if rules.empty:
        st.warning("No rules found. Lower minimum support or confidence.")
    else:
        display = rules[
            ["antecedents", "consequents", "support", "confidence", "lift"]
        ].copy()
        display.columns = [
            "If Customer Buys", "Then Often Buys",
            "Support", "Confidence", "Lift"
        ]
        st.dataframe(
            display.head(20).style.format({
                "Support": "{:.3f}",
                "Confidence": "{:.3f}",
                "Lift": "{:.2f}"
            }),
            use_container_width=True
        )

        st.info(
            "Support = how frequently the combination appears. "
            "Confidence = how often the consequent appears when the antecedent appears. "
            "Lift > 1 generally indicates a positive association."
        )

with tab3:
    st.subheader("Customer Segmentation — K-Means")
    customer = build_customer_features(df)
    clustered, model, scaler = cluster_customers(customer, n_clusters)

    st.dataframe(clustered.head(20), use_container_width=True)

    summary = (
        clustered.groupby("Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            Transactions=("transactions", "mean"),
            Total_Items=("total_items", "mean"),
            Unique_Products=("unique_products", "mean")
        )
        .round(2)
        .reset_index()
    )
    st.markdown("### Cluster Summary")
    st.dataframe(summary, use_container_width=True)

    fig, ax = plt.subplots()
    ax.scatter(
        clustered["transactions"],
        clustered["total_items"],
        c=clustered["Cluster"],
        cmap="viridis",
        alpha=0.7
    )
    ax.set_xlabel("Number of Transactions")
    ax.set_ylabel("Total Items Purchased")
    ax.set_title("Customer Clusters")
    st.pyplot(fig)

with tab4:
    st.subheader("Business Insights")
    top_product = (
        df.groupby("Product")["Quantity"].sum().sort_values(ascending=False).index[0]
    )
    basket, frequent, rules = mine_rules(df, min_support, min_confidence)
    customer = build_customer_features(df)
    clustered, _, _ = cluster_customers(customer, n_clusters)

    st.markdown(f"- **Top product by quantity:** {top_product}")

    if not rules.empty:
        r = rules.iloc[0]
        st.markdown(
            f"- **Strongest discovered rule:** "
            f"**{r['antecedents']} → {r['consequents']}** "
            f"(confidence {r['confidence']:.1%}, lift {r['lift']:.2f})"
        )

    largest_cluster = clustered["Cluster"].value_counts().idxmax()
    st.markdown(
        f"- **Largest customer segment:** Cluster {largest_cluster} "
        f"with {int((clustered['Cluster'] == largest_cluster).sum())} customers."
    )

    st.success(
        "These insights can support product bundling, cross-selling, "
        "customer segmentation and targeted marketing decisions."
    )

st.divider()
st.caption("Academic project • Data Mining & Data Warehousing • SMart")
