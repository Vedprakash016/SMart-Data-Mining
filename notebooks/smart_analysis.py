from pathlib import Path
# SmartMart Data Mining Analysis
# Run this file cell-by-cell in Jupyter/VS Code if desired.

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(PROJECT_ROOT / "data" / "transactions.csv")
print(df.head())
print(df.info())

# 1. Cleaning
df = df.drop_duplicates()
df = df.dropna(subset=["TransactionID", "CustomerID", "Product"])
df["Product"] = df["Product"].str.strip()

# 2. Product frequency
product_frequency = df.groupby("Product")["Quantity"].sum().sort_values(ascending=False)
print(product_frequency)

# 3. Apriori
basket = (
    df.assign(value=1)
      .pivot_table(index="TransactionID", columns="Product",
                  values="value", aggfunc="max", fill_value=0)
      .astype(bool)
)
frequent = apriori(basket, min_support=0.03, use_colnames=True)
rules = association_rules(frequent, metric="confidence", min_threshold=0.25)
rules = rules.sort_values(["lift", "confidence"], ascending=False)
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(15))

# 4. K-Means customer segmentation
customer = (
    df.groupby("CustomerID")
      .agg(transactions=("TransactionID", "nunique"),
           total_items=("Quantity", "sum"),
           unique_products=("Product", "nunique"))
      .reset_index()
)
customer["avg_items_per_transaction"] = (
    customer["total_items"] / customer["transactions"]
)

features = ["transactions", "total_items", "unique_products", "avg_items_per_transaction"]
X = StandardScaler().fit_transform(customer[features])
model = KMeans(n_clusters=4, random_state=42, n_init=10)
customer["Cluster"] = model.fit_predict(X)
print(customer.groupby("Cluster")[features].mean().round(2))
