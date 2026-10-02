import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def build_customer_features(df):
    customer = (
        df.groupby("CustomerID")
          .agg(
              transactions=("TransactionID", "nunique"),
              total_items=("Quantity", "sum"),
              unique_products=("Product", "nunique")
          )
          .reset_index()
    )

    customer["avg_items_per_transaction"] = (
        customer["total_items"] / customer["transactions"]
    )
    return customer

def cluster_customers(customer, n_clusters=4):
    features = [
        "transactions",
        "total_items",
        "unique_products",
        "avg_items_per_transaction"
    ]

    scaler = StandardScaler()
    X = scaler.fit_transform(customer[features])

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    customer = customer.copy()
    customer["Cluster"] = model.fit_predict(X)

    return customer, model, scaler
