import pandas as pd

def load_and_clean_data(path="data/transactions.csv"):
    df = pd.read_csv(path)

    # Basic data cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=["TransactionID", "CustomerID", "Product"])
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(1)
    df["Product"] = df["Product"].astype(str).str.strip()

    return df
