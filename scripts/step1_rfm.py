import pandas as pd

# Load the dataset
df = pd.read_csv("online_retail_II.csv")
print("Original rows:", len(df))

# Handle column name variations (Customer ID vs CustomerID)
if "Customer ID" in df.columns:
    df = df.rename(columns={"Customer ID": "CustomerID"})

# Cleaning: remove missing IDs, keep only valid positive transactions
df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["CustomerID"] = df["CustomerID"].astype(str)
df["Revenue"] = df["Quantity"] * df["Price"]
print("Cleaned rows:", len(df))

# Snapshot date = day after the last transaction (standard RFM practice)
snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# RECENCY: days since each customer's last purchase
last_purchase = df.groupby("CustomerID")["InvoiceDate"].max().reset_index()
last_purchase.columns = ["CustomerID", "LastDate"]
last_purchase["Recency"] = (snapshot - last_purchase["LastDate"]).dt.days

# FREQUENCY: number of unique invoices per customer
freq = df.groupby("CustomerID")["Invoice"].nunique().reset_index()
freq.columns = ["CustomerID", "Frequency"]

# MONETARY: total revenue per customer
mon = df.groupby("CustomerID")["Revenue"].sum().reset_index()
mon.columns = ["CustomerID", "Monetary"]

# Country (for later geographic insights)
country = df.drop_duplicates("CustomerID")[["CustomerID", "Country"]]

# Merge into the final RFM table
rfm = last_purchase.merge(freq, on="CustomerID").merge(mon, on="CustomerID").merge(country, on="CustomerID")
rfm = rfm[["CustomerID", "Country", "Recency", "Frequency", "Monetary"]]

print("Unique customers:", len(rfm))
print(rfm[["Recency", "Frequency", "Monetary"]].describe())
print("Top 5 customers by Monetary:")
print(rfm.sort_values("Monetary", ascending=False).head())

rfm.to_csv("rfm.csv", index=False)
print("SUCCESS: rfm.csv saved!")