import pandas as pd
import numpy as np
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# ==========================================
# 1. LOAD RAW TRANSACTION DATA
# ==========================================
df = pd.read_csv("online_retail_II.csv")
if "Customer ID" in df.columns:
    df = df.rename(columns={"Customer ID": "CustomerID"})

# Clean data (Lifetimes requires positive values and no missing IDs)
df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["CustomerID"] = df["CustomerID"].astype(float).astype(int).astype(str)
df["Revenue"] = df["Quantity"] * df["Price"]

# ==========================================
# 2. CREATE SUMMARY DATA FOR LIFETIMES
# ==========================================
summary = summary_data_from_transaction_data(
    df,
    customer_id_col="CustomerID",
    datetime_col="InvoiceDate",
    monetary_value_col="Revenue",
    freq="D"
)

# Gamma-Gamma model requires repeat customers (frequency > 0)
clv_data = summary[summary["frequency"] > 0].copy()
clv_data.index = clv_data.index.astype(str)

# ==========================================
# 3. PREDICT FUTURE TRANSACTIONS (BG/NBD Model)
# ==========================================
print("Fitting BG/NBD model for expected transactions...")
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(clv_data["frequency"], clv_data["recency"], clv_data["T"])

# Predict expected purchases in the next 180 days (6 months)
t_future = 180
clv_data["expected_purchases_6m"] = bgf.conditional_expected_number_of_purchases_up_to_time(
    t_future, 
    clv_data["frequency"], 
    clv_data["recency"], 
    clv_data["T"]
)

# ==========================================
# 4. PREDICT AVERAGE PROFIT (Gamma-Gamma Model)
# ==========================================
print("Fitting Gamma-Gamma model for expected profit...")
ggf = GammaGammaFitter(penalizer_coef=0.0)
ggf.fit(clv_data["frequency"], clv_data["monetary_value"])

clv_data["expected_average_profit"] = ggf.conditional_expected_average_profit(
    clv_data["frequency"], 
    clv_data["monetary_value"]
)

# ==========================================
# 5. CALCULATE 6-MONTH CLV
# ==========================================
# Assume a 20% gross profit margin (standard retail assumption)
gross_profit_margin = 0.20

clv_data["predicted_CLV_6m"] = (
    clv_data["expected_purchases_6m"] * 
    clv_data["expected_average_profit"] * 
    gross_profit_margin
)

# ==========================================
# 6. MERGE WITH OUR SEGMENTED RFM DATA
# ==========================================
rfm_segmented = pd.read_csv("rfm_segmented.csv")

# Handle column name variation from Step 1 ("Customer ID" vs "CustomerID")
if "Customer ID" in rfm_segmented.columns:
    rfm_segmented = rfm_segmented.rename(columns={"Customer ID": "CustomerID"})

# Normalize IDs (strips any ".0" so the merge matches perfectly)
rfm_segmented["CustomerID"] = rfm_segmented["CustomerID"].astype(float).astype(int).astype(str)

clv_results = clv_data[["expected_purchases_6m", "predicted_CLV_6m"]].reset_index()
clv_results = clv_results.rename(columns={"index": "CustomerID"})

final_df = rfm_segmented.merge(clv_results, on="CustomerID", how="left")
final_df["predicted_CLV_6m"] = final_df["predicted_CLV_6m"].fillna(0) # Single-purchase customers get 0 for this model

# ==========================================
# 7. BUSINESS INSIGHTS & EXPORT
# ==========================================
print("\nTop 10 Customers by Predicted 6-Month CLV:")
print(final_df.sort_values("predicted_CLV_6m", ascending=False)[["CustomerID", "Country", "Segment", "predicted_CLV_6m"]].head(10))

print("\nAverage Predicted 6-Month CLV per Segment:")
segment_clv = final_df.groupby("Segment")["predicted_CLV_6m"].mean().sort_values(ascending=False)
print(segment_clv)

final_df.to_csv("customers_with_clv.csv", index=False)
print("\nSUCCESS: customers_with_clv.csv saved!")

# Save chart for portfolio
plt.figure(figsize=(10, 6))
sns.barplot(x=segment_clv.index, y=segment_clv.values, palette="viridis")
plt.title("Average Predicted 6-Month CLV by Customer Segment")
plt.ylabel("Predicted CLV ($)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("clv_by_segment.png", dpi=150)
print("Saved: clv_by_segment.png")