import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# ==========================================
# 1. LOAD RFM DATA
# ==========================================
rfm = pd.read_csv("rfm.csv")
print("Customers loaded:", len(rfm))

# ==========================================
# 2. HANDLE OUTLIERS (Log Transformation)
# ==========================================
rfm['Recency_log'] = np.log1p(rfm['Recency'])
rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
rfm['Monetary_log'] = np.log1p(rfm['Monetary'])

features = ['Recency_log', 'Frequency_log', 'Monetary_log']

# ==========================================
# 3. SCALE THE FEATURES
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[features])

# ==========================================
# 4. FIND OPTIMAL NUMBER OF CLUSTERS
# ==========================================
K = range(2, 9)
inertias = []
sil_scores = []

for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
    print(f"k={k} | Inertia: {km.inertia_:.0f} | Silhouette: {sil_scores[-1]:.3f}")

# Save Elbow + Silhouette chart for portfolio
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(K), inertias, marker='o')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('Clusters (k)')
axes[1].plot(list(K), sil_scores, marker='o', color='green')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('Clusters (k)')
plt.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=150)
print("\nSaved: elbow_silhouette.png")

# ==========================================
# 5. RUN FINAL K-MEANS
# ==========================================
best_k = list(K)[int(np.argmax(sil_scores))]
print(f"Best k by silhouette score: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# ==========================================
# 6. PROFILE & LABEL THE SEGMENTS
# ==========================================
r_med = rfm['Recency'].median()
f_med = rfm['Frequency'].median()
m_med = rfm['Monetary'].median()

profile = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()

def label_cluster(row):
    if row['Recency'] < r_med and row['Frequency'] > f_med and row['Monetary'] > m_med:
        return "Champions"
    if row['Recency'] < r_med and row['Frequency'] > f_med:
        return "Loyal Customers"
    if row['Recency'] < r_med:
        return "New / Potential"
    if row['Frequency'] > f_med:
        return "At Risk (Valuable but Sleeping)"
    return "Hibernating / Lost"

profile['Segment'] = profile.apply(label_cluster, axis=1)
print("\nCluster Profile:")
print(profile)

# Map segment names back to each customer
rfm['Segment'] = rfm['Cluster'].map(profile.set_index('Cluster')['Segment'])

print("\nSegment Sizes:")
print(rfm['Segment'].value_counts())

# Save segmented data + scatter chart for portfolio
rfm.to_csv("rfm_segmented.csv", index=False)
print("\nSaved: rfm_segmented.csv")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Segment', palette='viridis', s=40, alpha=0.7)
plt.title('Customer Segments: Recency vs Monetary')
plt.savefig("segments_scatter.png", dpi=150)
print("Saved: segments_scatter.png")

print("\nSUCCESS: Step 2 complete!")