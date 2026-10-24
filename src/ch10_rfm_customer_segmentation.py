"""
Chapter 10: Unsupervised Customer Segmentation (RFM Analysis & K-Means)
Modern Machine Learning & Data Science Engineering
"""
import os
import csv
import numpy as np

def run_rfm_segmentation():
    print("=====================================================================")
    print("   UNSUPERVISED CUSTOMER SEGMENTATION VIA RFM ANALYSIS               ")
    print("=====================================================================")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "customer_rfm.csv")

    cids, recs, freqs, mons = [], [], [], []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cids.append(row["customer_id"])
            recs.append(float(row["recency_days"]))
            freqs.append(float(row["frequency_purchases"]))
            mons.append(float(row["monetary_value"]))

    recs = np.array(recs)
    freqs = np.array(freqs)
    mons = np.array(mons)
    print(f"Loaded {len(cids)} customer profiles.")

    # Log transformation for right-skewed monetary spend
    X = np.column_stack([np.log1p(recs), np.log1p(freqs), np.log1p(mons)])
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

    # K-Means clustering
    np.random.seed(42)
    centroids = X_scaled[np.random.choice(len(X_scaled), size=4, replace=False)].copy()

    for iteration in range(20):
        dists = np.linalg.norm(X_scaled[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([X_scaled[labels == k].mean(axis=0) for k in range(4)])
        if np.allclose(centroids, new_centroids, atol=1e-4):
            print(f"K-Means converged in {iteration + 1} iterations.")
            break
        centroids = new_centroids

    print("\nCluster Profile Summary (Customer Personas):")
    for k in range(4):
        mask = (labels == k)
        cnt = np.sum(mask)
        avg_rec = np.mean(recs[mask])
        avg_freq = np.mean(freqs[mask])
        avg_mon = np.mean(mons[mask])
        
        if avg_mon > 3000: persona = "VIP Champions"
        elif avg_rec > 180: persona = "Dormant / Lost"
        elif avg_freq > 8: persona = "Loyal Regulars"
        else: persona = "Occasional Buyers"
        
        print(f"  Cluster {k} ({persona:<16}): {cnt:>3} customers | Recency: {avg_rec:5.1f}d | Freq: {avg_freq:4.1f} | Avg Spend: ${avg_mon:,.2f}")

if __name__ == "__main__":
    run_rfm_segmentation()
