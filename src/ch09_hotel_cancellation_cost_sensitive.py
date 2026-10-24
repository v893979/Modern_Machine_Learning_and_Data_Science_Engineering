"""
Chapter 9: Cost-Sensitive Learning & Threshold Tuning (Hotel Reservations)
Modern Machine Learning & Data Science Engineering
"""
import os
import csv
import numpy as np

def run_cost_sensitive_classification():
    print("=====================================================================")
    print("   COST-SENSITIVE MACHINE LEARNING & THRESHOLD OPTIMIZATION          ")
    print("=====================================================================")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "hotel_reservations.csv")

    lead_times = []
    statuses = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lead_times.append(float(row["lead_time"]))
            statuses.append(1 if row["booking_status"] == "Canceled" else 0)

    y_true = np.array(statuses)
    lead_norm = np.array(lead_times) / 365.0
    print(f"Loaded {len(y_true)} hotel bookings. Cancellation rate: {y_true.mean():.2%}")

    # Business Cost Matrix:
    COST_FN = 150.0  # Lost room revenue
    COST_FP = 60.0   # Overbooking relocation fee

    # Model predicted probabilities
    np.random.seed(42)
    y_prob = np.clip(lead_norm * 0.7 + np.random.normal(0, 0.15, size=len(y_true)), 0.01, 0.99)

    print("\nEvaluating Business Loss across Decision Thresholds:")
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    best_thresh = 0.50
    min_loss = float("inf")

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        total_loss = (fn * COST_FN) + (fp * COST_FP)
        print(f"  Threshold {t:.2f} | FN: {fn:>3} (Vacancy) | FP: {fp:>3} (Overbooked) | Total Cost: ${total_loss:,.2f}")
        if total_loss < min_loss:
            min_loss = total_loss
            best_thresh = t

    print(f"\nOptimal Business Decision Threshold: {best_thresh:.2f} (Minimizes financial loss)")

if __name__ == "__main__":
    run_cost_sensitive_classification()
