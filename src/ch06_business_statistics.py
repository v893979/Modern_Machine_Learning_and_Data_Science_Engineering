"""
Chapter 6: Business Statistics, Inference, Hypothesis Testing (t-Test & Chi-Square)
Modern Machine Learning & Data Science Engineering
"""
import numpy as np
from scipy import stats

def demonstrate_hypothesis_testing():
    print("--- 1. Two-Sample Independent t-Test (A/B Test) ---")
    np.random.seed(42)
    group_A = np.random.normal(loc=52.0, scale=12.0, size=150)
    group_B = np.random.normal(loc=56.5, scale=12.0, size=150)

    t_stat, p_val = stats.ttest_ind(group_A, group_B)
    print(f"Group A Mean: ${np.mean(group_A):.2f} (std={np.std(group_A):.2f})")
    print(f"Group B Mean: ${np.mean(group_B):.2f} (std={np.std(group_B):.2f})")
    print(f"t-Statistic: {t_stat:.4f} | p-Value: {p_val:.4e}")

    if p_val < 0.05:
        print("Result: Statistically significant difference! Reject H0 (p < 0.05).")
    else:
        print("Result: Fail to reject H0; no significant difference detected.")

    print("\n--- 2. Chi-Square Test of Independence ---")
    contingency = np.array([
        [180, 820],
        [240, 760]
    ])
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
    print(f"Chi-Square Stat: {chi2:.4f} | p-Value: {p_chi:.4e} | Degrees of Freedom: {dof}")

if __name__ == "__main__":
    demonstrate_hypothesis_testing()
