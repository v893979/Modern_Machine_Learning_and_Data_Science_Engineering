"""
Chapter 1 & 2: Python Foundations, Syntax, and Data Collections
Modern Machine Learning & Data Science Engineering
"""

def demonstrate_python_collections():
    print("--- 1. List Comprehensions & Transformations ---")
    prices = [12.50, 45.00, 120.00, 8.99, 65.40]
    discounted = [round(p * 0.90, 2) if p > 50.0 else p for p in prices]
    print(f"Original Prices:   {prices}")
    print(f"Discounted Prices: {discounted}")

    print("\n--- 2. Dictionary Lookups & Aggregation ---")
    loan_applicants = [
        {"id": "A101", "income": 75000, "debt": 18000, "approved": True},
        {"id": "A102", "income": 42000, "debt": 25000, "approved": False},
        {"id": "A103", "income": 110000, "debt": 22000, "approved": True},
    ]
    for app in loan_applicants:
        dti = round(app["debt"] / app["income"], 4)
        app["dti"] = dti
        print(f"Applicant {app['id']} | Income: ${app['income']:,} | DTI: {dti:.2%} | Approved: {app['approved']}")

if __name__ == "__main__":
    demonstrate_python_collections()
