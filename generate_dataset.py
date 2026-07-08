"""
generate_dataset.py
Generates a synthetic customer churn dataset with demographic,
usage, and engagement features for the Customer Churn Prediction project.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 3000

def generate_data(n=N):
    data = {}

    # Demographic features
    data["customer_id"] = [f"CUST{1000 + i}" for i in range(n)]
    data["age"] = np.random.randint(18, 70, size=n)
    data["gender"] = np.random.choice(["Male", "Female"], size=n)
    data["region"] = np.random.choice(
        ["North", "South", "East", "West"], size=n
    )

    # Account / subscription features
    data["tenure_months"] = np.random.randint(1, 72, size=n)
    data["subscription_type"] = np.random.choice(
        ["Basic", "Standard", "Premium"], size=n, p=[0.4, 0.4, 0.2]
    )
    data["monthly_charges"] = np.round(
        np.random.normal(loc=55, scale=20, size=n).clip(10, 150), 2
    )

    # Usage features
    data["avg_sessions_per_week"] = np.round(
        np.random.exponential(scale=4, size=n).clip(0, 30), 1
    )
    data["avg_session_duration_min"] = np.round(
        np.random.normal(loc=20, scale=8, size=n).clip(1, 90), 1
    )
    data["support_tickets_raised"] = np.random.poisson(lam=1.2, size=n)

    # Engagement features
    data["email_open_rate"] = np.round(np.random.uniform(0, 1, size=n), 2)
    data["app_login_frequency"] = np.random.randint(0, 30, size=n)
    data["has_autopay"] = np.random.choice([0, 1], size=n, p=[0.35, 0.65])
    data["used_promo_offer"] = np.random.choice([0, 1], size=n, p=[0.7, 0.3])

    df = pd.DataFrame(data)

    # ---- Build a churn probability from a weighted combination of features ----
    churn_score = (
        -0.03 * df["tenure_months"]
        + 0.015 * df["monthly_charges"]
        - 0.08 * df["avg_sessions_per_week"]
        - 0.02 * df["avg_session_duration_min"]
        + 0.25 * df["support_tickets_raised"]
        - 1.2 * df["email_open_rate"]
        - 0.05 * df["app_login_frequency"]
        - 0.8 * df["has_autopay"]
        - 0.4 * df["used_promo_offer"]
        + np.where(df["subscription_type"] == "Basic", 0.6, 0)
        + np.where(df["subscription_type"] == "Premium", -0.5, 0)
        + np.random.normal(0, 1, size=n)  # noise
    )

    # Convert score to probability via sigmoid, then sample churn
    prob = 1 / (1 + np.exp(-churn_score))
    df["churn"] = np.random.binomial(1, prob)

    return df


if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/customer_churn.csv", index=False)
    print(f"Generated dataset with {len(df)} rows.")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    print("Saved to data/customer_churn.csv")
