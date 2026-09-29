"""
generate_data.py
-----------------
Generates a realistic SYNTHETIC credit-scoring dataset because no dataset
was provided with the assignment.

If you'd rather use a REAL dataset instead, two good free options are:
  1. "Give Me Some Credit" (Kaggle)      -> https://www.kaggle.com/c/GiveMeSomeCredit
  2. UCI "Statlog (German Credit Data)"  -> https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

To switch to a real dataset, just drop the CSV into data/credit_data.csv
with a 'creditworthy' (0/1) target column and skip running this script.

Run:
    python generate_data.py
"""

import numpy as np
import pandas as pd
import os

RANDOM_STATE = 42
N_SAMPLES = 5000

np.random.seed(RANDOM_STATE)


def generate_synthetic_credit_data(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """Creates synthetic but realistic financial-history features and a
    creditworthiness label derived from them with some noise, mimicking
    how a real underwriting dataset behaves."""

    age = np.random.randint(21, 65, n_samples)
    annual_income = np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples).round(2)
    monthly_debt = (annual_income / 12 * np.random.uniform(0.05, 0.55, n_samples)).round(2)
    existing_loans = np.random.poisson(1.2, n_samples)
    credit_history_length = np.random.randint(0, 30, n_samples)  # years
    num_late_payments = np.random.poisson(1.5, n_samples)
    credit_utilization = np.clip(np.random.normal(0.35, 0.2, n_samples), 0, 1).round(3)
    num_credit_inquiries = np.random.poisson(1.0, n_samples)
    employment_length = np.random.randint(0, 40, n_samples)  # years
    savings_balance = np.random.lognormal(mean=8.5, sigma=1.2, size=n_samples).round(2)
    home_owner = np.random.binomial(1, 0.4, n_samples)
    loan_amount_requested = np.random.lognormal(mean=9.2, sigma=0.6, size=n_samples).round(2)

    debt_to_income = (monthly_debt * 12) / (annual_income + 1)

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    # Latent "creditworthiness score" built from a weighted, standardized
    # combination of sensible risk factors, then converted into a binary
    # label via a logistic link + noise so the classification task isn't
    # trivially linearly separable.
    score = (
        0.0
        + 0.9 * z(annual_income)
        - 1.1 * z(debt_to_income)
        - 0.7 * z(num_late_payments)
        - 0.9 * z(credit_utilization)
        - 0.4 * z(num_credit_inquiries)
        + 0.5 * z(credit_history_length)
        + 0.3 * z(employment_length)
        + 0.7 * z(savings_balance)
        + 0.3 * home_owner
        - 0.5 * z(loan_amount_requested / (annual_income + 1))
    )

    prob = 1 / (1 + np.exp(-score))
    noise = np.random.normal(0, 0.35, n_samples)
    creditworthy = ((prob + noise) > np.median(prob + noise)).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "monthly_debt": monthly_debt,
        "existing_loans": existing_loans,
        "credit_history_length_years": credit_history_length,
        "num_late_payments": num_late_payments,
        "credit_utilization": credit_utilization,
        "num_credit_inquiries": num_credit_inquiries,
        "employment_length_years": employment_length,
        "savings_balance": savings_balance,
        "home_owner": home_owner,
        "loan_amount_requested": loan_amount_requested,
        "debt_to_income_ratio": debt_to_income.round(3),
        "creditworthy": creditworthy,  # target: 1 = good credit risk, 0 = bad
    })

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_synthetic_credit_data()
    out_path = os.path.join("data", "credit_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset saved to {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Class balance:\n{df['creditworthy'].value_counts(normalize=True)}")
