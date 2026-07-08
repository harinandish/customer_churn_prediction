"""
predict.py
Load the trained churn model and run predictions on new customer data.
"""

import argparse
import joblib
import pandas as pd

MODEL_PATH = "models/churn_model.pkl"


def main():
    parser = argparse.ArgumentParser(description="Predict customer churn")
    parser.add_argument(
        "--input", required=True,
        help="Path to a CSV file with new customer records (same columns as training data, minus 'churn')"
    )
    parser.add_argument(
        "--output", default="output/predictions.csv",
        help="Path to save predictions"
    )
    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(args.input)

    ids = df["customer_id"] if "customer_id" in df.columns else None
    X = df.drop(columns=["customer_id"], errors="ignore")

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "customer_id": ids if ids is not None else range(len(df)),
        "churn_prediction": predictions,
        "churn_probability": probabilities.round(4)
    })

    result.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(result.head())


if __name__ == "__main__":
    main()
