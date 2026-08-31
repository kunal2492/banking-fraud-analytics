from pathlib import Path

import joblib
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fraud_detection_random_forest.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.joblib"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

def load_model():
    """Load the trained fraud model and its metadata."""

    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)

    return model, metadata


# --------------------------------------------------
# Fraud prediction
# --------------------------------------------------

def predict_fraud(transaction, model, metadata):
    """
    Predict fraud probability for one transaction.

    Parameters
    ----------
    transaction : dict
        Transaction information.

    model :
        Trained scikit-learn classifier.

    metadata : dict
        Model metadata including feature order
        and decision threshold.
    """

    transaction_df = pd.DataFrame([transaction])

    required_features = metadata["features"]

    missing_features = [
        feature
        for feature in required_features
        if feature not in transaction_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Preserve exact training feature order
    transaction_df = transaction_df[required_features]

    fraud_probability = model.predict_proba(
        transaction_df
    )[0, 1]

    threshold = metadata["threshold"]

    prediction = int(
        fraud_probability >= threshold
    )

    return {
        "fraud_probability": float(fraud_probability),
        "threshold": float(threshold),
        "prediction": prediction,
        "label": "FRAUD" if prediction == 1 else "LEGITIMATE",
    }


# --------------------------------------------------
# Example
# --------------------------------------------------

if __name__ == "__main__":

    model, metadata = load_model()

    fraud_transaction = {
        "step": 1,
        "amount": 181.00,
        "oldbalanceOrg": 181.00,
        "newbalanceOrig": 0.00,
        "oldbalanceDest": 0.00,
        "newbalanceDest": 0.00,
        "is_transfer": 1,
    }

    legitimate_transaction = {
        "step": 1,
        "amount": 56953.90,
        "oldbalanceOrg": 1942.02,
        "newbalanceOrig": 0.00,
        "oldbalanceDest": 0.00,
        "newbalanceDest": 63100.72,
        "is_transfer": 0,
    }

    for name, transaction in [
    ("Fraud Example", fraud_transaction),
    ("Legitimate Example", legitimate_transaction),
    ]:
        
        result = predict_fraud(
            transaction,
            model,
            metadata
        )

        print(f"\n{name}")
        print("=" * 40)
        print(
            f"Fraud probability: "
            f"{result['fraud_probability']:.4f}"
        )
        print(
            f"Decision threshold: "
            f"{result['threshold']:.2f}"
        )
        print(
            f"Prediction: "
            f"{result['label']}"
        )