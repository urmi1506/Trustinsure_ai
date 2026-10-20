import numpy as np
import shap
import tensorflow as tf

from src.preprocessing.preprocess import preprocess
from src.preprocessing.preprocess import preprocess_new_data


MODEL_VERSION = "v1"

MODEL_PATH = (
    f"src/artifacts/{MODEL_VERSION}/insurance_model.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

# Get training data for SHAP background
X_train, X_test, y_train, y_test, preprocessor = preprocess()

feature_names = preprocessor.get_feature_names_out()

# Use a small background dataset
background = X_train[:100]

explainer = shap.DeepExplainer(
    model,
    background
)


def explain_prediction(customer_data):
    """
    Generate SHAP feature contributions
    for one customer.
    """

    # Apply the SAME preprocessing used by prediction
    X_customer = preprocess_new_data(
        customer_data,
        preprocessor
    )

    shap_values = explainer.shap_values(X_customer)

    shap_values = np.array(shap_values)

    # Handle different SHAP output shapes
    values = shap_values.reshape(-1)

    explanation = []

    for feature, contribution in zip(feature_names, values):
        explanation.append({
            "feature": feature,
            "impact": float(contribution)
        })

    # Sort by absolute impact
    explanation.sort(
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    return explanation