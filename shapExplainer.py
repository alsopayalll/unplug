
import joblib
import pandas as pd
import numpy as np
import shap

from pathlib import Path


# ==========================================
# FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "mentalHealth.csv"
MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"


# ==========================================
# LOAD DATA AND MODEL
# ==========================================

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)

top_countries = [
    "Other",
    "India",
    "USA",
    "Canada",
    "Australia",
    "UK",
    "Germany",
    "Mexico",
    "Turkey",
    "France"
]


# ==========================================
# PREPARE TRAINING FEATURES
# ==========================================

df["Grouped_Countries"] = df["Country"].apply(
    lambda country: (
        country if country in top_countries else "Other"
    )
)

expected_features = [
    "Study_Hours",
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
    "Stress_Level",
    "Gender",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Grouped_Countries"
]

X = df[expected_features]


# ==========================================
# GET PIPELINE COMPONENTS
# ==========================================

preprocessor = model.named_steps["preprocessor"]

rf_model = model.named_steps["rf_model"]


# ==========================================
# TRANSFORM TRAINING DATA
# ==========================================

X_transformed = preprocessor.transform(X)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_transformed = np.asarray(X_transformed)


# ==========================================
# GET FEATURE NAMES
# ==========================================

# ==========================================
# GET FEATURE NAMES
# ==========================================

# Get original columns handled by each transformer
skewed_columns = list(
    preprocessor.transformers_[0][2]
)

numeric_columns = list(
    preprocessor.transformers_[1][2]
)

ordinal_columns = list(
    preprocessor.transformers_[2][2]
)

normal_columns = list(
    preprocessor.transformers_[3][2]
)


# Get OneHotEncoder from normal pipeline
onehot_encoder = (
    preprocessor.named_transformers_["normal"]
    .named_steps["onehot"]
)

# Generate names for one-hot encoded features
onehot_feature_names = list(
    onehot_encoder.get_feature_names_out(normal_columns)
)


# Combine names in the SAME order as ColumnTransformer
feature_names = (
    skewed_columns
    + numeric_columns
    + ordinal_columns
    + onehot_feature_names
)


# Verify feature names match transformed data
if len(feature_names) != X_transformed.shape[1]:

    raise ValueError(
        f"Feature name mismatch: "
        f"{len(feature_names)} names but "
        f"{X_transformed.shape[1]} transformed columns."
    )


print("SHAP feature names loaded:", len(feature_names))


# Check feature name count
if len(feature_names) != X_transformed.shape[1]:

    feature_names = [
        f"Feature {i + 1}"
        for i in range(X_transformed.shape[1])
    ]


# ==========================================
# CREATE SHAP EXPLAINER
# ==========================================

background_size = min(100, len(X_transformed))

background_data = shap.sample(
    X_transformed,
    background_size,
    random_state=42
)

explainer = shap.TreeExplainer(
    rf_model,
    data=background_data,
    feature_perturbation="interventional"
)


# ==========================================
# EXPLAIN USER PREDICTION
# ==========================================

def explain_prediction(user_data):

    # Convert dictionary into DataFrame
    user_df = pd.DataFrame([user_data])

    # Group countries using the same rule
    user_df["Grouped_Countries"] = user_df["Country"].apply(
        lambda country: (
            country if country in top_countries else "Other"
        )
    )

    # Keep exact training feature order
    user_df = user_df[expected_features]

    # Apply existing preprocessing pipeline
    transformed_input = preprocessor.transform(user_df)

    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()

    transformed_input = np.asarray(transformed_input)

    # Generate prediction using the complete pipeline
    prediction = float(model.predict(user_df)[0])

    # Calculate SHAP values
    shap_values = explainer.shap_values(transformed_input)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[0, :, 0]
    else:
        shap_values = shap_values.reshape(-1)

    # Get baseline
    baseline = np.asarray(
        explainer.expected_value
    ).reshape(-1)[0]

    # Check dimensions
    if len(shap_values) != len(feature_names):
        raise ValueError(
            "SHAP values and transformed feature names do not match."
        )

    # Get 10 most influential features
    top_indices = np.argsort(
        np.abs(shap_values)
    )[::-1][:10]

    top_features = []

    for index in top_indices:

        top_features.append({

            "feature": str(feature_names[index]),

            "contribution": round(
                float(shap_values[index]), 4
            )

        })

    return {

        "predicted_score": round(prediction, 4),

        "baseline": round(float(baseline), 4),

        "top_features": top_features

    }
