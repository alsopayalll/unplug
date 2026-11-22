import pandas as pd
import numpy as np
import joblib
import shap

df = pd.read_csv("mentalHealth.csv")

top_countries = [
    "Other", "India", "USA", "Canada", "Australia",
    "UK", "Germany", "Mexico", "Turkey", "France"
]

df["Grouped_Countries"] = df["Country"].apply(
    lambda country: country if country in top_countries else "Other"
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

model = joblib.load("Mental_Health_Model.pkl")

preprocessor = model.named_steps["preprocessor"]
rf_model = model.named_steps["rf_model"]

X_transformed = preprocessor.transform(X)
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

feature_names = [
    "Study_Hours",
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
    "Stress_Level"
]

onehot_features = (
    preprocessor
    .named_transformers_["normal"]
    .named_steps["onehot"]
    .get_feature_names_out([
        "Gender",
        "Academic_Level",
        "Most_Used_Platform",
        "Purpose_Of_Use",
        "Grouped_Countries"
    ])
)

feature_names.extend(onehot_features)

background_data = shap.sample(
    X_transformed,
    100,
    random_state=42
)

explainer = shap.TreeExplainer(
    rf_model,
    data=background_data,
    feature_perturbation="interventional"
)

def explain_prediction(user_data):
    user_df = pd.DataFrame([user_data])
    user_df["Grouped_Countries"] = user_df["Country"].apply(
        lambda country: country if country in top_countries else "Other"
    )
    user_df = user_df[expected_features]
    transformed_input = preprocessor.transform(user_df)

    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()

    predicted_score = model.predict(user_df)[0]

    shap_values = explainer.shap_values(transformed_input)[0]

    baseline = float(np.array(explainer.expected_value).flatten()[0])

    top_indices = np.argsort(np.abs(shap_values))[-10:][::-1]

    top_features = []

    for index in top_indices:
        top_features.append({
            "feature": feature_names[index],
            "contribution": float(shap_values[index])
        })

    return {
        "predicted_score": float(predicted_score),
        "baseline": baseline,
        "top_features": top_features
    }

if __name__ == "__main__":

    test_user = df.iloc[100].to_dict()

    result = explain_prediction(test_user)

    print("\nPredicted Score:", result["predicted_score"])
    print("SHAP Baseline:", result["baseline"])

    print("\nTop Influential Features:")

    for item in result["top_features"]:
        print(
            item["feature"],
            ":", item["contribution"]
        )
