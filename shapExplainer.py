import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
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

feature_names = (
    ["Study_Hours"]
    + ["Age", "Avg_Daily_Usage_Hours", "Daily_Unlocks",
       "Physical_Activity_Hours", "Sleep_Hours_Per_Night"]
    + ["Stress_Level"]
    + list(
        preprocessor.named_transformers_["normal"]
        .named_steps["onehot"]
        .get_feature_names_out([
            "Gender",
            "Academic_Level",
            "Most_Used_Platform",
            "Purpose_Of_Use",
            "Grouped_Countries"
        ])
    )
)

bg = shap.sample(X_transformed,100,random_state = 42)
explainer = shap.TreeExplainer(rf_model, data=bg,feature_perturbation="interventional")

#explainer for user data
def explain_prediction(user_data):

    user_df = pd.DataFrame([user_data])

    user_df["Grouped_Countries"] = user_df["Country"].apply(
        lambda c: c if c in top_countries else "Other"
    )
    user_df = user_df[expected_features]

    transformed_input = preprocessor.transform(user_df)

    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()

    prediction = model.predict(user_df)[0]

    values = explainer.shap_values(transformed_input)[0]

    top_indices = np.argsort(np.abs(values))[::-1][:10]

    top_features = []

    for i in top_indices:
        top_features.append({
            "feature": feature_names[i],
            "contribution": round(float(values[i]), 4)
        })

    return {
        "predicted_score": round(float(prediction), 4),
        "baseline": round(float(explainer.expected_value), 4),
        "top_features": top_features
    }

#testing
test_user = df.iloc[100].to_dict()

result = explain_prediction(test_user)

print("\nPrediction:", result["predicted_score"])
print("Baseline:", result["baseline"])
print("\nTop influencing features:")

for feature in result["top_features"]:
    print(feature)
