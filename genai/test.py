from genai.shapExplainer import explain_prediction
from genai.recommendation import generate_recommendations

# Sample user input
user_data = {
    "Age": 27,
    "Gender": "Female",
    "Country": "India",
    "Academic_Level": "Undergraduate",
    "Most_Used_Platform": "YouTube",
    "Purpose_Of_Use": "Entertainment",
    "Avg_Daily_Usage_Hours": 6.5,
    "Daily_Unlocks": 80,
    "Study_Hours": 5,
    "Physical_Activity_Hours": 1,
    "Sleep_Hours_Per_Night": 6,
    "Stress_Level": "Medium"
}

# Step 1: Generate SHAP explanation
shap_result = explain_prediction(user_data)

print("\nSHAP RESULT:")
print(shap_result)

# Step 2: Generate Gemini recommendations
ai_result = generate_recommendations(
    user_data,
    shap_result
)

print("\nUNPLUG AI RECOMMENDATIONS:")
print(ai_result["recommendations"])
