from genai.llm import generate_response
from genai.prompt_template import create_recommendation_prompt
from genai.fallback import generate_fallback_recommendations

def generate_recommendations(user_data, shap_result):

    prediction = shap_result["predicted_score"]
    baseline = shap_result["baseline"]
    top_features = shap_result["top_features"]

    prompt = create_recommendation_prompt(
        user_data=user_data,
        prediction=prediction,
        baseline=baseline,
        top_features=top_features
    )

    try:
        #gemini
        recommendations = generate_response(prompt)

        return {
            "predicted_score": prediction,
            "baseline": baseline,
            "top_features": top_features,
            "recommendations": recommendations,
            "recommendation_source": "Gemini"
        }

    except Exception as error:

        print("Gemini unavailable:", error)
        print("Using local fallback recommendations...")

        #own rule based system
        recommendations = generate_fallback_recommendations(user_data)

        return {
            "predicted_score": prediction,
            "baseline": baseline,
            "top_features": top_features,
            "recommendations": recommendations,
            "recommendation_source": "Fallback"
        }