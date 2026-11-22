
import os
import json
import pandas as pd
import joblib

from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from genai.shapExplainer import explain_prediction


# ==========================================
# ENVIRONMENT VARIABLES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")


# ==========================================
# LOAD ML MODEL
# ==========================================

model = joblib.load(
    BASE_DIR / "Mental_Health_Model.pkl"
)


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
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="Unplug - Mental Health Prediction API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# INPUT MODEL
# ==========================================

class StudentData(BaseModel):

    age: int = Field(..., ge=10, le=100)

    gender: Literal["Male", "Female"]

    country: str

    academic_level: Literal[
        "Undergraduate",
        "Graduate",
        "High School"
    ]

    most_used_platform: Literal[
        "Facebook",
        "LinkedIn",
        "Instagram",
        "Snapchat",
        "Twitter",
        "YouTube",
        "TikTok",
        "LINE",
        "KakaoTalk",
        "VKontakte",
        "WhatsApp",
        "WeChat"
    ]

    purpose_of_use: Literal[
        "Networking",
        "Education",
        "Entertainment",
        "News"
    ]

    avg_daily_usage_hours: float = Field(
        ..., ge=0, le=24
    )

    daily_unlocks: int = Field(
        ..., ge=0, le=500
    )

    study_hours: float = Field(
        ..., ge=0, le=24
    )

    physical_activity_hours: float = Field(
        ..., ge=0, le=24
    )

    sleep_hours_per_night: float = Field(
        ..., ge=0, le=24
    )

    stress_level: Literal[
        "Medium",
        "Low",
        "Very High",
        "High"
    ]


# ==========================================
# RESPONSE MODEL
# ==========================================

class PredictionResponse(BaseModel):

    predicted_score: float


# ==========================================
# CONVERT INPUT TO SHAP FORMAT
# ==========================================

def prepare_user_data(data: StudentData):

    country_group = (
        data.country
        if data.country in top_countries
        else "Other"
    )

    return {

        "Age": data.age,
        "Gender": data.gender,
        "Country": data.country,
        "Academic_Level": data.academic_level,
        "Most_Used_Platform": data.most_used_platform,
        "Purpose_Of_Use": data.purpose_of_use,
        "Avg_Daily_Usage_Hours": data.avg_daily_usage_hours,
        "Daily_Unlocks": data.daily_unlocks,
        "Study_Hours": data.study_hours,
        "Physical_Activity_Hours": data.physical_activity_hours,
        "Sleep_Hours_Per_Night": data.sleep_hours_per_night,
        "Stress_Level": data.stress_level

    }


# ==========================================
# FALLBACK RECOMMENDATIONS
# ==========================================

def get_fallback_recommendations(data: StudentData):

    recommendations = []

    if data.sleep_hours_per_night < 7:
        recommendations.append(
            "Try building a consistent sleep routine and allow enough time for rest."
        )

    if data.physical_activity_hours < 0.5:
        recommendations.append(
            "Consider adding a short walk or light physical activity to your daily routine."
        )

    if data.avg_daily_usage_hours > 6:
        recommendations.append(
            "Try taking regular screen breaks and setting specific times to disconnect from social media."
        )

    if data.study_hours > 8:
        recommendations.append(
            "Break long study sessions into smaller focused sessions with regular breaks."
        )

    if data.daily_unlocks > 100:
        recommendations.append(
            "Consider reducing unnecessary phone checking by keeping your phone away during focused activities."
        )

    if not recommendations:
        recommendations.append(
            "Continue maintaining a balanced routine with adequate sleep, physical activity and study breaks."
        )

    recommendations.append(
        "If stress or emotional difficulties are affecting your daily life, consider speaking with a trusted person or qualified mental health professional."
    )

    return recommendations[:5]


# ==========================================
# GEMINI RECOMMENDATION GENERATOR
# ==========================================

def generate_recommendations(data, shap_result):

    fallback = get_fallback_recommendations(data)

    if not GEMINI_API_KEY:
        return {
            "recommendations": fallback,
            "recommendation_source": "fallback",
            "recommendation_message": "Gemini API key is not configured."
        }

    try:

        from google import genai

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        student_information = data.model_dump()

        prompt = f"""
You are a supportive student wellness assistant for an
educational project called Unplug.

Generate personalized, practical and easy-to-follow
wellness recommendations using the student information
and the machine learning explanation below.

STUDENT INFORMATION:
{json.dumps(student_information, indent=2)}

MACHINE LEARNING PREDICTION AND SHAP EXPLANATION:
{json.dumps(shap_result, indent=2)}

INSTRUCTIONS:

1. Generate 3 to 5 personalized recommendations.
2. Consider the student's sleep, study hours, physical
   activity, screen usage and other relevant inputs.
3. Use SHAP contributions to understand which features
   influenced this individual prediction.
4. A positive SHAP value means the feature pushes the
   model prediction higher; a negative value means it
   pushes the prediction lower. Do not assume either
   direction automatically means better or worse health.
5. Do not claim that a feature causes stress.
6. Do not invent medical diagnoses or interpret the
   numerical score as a clinical diagnosis.
7. Recommendations should be realistic, supportive,
   specific and non-judgmental.
8. Avoid repeating the same recommendation.

Return ONLY valid JSON in this exact format:

{{
    "recommendations": [
        "First recommendation",
        "Second recommendation",
        "Third recommendation"
    ]
}}
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        response_text = response.text

        if not response_text:
            raise ValueError("Gemini returned an empty response.")

        # Remove Markdown code fences if present
        response_text = response_text.strip()

        if response_text.startswith("```"):
            response_text = response_text.replace(
                "```json", ""
            ).replace("```", "").strip()

        result = json.loads(response_text)

        recommendations = result.get("recommendations")

        if (
            not isinstance(recommendations, list)
            or not recommendations
            or not all(
                isinstance(item, str) and item.strip()
                for item in recommendations
            )
        ):
            raise ValueError("Invalid Gemini recommendation format.")

        return {
            "recommendations": recommendations[:5],
            "recommendation_source": "gemini",
            "recommendation_message": "Personalized recommendations generated successfully."
        }

    except Exception as error:

        print("Gemini recommendation error:", str(error))

        return {
            "recommendations": fallback,
            "recommendation_source": "fallback",
            "recommendation_message": "Gemini was unavailable. Showing fallback recommendations."
        }


@app.get("/")
def home():

    return {
        "message": "Unplug Mental Health Prediction API is running"
    }

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: StudentData):

    user_data = prepare_user_data(data)

    input_rows = pd.DataFrame([{
        **user_data,
        "Grouped_Countries": (
            user_data["Country"]
            if user_data["Country"] in top_countries
            else "Other"
        )
    }])

    prediction = float(
        model.predict(input_rows)[0]
    )

    return PredictionResponse(
        predicted_score=prediction
    )

@app.post("/explain")
def explain(data: StudentData):

    user_data = prepare_user_data(data)

    result = explain_prediction(user_data)

    return result

@app.post("/recommend")
def recommend(data: StudentData):

    # Step 1: Prepare student information
    user_data = prepare_user_data(data)

    # Step 2: Generate prediction and SHAP explanation
    shap_result = explain_prediction(user_data)

    # Step 3: Generate personalized recommendations
    recommendation_result = generate_recommendations(
        data,
        shap_result
    )

    # Step 4: Combine everything into one response
    return {
        **shap_result,
        **recommendation_result
    }
