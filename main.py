
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "Mental_Health_Model.pkl")

top_countries = [
    'Other', 'India', 'USA', 'Canada', 'Australia',
    'UK', 'Germany', 'Mexico', 'Turkey', 'France'
]

app = FastAPI(title="Unplug - Mental Health Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this before production
    allow_methods=["*"],
    allow_headers=["*"],
)

class StudentData(BaseModel):

    age: int = Field(..., ge=10, le=100)

    gender: Literal['Male', 'Female']

    country: str

    academic_level: Literal[
        'Undergraduate', 'Graduate', 'High School'
    ]

    most_used_platform: Literal[
        'Facebook', 'LinkedIn', 'Instagram', 'Snapchat',
        'Twitter', 'YouTube', 'TikTok', 'LINE',
        'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat'
    ]

    purpose_of_use: Literal[
        'Networking', 'Education', 'Entertainment', 'News'
    ]

    avg_daily_usage_hours: float = Field(..., ge=0, le=24)

    daily_unlocks: int = Field(..., ge=0)

    study_hours: float = Field(..., ge=0, le=24)

    physical_activity_hours: float = Field(..., ge=0, le=24)

    sleep_hours_per_night: float = Field(..., ge=0, le=24)

    stress_level: Literal[
        'Medium', 'Low', 'Very High', 'High'
    ]


class PredictionResponse(BaseModel):
    predicted_score: float

@app.get("/")
def home():
    return {
        "message": "Unplug Mental Health Prediction API is running"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentData):

    # Group countries
    country_group = (
        data.country
        if data.country in top_countries
        else "Other"
    )

    # Convert input into DataFrame
    input_rows = pd.DataFrame([{

        'Age': data.age,
        'Gender': data.gender,
        'Country': data.country,
        'Academic_Level': data.academic_level,
        'Most_Used_Platform': data.most_used_platform,
        'Purpose_Of_Use': data.purpose_of_use,
        'Avg_Daily_Usage_Hours': data.avg_daily_usage_hours,
        'Daily_Unlocks': data.daily_unlocks,
        'Study_Hours': data.study_hours,
        'Physical_Activity_Hours': data.physical_activity_hours,
        'Sleep_Hours_Per_Night': data.sleep_hours_per_night,
        'Stress_Level': data.stress_level,
        'Grouped_Countries': country_group

    }])

    # Generate prediction
    prediction = float(model.predict(input_rows)[0])

    return PredictionResponse(
        predicted_score=prediction
    )
