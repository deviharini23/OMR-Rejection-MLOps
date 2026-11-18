
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="OMR Rejection Prediction API",
    description="Predicts OMR sheet scanning rejection"
)

model = joblib.load("omr_model.pkl")


class OMRInput(BaseModel):

    scan_resolution_dpi: int = Field(ge=100, le=1200)

    rotation_angle: float = Field(ge=-15, le=15)

    print_contrast_score: float = Field(ge=0, le=100)

    alignment_marker_visibility: int = Field(ge=0, le=1)

    multiple_marked_questions: int = Field(ge=0, le=100)

    blank_answer_percentage: float = Field(ge=0, le=100)

    sheet_damage_score: float = Field(ge=0, le=100)

    scanner_type: str


@app.get("/")
def home():

    return {"message": "OMR Prediction API is running"}


@app.post("/predict")
def predict(data: OMRInput):

    row = pd.DataFrame([data.model_dump()])

    dpi_score = (row["scan_resolution_dpi"] / 600).clip(0, 1)

    contrast_score = row["print_contrast_score"] / 100

    rotation_score = (
        1 - row["rotation_angle"].abs() / 10
    ).clip(0, 1)

    alignment_score = row["alignment_marker_visibility"]

    damage_score = 1 - row["sheet_damage_score"] / 100

    row["scan_quality_index"] = (
        0.20 * dpi_score +
        0.25 * contrast_score +
        0.20 * rotation_score +
        0.25 * alignment_score +
        0.10 * damage_score
    )

    row["marking_error_ratio"] = (
        row["multiple_marked_questions"] +
        row["blank_answer_percentage"]
    ) / 100

    prediction = int(model.predict(row)[0])

    if prediction == 1:

        result = "Manual Verification Required"

    else:

        result = "Scan Accepted"

    return {
        "prediction": prediction,
        "result": result
    }
