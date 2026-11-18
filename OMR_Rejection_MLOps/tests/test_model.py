
import sys
from pathlib import Path

import pytest
import joblib
import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "omr_model.pkl"
)

model = joblib.load(MODEL_PATH)


def test_scan_resolution():

    assert 100 <= 300 <= 1200


def test_rotation_range():

    assert -15 <= 2.5 <= 15


def test_contrast_range():

    assert 0 <= 80 <= 100


def test_alignment_range():

    assert 0 <= 1 <= 1


def test_model_prediction():

    sample = pd.DataFrame([{
        "scan_resolution_dpi": 300,
        "rotation_angle": 2.5,
        "print_contrast_score": 80,
        "alignment_marker_visibility": 1,
        "multiple_marked_questions": 1,
        "blank_answer_percentage": 5,
        "sheet_damage_score": 10,
        "scanner_type": "Scanner_A",
        "scan_quality_index": 0.85,
        "marking_error_ratio": 0.06
    }])

    prediction = model.predict(sample)[0]

    assert prediction in [0, 1]
