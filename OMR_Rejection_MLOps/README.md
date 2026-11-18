# OMR Rejection Prediction using MLOps

## Aim
To predict whether an OMR answer sheet will be
accepted or rejected during automated scanning.

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- MLflow
- FastAPI
- GitHub Actions

## Dataset
- Version 1: 300 records
- Version 2: 300 records
- Total: 600 records

## Feature Engineering
1. scan_quality_index
2. marking_error_ratio

## Machine Learning Models
1. Logistic Regression
2. Random Forest

## Model Evaluation
Accuracy, Precision, Recall and F1-score.

## Model Registry
MLflow Model Registry.

## API
FastAPI provides the /predict endpoint.

## Output
- Scan Accepted
- Manual Verification Required

## Run Application

pip install -r requirements.txt

uvicorn api:app --reload

Open http://127.0.0.1:8000/docs

## Run Tests

pytest -v
