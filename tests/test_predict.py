from pathlib import Path
import joblib
from src.predict import predict_default

PROJECT_ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=PROJECT_ROOT/"model"/"credti_default_pipeline.joblib"

print("Project root:", PROJECT_ROOT)
print("Model path:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())

def test_pred_output():
    test_model=joblib.load(MODEL_PATH)

    customer={
  "LIMIT_BAL": 120000,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 39,
  "PAY_0": -1,
  "PAY_2": -1,
  "PAY_3": -1,
  "PAY_4": -1,
  "PAY_5": -1,
  "PAY_6": -1,
  "BILL_AMT1": 316,
  "BILL_AMT2": 316,
  "BILL_AMT3": 316,
  "BILL_AMT4": 0,
  "BILL_AMT5": 632,
  "BILL_AMT6": 316,
  "PAY_AMT1": 316,
  "PAY_AMT2": 316,
  "PAY_AMT3": 0,
  "PAY_AMT4": 632,
  "PAY_AMT5": 316,
  "PAY_AMT6": 0
}

    result=predict_default(test_model,customer)
    assert "default_probability" in result
    assert 0<= result["default_probability"]<=1
    assert result["flag_for_review"] == (
    result["default_probability"] >= 0.45)

    