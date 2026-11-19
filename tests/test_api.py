from fastapi.testclient import TestClient
from app import app

client=TestClient(app)

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

def test_health():
    response=client.get("/health")

    assert response.status_code==200
    assert response.json()["status"]=="ok"
    assert response.json()["model_loaded"] is True



def test_valid_prediction():

    response=client.post("/predict",json=customer)

    assert response.status_code==200
    result=response.json()

    assert 0<=result["default_probability"]<=1
    assert "flag_for_review" in result
    assert result["flag_for_review"] == (
    result["default_probability"] >= 0.45)


def test_invalid_prediction():
    invalid_customer=customer.copy()
    invalid_customer["AGE"]="hello"

    response=client.post("/predict",json=invalid_customer)

    assert response.status_code==422