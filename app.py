from pathlib import Path
import joblib
from fastapi import FastAPI
import pandas as pd
from src.predict import predict_default
from pydantic import BaseModel,Field,ConfigDict

app=FastAPI(title="Credit Default Risk API")

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model"/"credti_default_pipeline.joblib"

model=joblib.load(MODEL_PATH)

class CustomerInput(BaseModel):
    model_config=ConfigDict(extra="forbid")

    LIMIT_BAL:float=Field(gt=0)
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int = Field(gt=0)

    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


@app.get("/health")
def health():
    return {"status":"ok","model_loaded":True}

@app.post("/predict")
def predict(customer:CustomerInput):
    customer_dict=customer.model_dump()
    prediction=predict_default(model,customer_dict)
    return prediction