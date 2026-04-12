import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import traceback


app = FastAPI(title="Bank Marketing Predictor")

# Cargar el pipeline completo (preprocesador + modelo)
BASE_DIR = Path(__file__).resolve().parent
pipeline = joblib.load(BASE_DIR / "model.pkl")

class ClientData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str       # "yes" o "no"
    balance: float
    housing: str       # "yes" o "no"
    loan: str          # "yes" o "no"
    contact: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str

@app.get("/")
def root():
    return {"message": "Bank Marketing Predictor API"}

@app.post("/predict")
def predict(data: ClientData):
    try:
        df = pd.DataFrame([data.dict()])
        
        for col in ["default", "housing", "loan"]:
            df[col] = df[col].map({"yes": 1, "no": 0})
        
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0][1]
        
        return {
            "prediction": prediction,
            "subscribed": bool(prediction == "yes"),
            "probability_yes": round(float(probability), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}