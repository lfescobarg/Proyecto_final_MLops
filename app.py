import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
from typing import Optional


app = FastAPI(title="Bank Marketing Predictor")

# Cargar el pipeline completo (preprocesador + modelo)
BASE_DIR = Path(__file__).resolve().parent
pipeline = joblib.load(BASE_DIR / "model.pkl")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
MODEL_FEATURES = [
    "age",
    "balance",
    "campaign_log",
    "pdays_clean",
    "previous",
    "job",
    "marital",
    "education",
    "contact",
    "season",
    "poutcome",
    "housing",
    "loan",
    "prev_success",
]


def season_mapper(month: str) -> str:
    if month in ["dec", "jan", "feb"]:
        return "winter"
    if month in ["mar", "apr", "may"]:
        return "spring"
    if month in ["jun", "jul", "aug"]:
        return "summer"
    return "fall"

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
    month: Optional[str] = None
    season: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="bank_marketing.html",
        context={},
    )


@app.get("/api")
def root():
    return {"message": "Bank Marketing Predictor API"}

@app.post("/predict")
def predict(data: ClientData):
    try:
        df = pd.DataFrame([data.model_dump()])
        
        for col in ["housing", "loan"]:
            df[col] = df[col].map({"yes": 1, "no": 0})

        # Recreate the engineered features used during training.
        df["campaign_log"] = np.log1p(df["campaign"])
        df["pdays_clean"] = df["pdays"].replace(999, np.nan).astype(float)
        df["previous"] = df["previous"].astype(float)
        df["prev_success"] = (
            (df["previous"] > 0) & (df["poutcome"] == "success")
        ).astype(int)

        if df["month"].notna().any():
            df["season"] = df["month"].str.lower().map(season_mapper)
        elif df["season"].isna().any():
            raise HTTPException(
                status_code=400,
                detail="Debes enviar 'month' o 'season' para construir la variable 'season'.",
            )

        df = df[MODEL_FEATURES]
        
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
