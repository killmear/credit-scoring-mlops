from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

from src.config import ROOT
from src.validation import FEATURES, validate_frame


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "records": [
                        {
                            "LIMIT_BAL": 100000,
                            "SEX": 2,
                            "EDUCATION": 2,
                            "MARRIAGE": 1,
                            "AGE": 35,
                            "PAY_0": 0,
                            "PAY_2": 0,
                            "PAY_3": 0,
                            "PAY_4": 0,
                            "PAY_5": 0,
                            "PAY_6": 0,
                            "BILL_AMT1": 20000,
                            "BILL_AMT2": 18000,
                            "BILL_AMT3": 16000,
                            "BILL_AMT4": 14000,
                            "BILL_AMT5": 12000,
                            "BILL_AMT6": 10000,
                            "PAY_AMT1": 3000,
                            "PAY_AMT2": 3000,
                            "PAY_AMT3": 3000,
                            "PAY_AMT4": 3000,
                            "PAY_AMT5": 3000,
                            "PAY_AMT6": 3000,
                        }
                    ]
                }
            ]
        },
    )
    records: list[dict[str, StrictInt | StrictFloat]] = Field(min_length=1, max_length=1000)


@asynccontextmanager
async def lifespan(app):
    app.state.model = joblib.load(ROOT / "models/model.joblib")
    yield


app = FastAPI(title="Учебный кредитный скоринг", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "src/static"), name="static")


@app.get("/", include_in_schema=False)
@app.get("/details", include_in_schema=False)
def demo_page():
    return FileResponse(ROOT / "src/static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/predict",
    summary="Получить прогноз дефолта",
    description=(
        "Нажмите Try it out → Execute для проверки примера. "
        "probability — вероятность дефолта; class — 1 при вероятности ≥ 0.5, иначе 0."
    ),
)
def predict(request: PredictionRequest):
    frame = pd.DataFrame(request.records)
    try:
        validate_frame(frame)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    probabilities = app.state.model.predict_proba(frame[FEATURES])[:, 1]
    return {
        "predictions": [{"class": int(p >= 0.5), "probability": float(p)} for p in probabilities]
    }
