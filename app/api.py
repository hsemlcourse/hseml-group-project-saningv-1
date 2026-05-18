from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model_service import load_model, predict_message

app = FastAPI(title="SMS Spam Detection API", version="1.0.0")
MODEL = load_model()


class PredictionRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=["Free entry! Call now to claim your prize"])


class PredictionResponse(BaseModel):
    label: str
    target: int
    spam_score: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        prediction = predict_message(request.message, MODEL)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return PredictionResponse(
        label=prediction.label,
        target=prediction.target,
        spam_score=prediction.spam_score,
    )
