from typing import List
import joblib
import numpy as np
import pandas as pd
from analysis_engine import XAUUSDAnalysisEngine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="XAUUSD AI Trading & Analysis API",
    version="2.2",
    description=(
        "Lightweight backend serving pre-trained XGBoost model with"
        " multi-timeframe top-3 probability ranking and market diagnostics."
    ),
)

# Load the pre-trained model artifact safely on startup
model = None


@app.on_event("startup")
def load_model_artifact():
  global model
  try:
    model = joblib.load("model.joblib")
    print("Success: Pre-trained model.joblib loaded into RAM.")
  except Exception as e:
    print(f"Warning: Could not load model.joblib: {e}")
    model = None


class PredictionRequest(BaseModel):
  ema_10: float


class MultiTFRequest(BaseModel):
  timeframe: str
  ema_10: float


@app.get("/")
def read_root():
  return {
      "status": "online",
      "system": "XAUUSD AI Trading Engine",
      "model_loaded": model is not None,
  }


@app.get("/api/diagnostics")
def get_diagnostics():
  """Runs market diagnostics using your XAUUSDAnalysisEngine."""
  try:
    engine = XAUUSDAnalysisEngine()
    diagnostics = engine.get_full_diagnostic()
    return diagnostics
  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f"Diagnostic error: {str(e)}"
    )


@app.post("/api/predict")
def predict(payload: PredictionRequest):
  """Performs real-time online inference using the EMA 10 feature."""
  if model is None:
    raise HTTPException(
        status_code=503,
        detail="Model artifact not loaded or unavailable on server.",
    )

  try:
    X_input = pd.DataFrame([[payload.ema_10]], columns=["ema_10"])
    prediction = int(model.predict(X_input)[0])
    probabilities = (
        model.predict_proba(X_input)[0].tolist()
        if hasattr(model, "predict_proba")
        else []
    )

    return {
        "prediction": prediction,
        "signal": "BUY / BULLISH" if prediction == 1 else "SELL / BEARISH",
        "probabilities": probabilities,
        "feature_used": {"ema_10": payload.ema_10},
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/predict_top_setups")
def predict_top_setups(
    payload: List[MultiTFRequest], min_confidence: float = 0.50
):
  """Evaluates multiple timeframes/models in batch and returns the top 3 highest probability trade setups."""
  if model is None:
    raise HTTPException(
        status_code=503, detail="Model artifact unavailable on server."
    )

  results = []
  for item in payload:
    try:
      X_input = pd.DataFrame([[item.ema_10]], columns=["ema_10"])
      probabilities = (
          model.predict_proba(X_input)[0].tolist()
          if hasattr(model, "predict_proba")
          else [0.5, 0.5]
      )

      prediction = int(np.argmax(probabilities))
      highest_prob = max(probabilities)

      if highest_prob >= min_confidence:
        results.append({
            "timeframe": item.timeframe,
            "prediction": prediction,
            "signal": "BUY" if prediction == 1 else "SELL",
            "probability": round(float(highest_prob), 4),
            "ema_10": item.ema_10,
        })
    except Exception:
      continue  # Skip invalid items gracefully

  # Sort by highest probability descending and grab top 3
  top_3 = sorted(results, key=lambda x: x["probability"], reverse=True)[:3]

  return {
      "total_evaluated": len(payload),
      "selected_count": len(top_3),
      "top_trades": top_3,
  }


@app.post("/api/retrain")
def retrain_model():
  """Placeholder endpoint to prevent EA 404 errors during offline workflow."""
  return {
      "status": "disabled",
      "message": (
          "Server-side training is disabled. Use offline training locally and"
          " push model.joblib."
      ),
  }