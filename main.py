from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI(
    title="Data.Insights Multi-Timeframe Quantitative Engine", version="15.0"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Data.Insights.Engine")


class MarketInput(BaseModel):
  timeframe: str
  ema_curr: float
  ema_prev: float
  spread: Optional[float] = 25.0


class TradeSetupResponse(BaseModel):
  timeframe: str
  prediction: int
  signal: str
  probability_score: float
  entry: float
  sl: float
  tp: float


class PredictionOutput(BaseModel):
  total_evaluated: int
  selected_count: int
  top_trades: List[TradeSetupResponse]


@app.get("/")
def read_root():
  return {"status": "online", "version": "15.0"}


@app.post("/api/predict_top_setups", response_model=PredictionOutput)
def predict_top_setups(payload: List[MarketInput]):
  try:
    if not payload:
      raise HTTPException(
          status_code=status.HTTP_400_BAD_REQUEST, detail="Payload is empty."
      )

    top_trades = []
    tf_multipliers = {"M1": 1.0, "M5": 1.2, "M15": 1.5, "H1": 2.0, "H4": 2.5}

    for item in payload:
      tf = item.timeframe
      ema_curr = item.ema_curr
      ema_prev = item.ema_prev
      current_spread = item.spread or 25.0
      mult = tf_multipliers.get(tf, 1.0)

      # Calculate exact candle-over-candle slope
      slope = (ema_curr - ema_prev) * 100.0 * mult

      if slope >= 0:
        signal = "BUY"
        prediction = 1
        probability_score = round(
            float(np.clip(52.0 + (abs(slope) * 15.0), 52.0, 92.0)), 2
        )
      else:
        signal = "SELL"
        prediction = 0
        probability_score = round(
            float(np.clip(52.0 + (abs(slope) * 15.0), 52.0, 92.0)), 2
        )

      entry = round(ema_curr, 2)
      spread_buffer = max(2.00, current_spread * 0.05)
      risk_dist = 4.00 + spread_buffer
      reward_dist = risk_dist * 2.2

      if signal == "BUY":
        sl = round(entry - risk_dist, 2)
        tp = round(entry + reward_dist, 2)
      else:
        sl = round(entry + risk_dist, 2)
        tp = round(entry - reward_dist, 2)

      top_trades.append(
          TradeSetupResponse(
              timeframe=tf,
              prediction=prediction,
              signal=signal,
              probability_score=probability_score,
              entry=entry,
              sl=sl,
              tp=tp,
          )
      )

    return PredictionOutput(
        total_evaluated=len(payload),
        selected_count=len(top_trades),
        top_trades=top_trades,
    )
  except Exception as e:
    logger.error(f"[Execution Error]: {str(e)}")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
    )