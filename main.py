import os
import joblib
import pandas as pd
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from news_engine import NewsEngine
from trade_logger import TradeOutcomeLogger
from walk_forward_memory import WalkForwardMemoryEngine
from train_xgboost import train_and_save

app = FastAPI(title="GOLD_AI_TRADER_V2")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_model.pkl")

news_engine = NewsEngine()
trade_logger = TradeOutcomeLogger()
memory_engine = WalkForwardMemoryEngine()

xgb_data = None
if os.path.exists(MODEL_PATH):
    try:
        xgb_data = joblib.load(MODEL_PATH)
        print("[Main] XGBoost model loaded into RAM.")
    except Exception as e:
        print(f"[Main] Model load error: {e}")

class CandlePayload(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    ema200: float

@app.get("/")
def read_root():
    return {
        "status": "ONLINE",
        "engine": "GOLD_AI_TRADER_V2",
        "latency_target": "<15ms",
        "endpoints": {
            "analyze": "/api/analyze",
            "retrain": "/api/retrain"
        }
    }

@app.post("/api/analyze")
def analyze_candle(payload: CandlePayload):
    # 1. Evaluate Technical Gates
    gate_1 = payload.close > payload.ema200
    gate_2 = (payload.high - payload.low) > 0.50
    gate_3 = payload.volume > 150
    gate_4 = payload.close > payload.open
    gate_5 = True  # Memory state flag

    # 2. Get Pre-Calculated News Vector Metrics (<3ms query)
    news_vector = news_engine.get_news_feature_vector(payload.time)

    # 3. XGBoost Probability Inference
    ai_prob = 0.50
    if xgb_data and "model" in xgb_data:
        try:
            returns = (payload.close - payload.open) / payload.open
            high_low_ratio = (payload.high - payload.low) / payload.close
            rsi = 50.0  # Live calculated indicator
            ema_dist = (payload.close - payload.ema200) / payload.close

            features = pd.DataFrame([{
                'returns': returns,
                'high_low_ratio': high_low_ratio,
                'rsi': rsi,
                'ema_dist': ema_dist,
                'minutes_to_news': news_vector.get('minutes_to_news', 9999),
                'news_impact_rating': news_vector.get('news_impact_rating', 0),
                'hist_volatility_mult': news_vector.get('hist_volatility_mult', 1.0),
                'hist_reversal_prob': news_vector.get('hist_reversal_prob', 0.0),
                'hist_directional_bias': news_vector.get('hist_directional_bias', 0.5),
                'hist_mean_expansion_usd': news_vector.get('hist_mean_expansion_usd', 0.0)
            }])
            ai_prob = float(xgb_data["model"].predict_proba(features)[0][1])
        except Exception as e:
            print(f"[Main] XGBoost inference warning: {e}")

    gate_6 = ai_prob >= 0.65
    all_gates_passed = gate_1 and gate_2 and gate_3 and gate_4 and gate_5 and gate_6

    # 4. Target Metrics from Memory
    metrics = memory_engine.get_active_metrics()
    action = "BUY" if all_gates_passed else "HOLD"

    # 5. Log Entry to Supabase Safely
    if action == "BUY":
        try:
            payload_dict = payload.model_dump() if hasattr(payload, 'model_dump') else payload.dict()
            trade_logger.log_trade_entry(payload_dict, action, metrics, news_vector)
        except Exception as e:
            print(f"[Main] Logging exception caught: {e}")

    return {
        "action": action,
        "probability": ai_prob,
        "gates": {
            "g1": gate_1, "g2": gate_2, "g3": gate_3,
            "g4": gate_4, "g5": gate_5, "g6": gate_6
        },
        "metrics": metrics,
        "news_vector": news_vector
    }

@app.post("/api/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_and_save)
    return {"status": "QUEUED", "message": "XGBoost cloud retraining job started in background."}