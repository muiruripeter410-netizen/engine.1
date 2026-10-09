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

def calculate_dynamic_threshold(news_vector: dict, memory_metrics: dict) -> float:
    base_threshold = 0.52
    
    news_adj = 0.0
    mins_to_news = news_vector.get('minutes_to_news', 9999)
    if mins_to_news < 30 and news_vector.get('news_impact_rating', 0) >= 2:
        news_adj = 0.08
        
    drift = memory_metrics.get('hist_drift_multiplier', 1.0)
    drift_adj = -0.02 if drift > 1.1 else 0.0

    calculated_threshold = base_threshold + news_adj + drift_adj
    return max(0.51, min(0.65, calculated_threshold))

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
    # 1. Fetch Context Features (<3ms)
    news_vector = news_engine.get_news_feature_vector(payload.time)
    metrics = memory_engine.get_active_metrics()

    # 2. XGBoost Inference
    ai_prob = 0.50
    if xgb_data and "model" in xgb_data:
        try:
            returns = (payload.close - payload.open) / payload.open
            high_low_ratio = (payload.high - payload.low) / payload.close
            rsi = 50.0
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

    # 3. Dynamic Threshold Calculation
    dynamic_threshold = calculate_dynamic_threshold(news_vector, metrics)

    # 4. Filter Gates
    # Gate 1 (EMA) is intentionally disabled to allow AI-driven reversal entries
    g1_passed = True
    volatility_passed = (payload.high - payload.low) > 0.30
    volume_passed = payload.volume > 100
    memory_passed = True

    # BUY Gates
    buy_g4 = payload.close > payload.open
    buy_g6 = ai_prob >= dynamic_threshold
    buy_passed = g1_passed and volatility_passed and volume_passed and buy_g4 and memory_passed and buy_g6

    # SELL Gates
    sell_g4 = payload.close < payload.open
    sell_g6 = (1.0 - ai_prob) >= dynamic_threshold
    sell_passed = g1_passed and volatility_passed and volume_passed and sell_g4 and memory_passed and sell_g6

    # 5. Final Decision
    if buy_passed:
        action = "BUY"
    elif sell_passed:
        action = "SELL"
    else:
        action = "HOLD"

    # 6. Log Entries
    if action in ["BUY", "SELL"]:
        try:
            payload_dict = payload.model_dump() if hasattr(payload, 'model_dump') else payload.dict()
            trade_logger.log_trade_entry(payload_dict, action, metrics, news_vector)
        except Exception as e:
            print(f"[Main] Logging exception caught: {e}")

    return {
        "action": action,
        "probability": ai_prob,
        "active_threshold": dynamic_threshold,
        "buy_gates": {
            "g1": g1_passed, "g2": volatility_passed, "g3": volume_passed,
            "g4": buy_g4, "g5": memory_passed, "g6": buy_g6
        },
        "sell_gates": {
            "g1": g1_passed, "g2": volatility_passed, "g3": volume_passed,
            "g4": sell_g4, "g5": memory_passed, "g6": sell_g6
        },
        "metrics": metrics,
        "news_vector": news_vector
    }

@app.post("/api/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_and_save)
    return {"status": "QUEUED", "message": "XGBoost cloud retraining job started in background."}