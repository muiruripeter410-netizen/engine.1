from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from analysis_engine import XAUUSDAnalysisEngine

app = FastAPI(title="Data.Insights XAUUSD Institutional Cloud Engine", version="3.60")

# Initialize the Quantitative Analysis Engine
analyzer = XAUUSDAnalysisEngine("xauusd_m1_recent_1month.csv")

class BarPayload(BaseModel):
    time: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    ema200: float

def compute_trade_levels(tos: float, atr: float, spread: float):
    s = tos / 100.0
    sl_multiplier = 3.0 - (1.2 * s)
    tp_multiplier = 2.0 + (2.5 * s)
    
    raw_sl = spread + (sl_multiplier * max(atr, 1.50))
    raw_tp = spread + (tp_multiplier * max(atr, 1.50))
    
    # Institutional minimum safety floor for Gold ($3.50 SL, $5.00 TP)
    sl = max(3.50, round(raw_sl, 2))
    tp = max(5.00, round(raw_tp, 2))
    
    return sl, tp

@app.get("/")
def read_root():
    return {
        "status": "ONLINE", 
        "system": "Data.Insights XAUUSD Institutional Engine",
        "active_features": ["Dynamic Volatility Filtering", "Spread Cost Impact Adjustment", "Drift Monitoring"]
    }

@app.get("/api/analysis/full")
def get_full_system_analysis():
    """Returns complete diagnostic suite including drift status."""
    return analyzer.get_full_diagnostic()

@app.post("/api/analyze")
def analyze_bar(payload: BarPayload):
    try:
        # Pull live metrics from analysis engine
        vol_data = analyzer.run_volatility_analysis()
        spread_data = analyzer.run_spread_analysis()
        
        current_atr = vol_data.get("current_atr", 2.0)
        mean_spread = spread_data.get("mean", 0.30)
        regime = vol_data.get("current_regime", "NORMAL")
        
        # --- NEW: Active Feature & Drift Integration ---
        # Check if current bar volume drifts significantly from baseline
        baseline_vol_mean = 100.0 # Can be mapped from historical stats
        drift_check = analyzer.check_feature_drift(float(payload.volume), baseline_vol_mean)
        
        # Core Model / TOS Evaluation Loop incorporating market regime & drift penalty
        trend_score = 1.0 if payload.close > payload.ema200 else 0.0
        base_prob = 0.78 if trend_score == 1.0 else 0.30
        
        # Penalty if feature drift or extreme high volatility regime is detected
        if drift_check["drift_detected"] or regime == "HIGH":
            base_prob *= 0.85 # Dampen confidence under unstable conditions
            
        tos = round(base_prob * 100, 2)
        
        action = "HOLD"
        if base_prob > 0.65 and regime != "HIGH":
            action = "BUY"
        elif base_prob < 0.40 and regime != "HIGH":
            action = "SELL"
            
        dynamic_sl, dynamic_tp = compute_trade_levels(tos, current_atr, mean_spread)
        
        return {
            "action": action,
            "ai_prob": base_prob,
            "tos": tos,
            "dynamic_sl": dynamic_sl,
            "dynamic_tp": dynamic_tp,
            "regime": regime,
            "drift_detected": drift_check["drift_detected"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retrain")
def retrain_models():
    global analyzer
    analyzer = XAUUSDAnalysisEngine("xauusd_m1_recent_1month.csv")
    return {"status": "COMPLETE", "message": "Engine re-initialized and features re-calibrated."}