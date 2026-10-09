import os
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from supabase import create_client, Client

app = FastAPI(title="XAUUSD Dynamic Walk-Forward Engine v1.0")

# ----------------------------------------------------------------------
# ENVIRONMENT & SUPABASE INITIALIZATION
# ----------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client initialized successfully.")
    except Exception as e:
        print(f"Supabase initialization error: {e}")

# Path to the primary M15 historical dataset inside engine.1
DATA_PATH = os.path.join(os.path.dirname(__file__), "xauusd_m15_2023_2026.csv")
df_history: Optional[pd.DataFrame] = None


# ----------------------------------------------------------------------
# DATA PREPARATION & CANDLE RESAMPLING
# ----------------------------------------------------------------------
def load_and_prep_history():
    global df_history
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values('time', inplace=True)
        
        # Extract date-time features for fast memory lookup
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['day'] = df['time'].dt.day
        df['hour'] = df['time'].dt.hour
        
        df_history = df
        print(f"Loaded {len(df_history)} historical M15 bars into Walk-Forward memory.")
    else:
        print(f"Warning: Dataset file not found at {DATA_PATH}")

@app.on_event("startup")
def startup_event():
    load_and_prep_history()


def get_mtf_context(df: pd.DataFrame, current_time: datetime):
    """
    Extracts 4H and 1H structural boundaries to identify liquidity targets
    and daily expansion space, without forcing strict direction alignment across all timeframes.
    """
    recent_df = df[df['time'] <= current_time].tail(96)  # Last 24 hours of M15 data
    if len(recent_df) < 16:
        return {"bias": "NEUTRAL", "target_high": 0.0, "target_low": 0.0}

    # 4H macro boundary
    h4_high = recent_df['high'].max()
    h4_low = recent_df['low'].min()
    
    # 1H liquidity structure
    h1_recent = recent_df.tail(4)
    h1_close = h1_recent['close'].iloc[-1]
    h1_open = h1_recent['open'].iloc[0]
    
    # Directional target bias based on structural expansion
    bias = "BUY" if h1_close >= h1_open else "SELL"
    
    return {
        "bias": bias,
        "target_high": round(float(h4_high), 2),
        "target_low": round(float(h4_low), 2)
    }


# ----------------------------------------------------------------------
# PYDANTIC INCOMING PAYLOAD MODEL
# ----------------------------------------------------------------------
class CandlePayload(BaseModel):
    symbol: str
    time: str  # ISO 8601 string
    open: float
    high: float
    low: float
    close: float
    volume: int
    active_trade: bool = False
    trade_type: Optional[str] = None  # "BUY" or "SELL"
    entry_price: Optional[float] = None
    tp1_hit: Optional[bool] = False


# ----------------------------------------------------------------------
# MAIN API ENDPOINT
# ----------------------------------------------------------------------
@app.post("/api/analyze")
async def analyze_candle(payload: CandlePayload):
    if df_history is None:
        return {"action": "HOLD", "reason": "Historical dataset memory not loaded."}

    try:
        current_time = datetime.fromisoformat(payload.time.replace("Z", "+00:00"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ISO timestamp: {str(e)}")

    curr_year = current_time.year
    curr_month = current_time.month
    curr_day = current_time.day
    curr_hour = current_time.hour

    # ------------------------------------------------------------------
    # IN-TRADE RE-EVALUATION (For active trades in drawdown or loss)
    # ------------------------------------------------------------------
    if payload.active_trade and payload.entry_price is not None:
        # Check current PnL direction relative to 1-hour holding window
        if payload.trade_type == "BUY":
            in_loss = payload.close < payload.entry_price
        else:
            in_loss = payload.close > payload.entry_price

        if in_loss:
            # Query hourly memory for recovery probability
            hourly_recovery_data = df_history[
                (df_history['month'] == curr_month) &
                (df_history['day'] == curr_day) &
                (df_history['hour'] == curr_hour) &
                (df_history['year'] < curr_year)
            ]
            
            if not hourly_recovery_data.empty:
                avg_recovery_range = (hourly_recovery_data['high'] - hourly_recovery_data['low']).mean()
                current_adverse_dist = abs(payload.close - payload.entry_price)
                
                # If adverse drift exceeds expected hourly volatility, trigger early structural exit
                if current_adverse_dist > (avg_recovery_range * 1.2):
                    return {
                        "action": "CLOSE_TRADE",
                        "reason": f"Active trade loss (${current_adverse_dist:.2f}) exceeds hourly historical recovery limit (${avg_recovery_range:.2f})."
                    }
            return {"action": "HOLD_TRADE", "reason": "Trade in drawdown but within historical hourly recovery tolerance."}

    # ------------------------------------------------------------------
    # WALK-FORWARD DATE & HOURLY MEMORY LOOKUP (2023 -> 2024 -> 2025 -> 2026)
    # ------------------------------------------------------------------
    # Filter memory strictly for historical years prior to current year for same date and hour
    date_hour_memory = df_history[
        (df_history['month'] == curr_month) &
        (df_history['day'] == curr_day) &
        (df_history['hour'] == curr_hour) &
        (df_history['year'] < curr_year)
    ]

    # Fallback to date window (±1 day) if specific hour is empty due to market hours/weekends
    if date_hour_memory.empty:
        date_hour_memory = df_history[
            (df_history['month'] == curr_month) &
            (df_history['day'].isin([curr_day - 1, curr_day, curr_day + 1])) &
            (df_history['year'] < curr_year)
        ]

    if date_hour_memory.empty:
        return {"action": "HOLD", "reason": "Insufficient historical date/hour memory found."}

    # Calculate MFE & MAE dynamically from historical date excursions
    annual_excursions = []
    for yr, group in date_hour_memory.groupby('year'):
        day_open = group['open'].iloc[0]
        day_high = group['high'].max()
        day_low = group['low'].min()

        mfe = day_high - day_open
        mae = day_open - day_low
        annual_excursions.append({'year': yr, 'mfe': mfe, 'mae': mae})

    df_exc = pd.DataFrame(annual_excursions)
    learned_mfe = float(df_exc['mfe'].mean())
    learned_mae = float(df_exc['mae'].mean())

    # ------------------------------------------------------------------
    # FORECAST VS. ACTUAL DRIFT ADJUSTMENT
    # ------------------------------------------------------------------
    actual_hour_range = payload.high - payload.low
    forecast_hour_range = learned_mfe + learned_mae
    
    # Calculate drift multiplier to scale targets dynamically
    drift_factor = 1.0
    if forecast_hour_range > 0:
        drift_factor = max(0.8, min(1.5, actual_hour_range / forecast_hour_range))

    adj_mfe = learned_mfe * drift_factor
    adj_mae = learned_mae * drift_factor

    # Get Macro Boundaries
    mtf = get_mtf_context(df_history, current_time)
    bias = mtf['bias']

    # ------------------------------------------------------------------
    # THE 5-GATE VALIDATION ENGINE
    # ------------------------------------------------------------------

    # Gate 1: MTF Target Alignment Check
    target_distance = abs(mtf['target_high'] - payload.close) if bias == "BUY" else abs(payload.close - mtf['target_low'])
    if target_distance < 0.50:
        return {"action": "HOLD", "reason": f"Gate 1 Failed: Price too close to 4H macro boundary (${target_distance:.2f})."}

    # Gate 2: Directional Fit & Velocity Verification
    expected_expansion = adj_mfe if bias == "BUY" else adj_mae
    expected_drawdown = adj_mae if bias == "BUY" else adj_mfe

    # Gate 3: Tradable Range Sufficiency
    if expected_expansion < 1.20 or (expected_expansion / (expected_drawdown + 1e-5)) < 1.1:
        return {
            "action": "HOLD",
            "reason": f"Gate 3 Failed: Tradable range insufficient for {curr_month}/{curr_day} {curr_hour}:00 (Exp: ${expected_expansion:.2f}, Drawdown: ${expected_drawdown:.2f})."
        }

    # Gate 4: Dynamic SL / TP1 / TP2 Calculation
    # TP1 acts as conservative profit-lock boundary (50% MFE)
    # TP2 acts as full expansion target (100% MFE)
    tp1_offset = round(expected_expansion * 0.50, 2)
    tp2_offset = round(expected_expansion * 1.00, 2)
    sl_offset = round(expected_drawdown * 0.85, 2)

    # Gate 5: Reward-to-Risk Validation (Using TP1 as the minimum baseline)
    rr_ratio = tp1_offset / (sl_offset + 1e-5)
    if rr_ratio < 1.1:
        return {
            "action": "HOLD",
            "reason": f"Gate 5 Failed: Baseline R:R ratio insufficient ({rr_ratio:.2f} < 1.1)."
        }

    # Build Approved Decision Payload
    response_payload = {
        "action": bias,
        "symbol": payload.symbol,
        "tp1_offset": tp1_offset,
        "tp2_offset": tp2_offset,
        "sl_offset": sl_offset,
        "learned_mfe": round(adj_mfe, 2),
        "learned_mae": round(adj_mae, 2),
        "drift_factor": round(drift_factor, 2),
        "rr_ratio": round(rr_ratio, 2),
        "reason": f"All 5 Gates Passed for memory window {curr_month}/{curr_day} {curr_hour}:00."
    }

    # ------------------------------------------------------------------
    # SUPABASE CLOUD AUDIT LOGGING
    # ------------------------------------------------------------------
    if supabase:
        try:
            log_entry = {
                "timestamp": payload.time,
                "symbol": payload.symbol,
                "action": bias,
                "close_price": payload.close,
                "tp1_offset": tp1_offset,
                "tp2_offset": tp2_offset,
                "sl_offset": sl_offset,
                "drift_factor": round(drift_factor, 2),
                "rr_ratio": round(rr_ratio, 2),
                "created_at": datetime.utcnow().isoformat()
            }
            supabase.table("execution_logs").insert(log_entry).execute()
        except Exception as log_err:
            print(f"Supabase logging error: {log_err}")

    return response_payload