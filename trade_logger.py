import os
from datetime import datetime, timezone
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://wlbdtvqdfgpamodcrqno.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

class TradeOutcomeLogger:
    def __init__(self):
        self.supabase: Client = None
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            except Exception as e:
                print(f"[TradeLogger] Connection error: {e}")

    def log_trade_entry(self, payload: dict, action: str, metrics: dict, news_features: dict = None):
        if not self.supabase:
            return None
        
        news_features = news_features or {}
        try:
            data = {
                "timestamp": payload.get("time"),
                "symbol": "XAUUSDm",
                "action": action,
                "close_price": payload.get("close"),
                "tp1_offset": metrics.get("tp1_offset"),
                "tp2_offset": metrics.get("tp2_offset"),
                "sl_offset": metrics.get("sl_offset"),
                "drift_factor": metrics.get("drift_factor"),
                "rr_ratio": metrics.get("rr_ratio"),
                "minutes_to_news": news_features.get("minutes_to_news", 9999),
                "news_impact_rating": news_features.get("news_impact_rating", 0),
                "hist_volatility_mult": news_features.get("hist_volatility_mult", 1.0),
                "hist_reversal_prob": news_features.get("hist_reversal_prob", 0.0),
                "hist_directional_bias": news_features.get("hist_directional_bias", 0.5),
                "hist_mean_expansion_usd": news_features.get("hist_mean_expansion_usd", 0.0),
                "status": "OPEN"
            }
            return self.supabase.table("execution_logs").insert(data).execute()
        except Exception as e:
            print(f"[TradeLogger] Logging failed: {e}")
            return None

    def log_trade_close(self, ticket: int, profit: float, exit_reason: str, exit_price: float):
        if not self.supabase:
            return None
        try:
            data = {
                "ticket": ticket,
                "profit": profit,
                "exit_reason": exit_reason,
                "exit_price": exit_price,
                "status": "CLOSED",
                "closed_at": datetime.now(timezone.utc).isoformat()
            }
            return self.supabase.table("execution_logs").update(data).eq("ticket", ticket).execute()
        except Exception as e:
            print(f"[TradeLogger] Close log failed: {e}")
            return None