import os
from supabase import create_client, Client

class TradeOutcomeLogger:
    def __init__(self):
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
        self.supabase: Client = None
        if url and key:
            try:
                self.supabase = create_client(url, key)
            except Exception as e:
                print(f"[TradeLogger] Supabase client init error: {e}")

    def log_trade_entry(self, candle_data: dict, action: str, metrics: dict, news_vector: dict):
        """Logs trade execution entry safely to Supabase without crashing execution on error."""
        if not self.supabase:
            print("[TradeLogger] Supabase client not configured. Skipping remote log.")
            return

        try:
            log_payload = {
                "ticket": candle_data.get("ticket", 0),
                "timestamp": candle_data.get("time"),
                "symbol": candle_data.get("symbol", "XAUUSDm"),
                "action": action,
                "close_price": candle_data.get("close"),
                "tp1_offset": metrics.get("tp1_offset"),
                "tp2_offset": metrics.get("tp2_offset"),
                "sl_offset": metrics.get("sl_offset"),
                "drift_factor": metrics.get("drift_factor"),
                "rr_ratio": metrics.get("rr_ratio"),
                "minutes_to_news": news_vector.get("minutes_to_news"),
                "news_impact_rating": news_vector.get("news_impact_rating"),
                "hist_volatility_mult": news_vector.get("hist_volatility_mult"),
                "hist_reversal_prob": news_vector.get("hist_reversal_prob"),
                "hist_directional_bias": news_vector.get("hist_directional_bias"),
                "hist_mean_expansion_usd": news_vector.get("hist_mean_expansion_usd"),
                "status": "OPEN"
            }
            self.supabase.table("execution_logs").insert(log_payload).execute()
            print("[TradeLogger] Trade logged successfully to Supabase.")
        except Exception as e:
            print(f"[TradeLogger] Failed to log trade entry: {e}")