import os
import requests
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://wlbdtvqdfgpamodcrqno.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

class NewsEngine:
    def __init__(self):
        self.supabase: Client = None
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            except Exception:
                pass
        
        self.impact_cache = {}
        self.last_cache_update = None
        self.cached_events = []
        self.last_event_fetch = None

    def _refresh_db_cache(self):
        now = datetime.now(timezone.utc)
        if self.last_cache_update and (now - self.last_cache_update) < timedelta(hours=6):
            return

        if self.supabase:
            try:
                res = self.supabase.table("historical_news_impact").select("*").execute()
                if res.data:
                    for row in res.data:
                        self.impact_cache[row["event_type"]] = row
                    self.last_cache_update = now
            except Exception:
                pass

    def get_upcoming_news_status(self) -> dict:
        now = datetime.now(timezone.utc)
        if self.last_event_fetch and (now - self.last_event_fetch) < timedelta(minutes=30):
            return self.cached_events

        url = "https://nodedata.forexfactory.com/ff_calendar_thisweek.json"
        try:
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3)
            if res.status_code == 200:
                self.cached_events = [ev for ev in res.json() if ev.get("country") == "USD" and ev.get("impact") in ["High", "Medium"]]
                self.last_event_fetch = now
        except Exception:
            pass
        return self.cached_events

    def get_news_feature_vector(self, current_time_iso: str) -> dict:
        self._refresh_db_cache()
        events = self.get_upcoming_news_status()
        
        minutes_to_news = 9999
        impact_rating = 0
        event_name = "DEFAULT"

        try:
            curr_dt = datetime.fromisoformat(current_time_iso.replace("Z", "+00:00"))
        except Exception:
            curr_dt = datetime.now(timezone.utc)

        for ev in events:
            try:
                ev_dt = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
                diff = int((ev_dt - curr_dt).total_seconds() / 60)
                if -15 <= diff < minutes_to_news:
                    minutes_to_news = diff
                    event_name = ev.get("title", "DEFAULT")
                    impact_rating = 3 if ev.get("impact") == "High" else 2
            except Exception:
                continue

        event_key = "DEFAULT"
        for key in ["CPI", "NFP", "FOMC", "PPI"]:
            if key in event_name.upper():
                event_key = key
                break

        metrics = self.impact_cache.get(event_key, {
            "avg_1h_range_usd": 12.0,
            "volatility_mult": 1.2,
            "reversal_rate": 0.20,
            "directional_bias": 0.50,
            "drift_multiplier": 1.00
        })

        return {
            "minutes_to_news": minutes_to_news,
            "news_impact_rating": impact_rating,
            "hist_mean_expansion_usd": float(metrics["avg_1h_range_usd"]),
            "hist_volatility_mult": float(metrics["volatility_mult"]),
            "hist_reversal_prob": float(metrics["reversal_rate"]),
            "hist_directional_bias": float(metrics["directional_bias"]),
            "hist_drift_multiplier": float(metrics["drift_multiplier"])
        }