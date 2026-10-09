import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://wlbdtvqdfgpamodcrqno.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "xauusd_m15_2023_2026.csv")

class AutoNewsImpactUpdater:
    def __init__(self):
        self.supabase: Client = None
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            except Exception as e:
                print(f"[NewsUpdater] Supabase connection error: {e}")

    def run_update_pipeline(self):
        if not os.path.exists(DATASET_PATH):
            print(f"[NewsUpdater] Dataset not found at {DATASET_PATH}")
            return

        df = pd.read_csv(DATASET_PATH)
        df['bar_range'] = df['high'] - df['low']
        mean_range = df['bar_range'].mean()

        high_vol_bars = df[df['bar_range'] > (mean_range + 3.5 * df['bar_range'].std())]
        events = ['CPI', 'NFP', 'FOMC', 'PPI', 'DEFAULT']
        
        for event in events:
            sample_bars = high_vol_bars.tail(24) if event != 'DEFAULT' else high_vol_bars.tail(50)
            expansions = []
            reversals = 0
            bullish_closes = 0

            for idx in sample_bars.index:
                if idx < 16 or (idx + 4) >= len(df):
                    continue

                post_1h = df.iloc[idx : idx + 4]
                pre_close = df.iloc[idx - 1]['close']

                range_usd = post_1h['high'].max() - post_1h['low'].min()
                expansions.append(range_usd)

                initial_move = post_1h['close'].iloc[0] - pre_close
                final_move = post_1h['close'].iloc[-1] - pre_close

                if (initial_move > 0 and final_move < 0) or (initial_move < 0 and final_move > 0):
                    reversals += 1

                if final_move > 0:
                    bullish_closes += 1

            count = max(len(expansions), 1)
            avg_expansion = float(np.mean(expansions)) if expansions else 12.0
            vol_mult = round(avg_expansion / max(mean_range, 0.01), 2)
            rev_rate = round(reversals / count, 2)
            dir_bias = round(bullish_closes / count, 2)
            drift_mult = round(min(max(vol_mult / 1.5, 1.0), 3.0), 2)

            data = {
                "event_type": event,
                "avg_1h_range_usd": round(avg_expansion, 2),
                "volatility_mult": min(vol_mult, 4.0),
                "reversal_rate": rev_rate,
                "directional_bias": dir_bias,
                "drift_multiplier": drift_mult,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

            if self.supabase:
                try:
                    self.supabase.table("historical_news_impact").upsert(data, on_conflict="event_type").execute()
                    print(f"[NewsUpdater] Metrics updated for {event}.")
                except Exception as e:
                    print(f"[NewsUpdater] Upsert failed for {event}: {e}")

if __name__ == "__main__":
    updater = AutoNewsImpactUpdater()
    updater.run_update_pipeline()