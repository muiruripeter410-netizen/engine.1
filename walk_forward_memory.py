import os
import pandas as pd

class WalkForwardMemoryEngine:
    def __init__(self, dataset_path=None, *args, **kwargs):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # If dataset_path is a DataFrame or invalid type, fallback to string path
        if isinstance(dataset_path, str) and dataset_path:
            self.dataset_path = dataset_path
        else:
            self.dataset_path = os.path.join(BASE_DIR, "xauusd_m15_2023_2026.csv")
        
        self.metrics = {
            "tp1_offset": 5.0,
            "tp2_offset": 10.0,
            "sl_offset": 4.0,
            "drift_factor": 1.0,
            "rr_ratio": 2.0
        }
        self._load_memory()

    def _load_memory(self):
        """Safely loads recent baseline targets from dataset."""
        if not os.path.exists(self.dataset_path):
            return

        try:
            df = pd.read_csv(self.dataset_path)
            if 'high' in df.columns and 'low' in df.columns:
                recent_range = (df['high'] - df['low']).tail(100).mean()
                if pd.notna(recent_range) and recent_range > 0:
                    self.metrics["tp1_offset"] = round(recent_range * 0.8, 2)
                    self.metrics["tp2_offset"] = round(recent_range * 1.6, 2)
                    self.metrics["sl_offset"] = round(recent_range * 0.6, 2)
        except Exception as e:
            print(f"[WalkForwardMemory] Warning loading memory: {e}")

    def get_active_metrics(self) -> dict:
        return self.metrics