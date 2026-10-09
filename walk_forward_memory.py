import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, Any, Tuple


class WalkForwardMemoryEngine:
    """
    Core Date & Hourly Walk-Forward Memory Engine for XAUUSD.
    Manages 2023-2026 M15 data, dynamic candle resampling (30M, 1H, 4H),
    date-and-hour specific excursion extraction (MFE/MAE), and drift scaling.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df_history: pd.DataFrame = None
        self.load_dataset()

    def load_dataset(self) -> None:
        """Loads historical dataset and pre-computes temporal features."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at path: {self.data_path}")

        df = pd.read_csv(self.data_path)
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values('time', inplace=True)

        # Extract fast-lookup time features
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['day'] = df['time'].dt.day
        df['hour'] = df['time'].dt.hour
        df['dayofweek'] = df['time'].dt.dayofweek

        self.df_history = df
        print(f"[WalkForwardMemoryEngine] Ingested {len(self.df_history)} M15 records into memory.")

    def get_date_hour_memory(self, current_time: datetime) -> Tuple[float, float, float]:
        """
        Filters memory for exact date & hour across historical years (2023 -> 2024 -> 2025).
        Returns:
            (learned_mfe, learned_mae, drift_factor)
        """
        curr_year = current_time.year
        curr_month = current_time.month
        curr_day = current_time.day
        curr_hour = current_time.hour

        # Direct exact date + hour filter for prior years
        memory_subset = self.df_history[
            (self.df_history['month'] == curr_month) &
            (self.df_history['day'] == curr_day) &
            (self.df_history['hour'] == curr_hour) &
            (self.df_history['year'] < curr_year)
        ]

        # Fallback window: +/- 1 calendar day if exact date fell on a weekend/holiday
        if memory_subset.empty:
            memory_subset = self.df_history[
                (self.df_history['month'] == curr_month) &
                (self.df_history['day'].isin([curr_day - 1, curr_day, curr_day + 1])) &
                (self.df_history['hour'] == curr_hour) &
                (self.df_history['year'] < curr_year)
            ]

        # General hourly fallback for the month if date is still empty
        if memory_subset.empty:
            memory_subset = self.df_history[
                (self.df_history['month'] == curr_month) &
                (self.df_history['hour'] == curr_hour) &
                (self.df_history['year'] < curr_year)
            ]

        if memory_subset.empty:
            return 0.0, 0.0, 1.0

        # Extract per-year historical excursions
        annual_metrics = []
        for year, group in memory_subset.groupby('year'):
            if group.empty:
                continue
            open_p = group['open'].iloc[0]
            high_p = group['high'].max()
            low_p = group['low'].min()

            mfe = high_p - open_p
            mae = open_p - low_p
            annual_metrics.append({'mfe': mfe, 'mae': mae})

        if not annual_metrics:
            return 0.0, 0.0, 1.0

        df_metrics = pd.DataFrame(annual_metrics)
        learned_mfe = float(df_metrics['mfe'].mean())
        learned_mae = float(df_metrics['mae'].mean())

        # Measure current live volatility relative to expected historical hourly range
        recent_bars = self.df_history[self.df_history['time'] <= current_time].tail(4)  # Last 1 hour of M15
        if not recent_bars.empty:
            actual_hourly_range = recent_bars['high'].max() - recent_bars['low'].min()
            expected_range = learned_mfe + learned_mae
            if expected_range > 0:
                # Dynamic scaling factor bound between 0.7 (consolidation squeeze) and 1.5 (expansion)
                drift_factor = max(0.7, min(1.5, actual_hourly_range / expected_range))
            else:
                drift_factor = 1.0
        else:
            drift_factor = 1.0

        return learned_mfe, learned_mae, drift_factor

    def get_mtf_structure(self, current_time: datetime) -> Dict[str, Any]:
        """
        Resamples recent historical M15 data on the fly into 1H and 4H context
        to evaluate key support/resistance boundaries and target expansion space.
        """
        history_window = self.df_history[self.df_history['time'] <= current_time].tail(96)  # Last 24 hours
        if len(history_window) < 16:
            return {"macro_bias": "NEUTRAL", "h4_high": 0.0, "h4_low": 0.0}

        # Resample to 4H boundaries
        h4_high = float(history_window['high'].max())
        h4_low = float(history_window['low'].min())

        # Resample to 1H structure
        h1_recent = history_window.tail(4)
        h1_open = float(h1_recent['open'].iloc[0])
        h1_close = float(h1_recent['close'].iloc[-1])

        macro_bias = "BUY" if h1_close >= h1_open else "SELL"

        return {
            "macro_bias": macro_bias,
            "h4_high": round(h4_high, 2),
            "h4_low": round(h4_low, 2)
        }