import glob
import os
import numpy as np
import pandas as pd


class XAUUSDAnalysisEngine:

  def __init__(self, recent_csv_path="xauusd_m1_recent_1month.csv"):
    self.recent_csv_path = recent_csv_path
    self.df = self._load_data()

  def _load_data(self):
    try:
      part_files = sorted(glob.glob("xauusd_m1_recent_1month_part*.csv"))
      if part_files:
        df = pd.concat(
            [pd.read_csv(f, sep=None, engine="python") for f in part_files],
            ignore_index=True,
        )
      elif os.path.exists(self.recent_csv_path):
        df = pd.read_csv(self.recent_csv_path, sep=None, engine="python")
      else:
        return pd.DataFrame()

      df.columns = [
          str(c).lower().replace('"', "").replace("'", "").strip("<> ")
          for c in df.columns
      ]
      if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
      return df
    except Exception as e:
      return pd.DataFrame()

  def run_volume_analysis(self):
    if self.df.empty or "tick_volume" not in self.df.columns:
      return {"error": "Volume data unavailable"}
    vol = self.df["tick_volume"]
    return {
        "mean_volume": float(vol.mean()),
        "p95_volume": float(vol.quantile(0.95)),
    }

  def run_spread_analysis(self):
    if self.df.empty or "spread" not in self.df.columns:
      return {"error": "Spread data unavailable"}
    spreads = self.df["spread"]
    return {"mean": float(spreads.mean()), "p95": float(spreads.quantile(0.95))}

  def run_volatility_analysis(self):
    if (
        self.df.empty
        or "high" not in self.df.columns
        or "low" not in self.df.columns
    ):
      return {"error": "Price data unavailable"}
    ranges = self.df["high"] - self.df["low"]
    atr_14 = ranges.rolling(window=14).mean().dropna()
    p33, p66 = float(atr_14.quantile(0.33)), float(atr_14.quantile(0.66))
    current_atr = float(atr_14.iloc[-1]) if not atr_14.empty else 2.0
    regime = (
        "HIGH"
        if current_atr > p66
        else ("LOW" if current_atr < p33 else "MEDIUM")
    )
    return {"current_atr": current_atr, "current_regime": regime}

  def check_feature_drift(self, current_val: float, baseline_mean: float):
    diff = abs(current_val - baseline_mean)
    return {
        "drift_detected": bool(
            diff > (0.5 * abs(baseline_mean)) if baseline_mean != 0 else False
        )
    }

  def get_full_diagnostic(self):
    return {
        "volume_analysis": self.run_volume_analysis(),
        "spread_analysis": self.run_spread_analysis(),
        "volatility_analysis": self.run_volatility_analysis(),
    }