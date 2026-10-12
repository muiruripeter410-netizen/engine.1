import joblib
import pandas as pd
from analysis_engine import XAUUSDAnalysisEngine
from xgboost import XGBClassifier

# 1. Load data via your analysis engine
print("Loading data...")
engine = XAUUSDAnalysisEngine()
df = engine.df

if df.empty:
  print("Error: Dataset is empty.")
  exit(1)

print("Available columns:", df.columns.tolist())

# Identify the active price column for tick data (usually 'bid' or 'last')
price_col = (
    "bid"
    if "bid" in df.columns
    else ("last" if "last in df.columns" else "close")
)
if price_col not in df.columns:
  price_col = df.columns[
      2
  ]  # Fallback to the third column if names differ

print(f"Using price column '{price_col}' for EMA 10 calculation...")

# 2. Compute EMA 10 indicator and target label
df["ema_10"] = df[price_col].ewm(span=10, adjust=False).mean()
df["return"] = df[price_col].pct_change()
df["target"] = (df["return"].shift(-1) > 0).astype(int)

# 3. Train model using EMA 10
feature_cols = ["ema_10"]
df_clean = df.dropna(subset=feature_cols + ["target"])

X = df_clean[feature_cols]
y = df_clean["target"]

print("Training XGBoost model...")
model = XGBClassifier(n_estimators=50, max_depth=3)
model.fit(X, y)

# 4. Save model artifact
joblib.dump(model, "model.joblib")
print("Success! model.joblib generated.")