import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "xauusd_m15_2023_2026.csv")
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "xgboost_model.pkl")

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data['returns'] = data['close'].pct_change()
    data['high_low_ratio'] = (data['high'] - data['low']) / data['close']
    
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data['rsi'] = 100 - (100 / (1 + rs))
    
    data['ema_20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['ema_dist'] = (data['close'] - data['ema_20']) / data['close']
    
    # Baseline fallback features for news
    data['minutes_to_news'] = 9999
    data['news_impact_rating'] = 0
    data['hist_volatility_mult'] = 1.0
    data['hist_reversal_prob'] = 0.2
    data['hist_directional_bias'] = 0.5
    data['hist_mean_expansion_usd'] = 12.0

    future_close = data['close'].shift(-4)
    data['target'] = np.where(future_close > (data['close'] + 0.50), 1, 0)
    
    return data.dropna()

def train_and_save():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)
    processed_df = prepare_features(df)
    
    feature_cols = [
        'returns', 'high_low_ratio', 'rsi', 'ema_dist',
        'minutes_to_news', 'news_impact_rating', 'hist_volatility_mult',
        'hist_reversal_prob', 'hist_directional_bias', 'hist_mean_expansion_usd'
    ]
    
    X = processed_df[feature_cols]
    y = processed_df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"XGBoost Model Retrained. Accuracy: {acc * 100:.2f}%")
    
    joblib.dump({"model": model, "feature_names": feature_cols}, MODEL_OUTPUT_PATH)
    print(f"Model exported to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    train_and_save()