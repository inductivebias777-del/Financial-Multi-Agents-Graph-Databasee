# train_predictor.py

import pandas as pd
import numpy as np
import joblib
import os
import lightgbm as lgb
from tqdm import tqdm
os.chdir('/home/neo/Downloads/CODE_Other_Models/Financial_ADK/financial_rag_project')
# --- Configuration ---
PRICES_DIR = "./data/structured/prices"
MODEL_DIR = "./app/models/saved_models" # Matching your project structure
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Feature Engineering Parameters ---
WINDOW_SIZE = 10         # How many past days of data to use as features
PREDICTION_HORIZON = 1   # How many days into the future to predict (1 = next day)

def create_features(df):
    """Creates time-series features from a stock price DataFrame."""
    # Create a new DataFrame for features to avoid modifying the original
    featured_df = df[['Close', 'Volume']].copy()

    # 1. Lag Features (autoregressive part)
    # Use the previous 'WINDOW_SIZE' days' closing prices as features
    for i in range(1, WINDOW_SIZE + 1):
        featured_df[f'Close_lag_{i}'] = featured_df['Close'].shift(i)

    # 2. Rolling Window Features
    # Create rolling averages to capture recent trends
    featured_df['MA_5'] = featured_df['Close'].rolling(window=5).mean()
    featured_df['MA_20'] = featured_df['Close'].rolling(window=20).mean()

    # 3. Volume-based Features
    featured_df['Volume_lag_1'] = featured_df['Volume'].shift(1)
    featured_df['Volume_MA_5'] = featured_df['Volume'].rolling(window=5).mean()

    # 4. Create the target variable
    # The 'target' is the closing price 'PREDICTION_HORIZON' days in the future
    featured_df['target'] = featured_df['Close'].shift(-PREDICTION_HORIZON)

    # Drop rows with NaN values created by shifts and rolling windows
    featured_df.dropna(inplace=True)

    return featured_df

if __name__ == "__main__":
    price_files = [f for f in os.listdir(PRICES_DIR) if f.endswith('_prices.csv')]

    for file in tqdm(price_files, desc="Training Models for each stock"):
        ticker = file.split('_')[0]

        # Load data
        df = pd.read_csv(os.path.join(PRICES_DIR, file))
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        # Create features
        data = create_features(df)

        if data.empty:
            print(f"Skipping {ticker}: Not enough data to create features.")
            continue

        # Define features (X) and target (y)
        X = data.drop(columns=['target', 'Close', 'Volume'])
        y = data['target']

        # Train the model
        # We train on the entire history to make the best possible prediction for tomorrow
        print(f"\nTraining model for {ticker} with {len(X.columns)} features...")

        model = lgb.LGBMRegressor(
            random_state=42,
            n_estimators=200,      # More estimators for better performance
            learning_rate=0.05,
            num_leaves=31
        )
        model.fit(X, y)

        # Save the trained model and the list of features it expects
        joblib.dump(model, os.path.join(MODEL_DIR, f"{ticker}_price_regressor.joblib"))
        joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, f"{ticker}_features.joblib"))

        print(f"✓ Model for {ticker} saved.")

    print("\nTraining complete! All models saved.")