import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from stockstats import wrap
import json
import joblib
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load data
DATA_PATH = os.path.join(PROJECT_ROOT, "training_data", "XAUUSDm_m5.csv")
df = pd.read_csv(DATA_PATH)

# Rename tick_volume to volume for stockstats
df.rename(columns={'tick_volume': 'volume'}, inplace=True)

# Feature Engineering using stockstats
stock = wrap(df)

# 1. Basic Indicators (Calculated inside stock object)
_ = stock['rsi_14']
_ = stock['boll']
_ = stock['boll_ub']
_ = stock['boll_lb']
_ = stock['macd']
_ = stock['macds']
_ = stock['macdh']
_ = stock['atr']
_ = stock['cci']
_ = stock['adx']

# Convert stockstats object back to standard DataFrame to ensure we have all columns
df = pd.DataFrame(stock)

# 2. Advanced Features (Slopes & Trends) - Calculate on the DataFrame
# RSI Slope (Change over last 3 periods)
df['rsi_slope'] = df['rsi_14'] - df['rsi_14'].shift(3)

# MACD Slope
df['macd_slope'] = df['macd'] - df['macd'].shift(3)

# Bollinger Band Width (Volatility)
df['bb_width'] = (df['boll_ub'] - df['boll_lb']) / df['boll']

# Distance from MA
df['dist_ma'] = (df['close'] - df['boll']) / df['boll']

# Volume Trend
df['vol_trend'] = df['volume'] / df['volume'].rolling(20).mean()

# 3. Target Definition
# Predict if next candle closes HIGHER than current close + spread buffer
SPREAD_BUFFER = 0.0005 # 0.05% move (~10-20 points on Gold)
df['target'] = (df['close'].shift(-1) > df['close'] * (1 + SPREAD_BUFFER)).astype(int)

# Drop NaN
df_clean = df.dropna()

# Select Features
features = [
    'rsi_14', 'rsi_slope',
    'boll', 'boll_ub', 'boll_lb', 'bb_width', 'dist_ma',
    'macd', 'macds', 'macdh', 'macd_slope',
    'atr', 'cci', 'adx',
    'close', 'volume', 'vol_trend'
]

X = df_clean[features]
y = df_clean['target']

# Split data (Time Series split - no shuffle)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Training Data: {X_train.shape}")
print(f"Testing Data: {X_test.shape}")

# Hyperparameter Tuning (RandomizedSearch)
param_dist = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 300, 500, 1000],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5, 1]
}

clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)

print("Tuning Hyperparameters...")
random_search = RandomizedSearchCV(
    clf, 
    param_distributions=param_dist, 
    n_iter=20, 
    scoring='accuracy', 
    cv=3, 
    verbose=1, 
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

print(f"Best Parameters: {random_search.best_params_}")
best_model = random_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Model
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_scalper.json")
best_model.save_model(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
