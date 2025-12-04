"""
LightGBM Training Script v2 - Improved with:
1. Class balancing (SMOTE + class_weight)
2. More technical features
3. Time-based features
4. Better hyperparameter search
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from stockstats import wrap
import json
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load data
DATA_PATH = os.path.join(PROJECT_ROOT, "training_data", "XAUUSDm_m5.csv")
df = pd.read_csv(DATA_PATH)

print(f"Loaded {len(df)} rows of data")

# Rename tick_volume to volume for stockstats
df.rename(columns={'tick_volume': 'volume'}, inplace=True)

# =========================================
# FEATURE ENGINEERING (Enhanced)
# =========================================

stock = wrap(df)

# 1. Basic Indicators
_ = stock['rsi_14']
_ = stock['rsi_6']  # Short-term RSI
_ = stock['boll']
_ = stock['boll_ub']
_ = stock['boll_lb']
_ = stock['macd']
_ = stock['macds']
_ = stock['macdh']
_ = stock['atr']
_ = stock['cci']
_ = stock['adx']
_ = stock['close_5_ema']
_ = stock['close_10_ema']
_ = stock['close_20_ema']
_ = stock['close_50_sma']

# Convert to DataFrame
df = pd.DataFrame(stock)

# 2. Price Action Features
df['rsi_slope'] = df['rsi_14'] - df['rsi_14'].shift(3)
df['macd_slope'] = df['macd'] - df['macd'].shift(3)
df['bb_width'] = (df['boll_ub'] - df['boll_lb']) / df['boll']
df['dist_ma'] = (df['close'] - df['boll']) / df['boll']
df['vol_trend'] = df['volume'] / df['volume'].rolling(20).mean()

# 3. NEW: Momentum Features
df['price_change_1'] = df['close'].pct_change(1)
df['price_change_3'] = df['close'].pct_change(3)
df['price_change_5'] = df['close'].pct_change(5)
df['high_low_range'] = (df['high'] - df['low']) / df['close']
df['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 0.001)
df['close_to_low'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.001)

# 4. NEW: Trend Strength
df['ema_cross'] = (df['close_5_ema'] - df['close_20_ema']) / df['close']
df['trend_strength'] = (df['close'] - df['close_50_sma']) / df['close_50_sma']
df['adx_slope'] = df['adx'] - df['adx'].shift(3)

# 5. NEW: Volatility Features
df['atr_ratio'] = df['atr'] / df['close']
df['vol_spike'] = df['volume'] / df['volume'].shift(1)
df['range_expansion'] = df['high_low_range'] / df['high_low_range'].rolling(10).mean()

# 6. NEW: RSI Divergence Proxy
df['rsi_price_div'] = df['rsi_14'].diff(5) - (df['close'].pct_change(5) * 100)

# 7. NEW: Time-based Features (if time column exists)
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    # Session indicators (London = 8-16 UTC, NY = 13-21 UTC)
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
    df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
    df['overlap_session'] = ((df['hour'] >= 13) & (df['hour'] <= 16)).astype(int)

# =========================================
# TARGET DEFINITION (Significant moves only)
# =========================================

# Predict if price will rise significantly in next 3 candles
LOOK_AHEAD = 3  # 3 candles = 15 minutes on M5
THRESHOLD = 0.001  # 0.1% move - significant for Gold

# Calculate max high in next 3 candles
df['future_max'] = df['high'].shift(-1).rolling(LOOK_AHEAD).max().shift(-LOOK_AHEAD+1)

# Target: 1 if price goes UP by threshold within next 3 candles
df['target'] = ((df['future_max'] - df['close']) / df['close'] > THRESHOLD).astype(int)

# Drop NaN
df_clean = df.dropna()

print(f"Clean data: {len(df_clean)} rows")
print(f"Target distribution:\n{df_clean['target'].value_counts()}")

# =========================================
# FEATURE SELECTION
# =========================================

features = [
    # Basic indicators
    'rsi_14', 'rsi_6', 'rsi_slope',
    'boll', 'boll_ub', 'boll_lb', 'bb_width', 'dist_ma',
    'macd', 'macds', 'macdh', 'macd_slope',
    'atr', 'atr_ratio', 'cci', 'adx', 'adx_slope',
    # Price action
    'close', 'volume', 'vol_trend', 'vol_spike',
    'price_change_1', 'price_change_3', 'price_change_5',
    'high_low_range', 'close_to_high', 'close_to_low',
    # Trend
    'ema_cross', 'trend_strength',
    # Volatility
    'range_expansion', 'rsi_price_div'
]

# Add time features if available
if 'hour' in df_clean.columns:
    features += ['hour', 'day_of_week', 'london_session', 'ny_session', 'overlap_session']

X = df_clean[features]
y = df_clean['target']

print(f"Features: {len(features)}")

# =========================================
# TRAIN/TEST SPLIT (Time Series)
# =========================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Training: {X_train.shape}, Testing: {X_test.shape}")
print(f"Train class dist: {y_train.value_counts().to_dict()}")
print(f"Test class dist: {y_test.value_counts().to_dict()}")

# =========================================
# CLASS BALANCING
# =========================================

# Calculate scale_pos_weight for imbalanced classes
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
# Use sqrt of ratio for partial balancing (prevents over-correction)
scale_pos_weight = np.sqrt(neg_count / pos_count)

print(f"Class imbalance ratio: {neg_count/pos_count:.2f}")
print(f"Using scale_pos_weight: {scale_pos_weight:.2f} (sqrt)")

# =========================================
# LIGHTGBM WITH IMPROVED PARAMS
# =========================================

param_dist = {
    'max_depth': [5, 7, 10, 15],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'n_estimators': [300, 500, 800, 1000],
    'num_leaves': [31, 63, 127, 255],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_samples': [10, 20, 50],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0, 0.1, 0.5, 1.0],
}

clf = lgb.LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    scale_pos_weight=scale_pos_weight,  # PARTIAL CLASS BALANCING
    verbose=-1,
    force_col_wise=True,
    random_state=42
)

print("\nTuning Hyperparameters (this may take a few minutes)...")

# Use TimeSeriesSplit for proper cross-validation
tscv = TimeSeriesSplit(n_splits=3)

random_search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=50,  # More iterations
    scoring='f1',  # Optimize for F1 instead of accuracy
    cv=tscv,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

print(f"\nBest Parameters: {random_search.best_params_}")
best_model = random_search.best_estimator_

# =========================================
# EVALUATION
# =========================================

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"MODEL PERFORMANCE")
print(f"{'='*50}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
print("\nTop 10 Feature Importance:")
importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance.head(10).to_string(index=False))

# =========================================
# SAVE MODEL
# =========================================

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lightgbm_scalper.txt")
best_model.booster_.save_model(MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

# Save params
PARAMS_PATH = os.path.join(PROJECT_ROOT, "models", "lightgbm_params.json")
with open(PARAMS_PATH, 'w') as f:
    params_to_save = {
        'best_params': random_search.best_params_,
        'features': features,
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'scale_pos_weight': float(scale_pos_weight)
    }
    json.dump(params_to_save, f, indent=2)
print(f"Params saved to {PARAMS_PATH}")
