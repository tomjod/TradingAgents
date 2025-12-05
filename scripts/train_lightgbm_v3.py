"""
LightGBM Training Script v3 - Multi-Timeframe Features
Added H1 (hourly) features for better trend detection
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, f1_score
from stockstats import wrap
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================================
# LOAD DATA
# =========================================

DATA_PATH_M5 = os.path.join(PROJECT_ROOT, "training_data", "XAUUSDm_m5.csv")
DATA_PATH_H1 = os.path.join(PROJECT_ROOT, "training_data", "XAUUSDm_h1.csv")

df_m5 = pd.read_csv(DATA_PATH_M5)
print(f"Loaded M5: {len(df_m5)} rows")

# Try to load H1 data (optional)
has_h1_data = False
if os.path.exists(DATA_PATH_H1):
    df_h1 = pd.read_csv(DATA_PATH_H1)
    has_h1_data = True
    print(f"Loaded H1: {len(df_h1)} rows")
else:
    print("⚠️ No H1 data found. Run: python scripts/fetch_h1_data.py first")
    print("Continuing with M5 features only...")

# Rename columns
df_m5.rename(columns={'tick_volume': 'volume'}, inplace=True)

# =========================================
# M5 FEATURE ENGINEERING
# =========================================

stock = wrap(df_m5)

# Basic indicators
_ = stock['rsi_14']
_ = stock['rsi_6']
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

df = pd.DataFrame(stock)

# Price action features
df['rsi_slope'] = df['rsi_14'] - df['rsi_14'].shift(3)
df['macd_slope'] = df['macd'] - df['macd'].shift(3)
df['bb_width'] = (df['boll_ub'] - df['boll_lb']) / df['boll']
df['dist_ma'] = (df['close'] - df['boll']) / df['boll']
df['vol_trend'] = df['volume'] / df['volume'].rolling(20).mean()

# Momentum
df['price_change_1'] = df['close'].pct_change(1)
df['price_change_3'] = df['close'].pct_change(3)
df['price_change_5'] = df['close'].pct_change(5)
df['high_low_range'] = (df['high'] - df['low']) / df['close']
df['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 0.001)
df['close_to_low'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.001)

# Trend strength
df['ema_cross'] = (df['close_5_ema'] - df['close_20_ema']) / df['close']
df['trend_strength'] = (df['close'] - df['close_50_sma']) / df['close_50_sma']
df['adx_slope'] = df['adx'] - df['adx'].shift(3)

# Volatility
df['atr_ratio'] = df['atr'] / df['close']
df['vol_spike'] = df['volume'] / df['volume'].shift(1)
df['range_expansion'] = df['high_low_range'] / df['high_low_range'].rolling(10).mean()

# RSI divergence
df['rsi_price_div'] = df['rsi_14'].diff(5) - (df['close'].pct_change(5) * 100)

# Time features
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
    df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
    df['overlap_session'] = ((df['hour'] >= 13) & (df['hour'] <= 16)).astype(int)

# =========================================
# H1 FEATURE ENGINEERING (if available)
# =========================================

if has_h1_data:
    print("\nAdding H1 features...")
    df_h1.rename(columns={'tick_volume': 'volume'}, inplace=True)
    
    # Calculate H1 indicators
    h1_stock = wrap(df_h1)
    _ = h1_stock['rsi_14']
    _ = h1_stock['close_10_ema']
    _ = h1_stock['close_20_ema']
    _ = h1_stock['atr']
    _ = h1_stock['adx']
    
    df_h1 = pd.DataFrame(h1_stock)
    
    # H1 trend signals
    df_h1['h1_trend'] = np.where(
        df_h1['close'] > df_h1['close_10_ema'],
        np.where(df_h1['close_10_ema'] > df_h1['close_20_ema'], 2, 1),  # Strong/Weak UP
        np.where(df_h1['close_10_ema'] < df_h1['close_20_ema'], -2, -1)  # Strong/Weak DOWN
    )
    
    # Rename for merge
    df_h1_features = df_h1[['time', 'rsi_14', 'close_10_ema', 'close_20_ema', 'atr', 'adx', 'h1_trend']].copy()
    df_h1_features.columns = ['time', 'h1_rsi', 'h1_ema_10', 'h1_ema_20', 'h1_atr', 'h1_adx', 'h1_trend']
    df_h1_features['time'] = pd.to_datetime(df_h1_features['time'])
    
    # Round M5 time to nearest hour for merge
    df['time_h1'] = df['time'].dt.floor('H')
    df_h1_features['time_h1'] = df_h1_features['time'].dt.floor('H')
    
    # Merge H1 features into M5 data
    df = df.merge(df_h1_features.drop('time', axis=1), on='time_h1', how='left')
    df = df.drop('time_h1', axis=1)
    
    # Fill NaN values
    df['h1_rsi'] = df['h1_rsi'].ffill()
    df['h1_ema_10'] = df['h1_ema_10'].ffill()
    df['h1_ema_20'] = df['h1_ema_20'].ffill()
    df['h1_atr'] = df['h1_atr'].ffill()
    df['h1_adx'] = df['h1_adx'].ffill()
    df['h1_trend'] = df['h1_trend'].ffill()
    
    # Derived H1 features
    df['h1_rsi_diff'] = df['rsi_14'] - df['h1_rsi']  # RSI alignment
    df['m5_h1_ema_ratio'] = df['close_10_ema'] / df['h1_ema_10']  # Timeframe alignment
    df['atr_ratio_h1'] = df['atr'] / df['h1_atr']  # Volatility expansion

# =========================================
# TARGET DEFINITION
# =========================================

LOOK_AHEAD = 3
THRESHOLD = 0.001

df['future_max'] = df['high'].shift(-1).rolling(LOOK_AHEAD).max().shift(-LOOK_AHEAD+1)
df['target'] = ((df['future_max'] - df['close']) / df['close'] > THRESHOLD).astype(int)

df_clean = df.dropna()

print(f"\nClean data: {len(df_clean)} rows")
print(f"Target distribution:\n{df_clean['target'].value_counts()}")

# =========================================
# FEATURE SELECTION
# =========================================

features = [
    # M5 indicators
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

if 'hour' in df_clean.columns:
    features += ['hour', 'day_of_week', 'london_session', 'ny_session', 'overlap_session']

# Add H1 features if available
if has_h1_data and 'h1_rsi' in df_clean.columns:
    features += ['h1_rsi', 'h1_adx', 'h1_trend', 'h1_rsi_diff', 'm5_h1_ema_ratio', 'atr_ratio_h1']
    print("✅ H1 features added!")

X = df_clean[features]
y = df_clean['target']

print(f"Total Features: {len(features)}")

# =========================================
# TRAIN/TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"\nTraining: {X_train.shape}, Testing: {X_test.shape}")

# Class balancing
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = np.sqrt(neg_count / pos_count)

print(f"Scale pos weight: {scale_pos_weight:.2f}")

# =========================================
# LIGHTGBM TRAINING
# =========================================

param_dist = {
    'max_depth': [5, 7, 10, 15],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'n_estimators': [300, 500, 800, 1000],
    'num_leaves': [31, 63, 127],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_samples': [10, 20, 50],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5],
}

clf = lgb.LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    scale_pos_weight=scale_pos_weight,
    verbose=-1,
    force_col_wise=True,
    random_state=42
)

print("\n🔄 Tuning hyperparameters...")

tscv = TimeSeriesSplit(n_splits=3)

random_search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=50,
    scoring='f1',
    cv=tscv,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

print(f"\nBest Params: {random_search.best_params_}")
best_model = random_search.best_estimator_

# =========================================
# EVALUATION
# =========================================

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"MODEL PERFORMANCE (v3 - Multi-Timeframe)")
print(f"{'='*50}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
print("\nTop 15 Feature Importance:")
importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance.head(15).to_string(index=False))

# =========================================
# SAVE MODEL
# =========================================

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lightgbm_scalper.txt")
best_model.booster_.save_model(MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")

PARAMS_PATH = os.path.join(PROJECT_ROOT, "models", "lightgbm_params.json")
with open(PARAMS_PATH, 'w') as f:
    params_to_save = {
        'best_params': random_search.best_params_,
        'features': features,
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'scale_pos_weight': float(scale_pos_weight),
        'has_h1_features': has_h1_data
    }
    json.dump(params_to_save, f, indent=2)
print(f"Params saved to {PARAMS_PATH}")
