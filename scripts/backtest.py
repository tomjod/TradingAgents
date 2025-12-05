"""
Backtesting System for XAUUSD Scalping Bot
Simulates the soldier.py trading logic on historical data.

Usage:
    python scripts/backtest.py                    # Use default model
    python scripts/backtest.py --model path.txt  # Use specific model
    python scripts/backtest.py --days 30         # Last 30 days only
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from stockstats import wrap
from datetime import datetime, timedelta
import argparse
import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================
# CONFIGURATION (mirrors soldier.py)
# ============================================

def load_config():
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Trading params from config
SL_POINTS = config.get("sl_points", 2000)
TP_POINTS = config.get("tp_points", 500)
MAX_SPREAD = config.get("max_spread", 120)
TRAILING_STOP_START = config.get("trailing_stop_start", 350)
TRAILING_STEP = config.get("trailing_step", 100)
MAX_POSITIONS = config.get("max_positions", 4)

# Backtest-specific params
POINT = 0.01  # XAU typically uses 0.01 as point
SPREAD_POINTS = 30  # Average spread in points


# ============================================
# FEATURE CALCULATION (mirrors soldier.py)
# ============================================

def calculate_features(df):
    """Calculate all features for the model (same as soldier.py)"""
    stock = wrap(df.copy())
    
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
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
        df['overlap_session'] = ((df['hour'] >= 13) & (df['hour'] <= 16)).astype(int)
    
    # H1 features (simplified - in real bot these come from H1 data)
    df['h1_rsi'] = df['rsi_14'].rolling(12).mean()  # Approximate
    df['h1_adx'] = df['adx'].rolling(12).mean()
    df['h1_trend'] = np.where(df['close'] > df['close_10_ema'], 1, -1)
    df['h1_rsi_diff'] = df['rsi_14'] - df['h1_rsi']
    df['m5_h1_ema_ratio'] = df['close_10_ema'] / df['close_10_ema'].rolling(12).mean()
    df['atr_ratio_h1'] = df['atr'] / df['atr'].rolling(12).mean()
    
    return df


# ============================================
# REGIME DETECTION (mirrors soldier.py)
# ============================================

def detect_regime(df, idx):
    """Detect market regime for a specific candle"""
    if idx < 30:
        return 'UNKNOWN', get_default_params()
    
    window = df.iloc[idx-30:idx]
    
    # ATR ratio
    atr_14 = window['atr'].iloc[-14:].mean()
    atr_avg = window['atr'].mean()
    atr_ratio = atr_14 / atr_avg if atr_avg > 0 else 1.0
    
    # ADX
    adx = window['adx'].iloc[-1] if 'adx' in window.columns else 0
    
    if atr_ratio > 1.5:
        regime = 'VOLATILE'
        params = {
            'entry_threshold': 5,
            'prob_threshold_buy': 0.60,
            'prob_threshold_sell': 0.40,
            'prob_discard_buy': 0.45,
            'prob_discard_sell': 0.55,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'bb_points': 3,
            'trend_points': 1,
            'prob_bonus_threshold': 0.68,
        }
    elif adx > 25:
        regime = 'TRENDING'
        params = {
            'entry_threshold': 3,
            'prob_threshold_buy': 0.52,
            'prob_threshold_sell': 0.48,
            'prob_discard_buy': 0.38,
            'prob_discard_sell': 0.62,
            'rsi_oversold': 45,
            'rsi_overbought': 55,
            'bb_points': 1,
            'trend_points': 2,
            'prob_bonus_threshold': 0.58,
        }
    else:
        regime = 'RANGING'
        params = {
            'entry_threshold': 4,
            'prob_threshold_buy': 0.55,
            'prob_threshold_sell': 0.45,
            'prob_discard_buy': 0.40,
            'prob_discard_sell': 0.60,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'bb_points': 3,
            'trend_points': 1,
            'prob_bonus_threshold': 0.62,
        }
    
    return regime, params


def get_default_params():
    return {
        'entry_threshold': 4,
        'prob_threshold_buy': 0.55,
        'prob_threshold_sell': 0.45,
        'prob_discard_buy': 0.40,
        'prob_discard_sell': 0.60,
        'rsi_oversold': 40,
        'rsi_overbought': 60,
        'bb_points': 2,
        'trend_points': 1,
        'prob_bonus_threshold': 0.62,
    }


# ============================================
# SCORING LOGIC (mirrors soldier.py)
# ============================================

def calculate_score(row, prob, bias, regime_params):
    """Calculate entry score - same logic as soldier.py"""
    score = 0
    reasons = []
    
    if bias == "BULLISH":
        if prob < regime_params['prob_discard_buy']:
            return -99, ["DISCARD"]
        
        if row['close'] <= row['boll_lb'] * 1.005:
            score += regime_params['bb_points']
            reasons.append("DIP")
        
        if row['rsi_14'] < regime_params['rsi_oversold']:
            score += 1
            reasons.append(f"RSI:{row['rsi_14']:.0f}")
        
        if prob > regime_params['prob_threshold_buy']:
            score += 1
            reasons.append(f"PROB:{prob:.2f}")
        
        if prob > regime_params['prob_bonus_threshold']:
            score += 1
            reasons.append("STRONG_PROB")
        
        if row['close'] > row['close_10_ema']:
            score += regime_params['trend_points']
            reasons.append("TREND")
    
    elif bias == "BEARISH":
        if prob > regime_params['prob_discard_sell']:
            return -99, ["DISCARD"]
        
        if row['close'] >= row['boll_ub'] * 0.995:
            score += regime_params['bb_points']
            reasons.append("PEAK")
        
        if row['rsi_14'] > regime_params['rsi_overbought']:
            score += 1
            reasons.append(f"RSI:{row['rsi_14']:.0f}")
        
        if prob < regime_params['prob_threshold_sell']:
            score += 1
            reasons.append(f"PROB:{prob:.2f}")
        
        if row['close'] < row['close_10_ema']:
            score += regime_params['trend_points']
            reasons.append("TREND")
        
        if prob < (1.0 - regime_params['prob_bonus_threshold']):
            score += 1
            reasons.append("STRONG_PROB")
    
    return score, reasons


# ============================================
# POSITION SIMULATION
# ============================================

class Position:
    def __init__(self, entry_price, direction, sl, tp, entry_time):
        self.entry_price = entry_price
        self.direction = direction  # "BUY" or "SELL"
        self.sl = sl
        self.tp = tp
        self.entry_time = entry_time
        self.exit_price = None
        self.exit_time = None
        self.profit_points = 0
        self.closed = False
        self.exit_reason = ""


def simulate_position(position, candle, current_bias):
    """Simulate position for one candle - check SL/TP/signal exit"""
    if position.closed:
        return position
    
    high = candle['high']
    low = candle['low']
    close = candle['close']
    
    if position.direction == "BUY":
        # Check SL
        if low <= position.sl:
            position.exit_price = position.sl
            position.profit_points = (position.sl - position.entry_price) / POINT
            position.closed = True
            position.exit_reason = "SL"
            position.exit_time = candle['time']
        # Check TP (if set)
        elif position.tp and high >= position.tp:
            position.exit_price = position.tp
            position.profit_points = (position.tp - position.entry_price) / POINT
            position.closed = True
            position.exit_reason = "TP"
            position.exit_time = candle['time']
        # Check bias reversal with profit
        elif current_bias == "BEARISH":
            current_profit = (close - position.entry_price) / POINT
            if current_profit > 50:  # Exit on reversal with profit
                position.exit_price = close
                position.profit_points = current_profit
                position.closed = True
                position.exit_reason = "REVERSAL"
                position.exit_time = candle['time']
    
    elif position.direction == "SELL":
        # Check SL
        if high >= position.sl:
            position.exit_price = position.sl
            position.profit_points = (position.entry_price - position.sl) / POINT
            position.closed = True
            position.exit_reason = "SL"
            position.exit_time = candle['time']
        # Check TP (if set)
        elif position.tp and low <= position.tp:
            position.exit_price = position.tp
            position.profit_points = (position.entry_price - position.tp) / POINT
            position.closed = True
            position.exit_reason = "TP"
            position.exit_time = candle['time']
        # Check bias reversal with profit
        elif current_bias == "BULLISH":
            current_profit = (position.entry_price - close) / POINT
            if current_profit > 50:
                position.exit_price = close
                position.profit_points = current_profit
                position.closed = True
                position.exit_reason = "REVERSAL"
                position.exit_time = candle['time']
    
    return position


# ============================================
# MAIN BACKTEST LOGIC
# ============================================

def run_backtest(model_path, data_path, start_date=None, end_date=None):
    """Run backtest simulation"""
    print(f"\n{'='*60}")
    print("🔬 BACKTEST STARTING")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    
    # Load model
    model = lgb.Booster(model_file=model_path)
    print("✅ Model loaded")
    
    # Load data
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    
    # Filter by date range
    if start_date:
        df = df[df['time'] >= start_date]
    if end_date:
        df = df[df['time'] <= end_date]
    
    print(f"Data range: {df['time'].min()} to {df['time'].max()}")
    print(f"Total candles: {len(df)}")
    
    # Calculate all features
    print("Calculating features...")
    df = calculate_features(df)
    df = df.dropna()
    print(f"Clean data: {len(df)} candles")
    
    # Get feature list from model params
    params_path = model_path.replace('.txt', '_params.json').replace('_scalper', '_params')
    if os.path.exists(params_path.replace('_params_params', '_params')):
        params_path = params_path.replace('_params_params', '_params')
    
    # Default feature list (v3)
    features = [
        'rsi_14', 'rsi_6', 'rsi_slope',
        'boll', 'boll_ub', 'boll_lb', 'bb_width', 'dist_ma',
        'macd', 'macds', 'macdh', 'macd_slope',
        'atr', 'atr_ratio', 'cci', 'adx', 'adx_slope',
        'close', 'volume', 'vol_trend', 'vol_spike',
        'price_change_1', 'price_change_3', 'price_change_5',
        'high_low_range', 'close_to_high', 'close_to_low',
        'ema_cross', 'trend_strength',
        'range_expansion', 'rsi_price_div',
        'hour', 'day_of_week', 'london_session', 'ny_session', 'overlap_session',
        'h1_rsi', 'h1_adx', 'h1_trend', 'h1_rsi_diff', 'm5_h1_ema_ratio', 'atr_ratio_h1'
    ]
    
    # Make predictions
    print("Running predictions...")
    X = df[features].fillna(0)
    df['prob'] = model.predict(X)
    
    # Simulation state
    positions = []
    closed_positions = []
    trade_cooldown = 0
    last_trade_idx = -60  # 60 candles = 5 hours on M5
    
    # Stats
    trades_taken = 0
    
    print("\n🔄 Running simulation...")
    
    for i in range(100, len(df)):  # Start after 100 candles for indicators
        row = df.iloc[i]
        prob = row['prob']
        
        # Update open positions
        for pos in positions:
            current_bias = "BULLISH" if prob > 0.55 else "BEARISH" if prob < 0.45 else "NEUTRAL"
            pos = simulate_position(pos, row, current_bias)
            if pos.closed and pos not in closed_positions:
                closed_positions.append(pos)
        
        # Remove closed positions
        positions = [p for p in positions if not p.closed]
        
        # Skip if max positions reached
        if len(positions) >= MAX_POSITIONS:
            continue
        
        # Skip if in cooldown
        if i - last_trade_idx < 60:  # ~5 hours cooldown
            continue
        
        # Get regime
        regime, regime_params = detect_regime(df, i)
        
        # Determine bias from probability
        if prob > 0.55:
            bias = "BULLISH"
        elif prob < 0.45:
            bias = "BEARISH"
        else:
            continue  # Neutral, skip
        
        # Calculate score
        score, reasons = calculate_score(row, prob, bias, regime_params)
        
        # Check if we should enter
        if score >= regime_params['entry_threshold']:
            entry_price = row['close']
            
            if bias == "BULLISH":
                sl = entry_price - SL_POINTS * POINT
                tp = None  # Exit on signal
                direction = "BUY"
            else:
                sl = entry_price + SL_POINTS * POINT
                tp = None
                direction = "SELL"
            
            pos = Position(entry_price, direction, sl, tp, row['time'])
            positions.append(pos)
            trades_taken += 1
            last_trade_idx = i
    
    # Close any remaining positions at last candle
    last_row = df.iloc[-1]
    for pos in positions:
        if not pos.closed:
            pos.exit_price = last_row['close']
            if pos.direction == "BUY":
                pos.profit_points = (pos.exit_price - pos.entry_price) / POINT
            else:
                pos.profit_points = (pos.entry_price - pos.exit_price) / POINT
            pos.closed = True
            pos.exit_reason = "END"
            pos.exit_time = last_row['time']
            closed_positions.append(pos)
    
    # ============================================
    # RESULTS
    # ============================================
    
    print(f"\n{'='*60}")
    print("📊 BACKTEST RESULTS")
    print(f"{'='*60}")
    
    total_trades = len(closed_positions)
    if total_trades == 0:
        print("❌ No trades executed!")
        return {}
    
    wins = sum(1 for p in closed_positions if p.profit_points > 0)
    losses = sum(1 for p in closed_positions if p.profit_points < 0)
    
    total_profit_points = sum(p.profit_points for p in closed_positions)
    gross_profit = sum(p.profit_points for p in closed_positions if p.profit_points > 0)
    gross_loss = abs(sum(p.profit_points for p in closed_positions if p.profit_points < 0))
    
    win_rate = wins / total_trades * 100
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    avg_win = gross_profit / wins if wins > 0 else 0
    avg_loss = gross_loss / losses if losses > 0 else 0
    
    # Max drawdown
    cumulative = []
    running_total = 0
    peak = 0
    max_dd = 0
    for p in closed_positions:
        running_total += p.profit_points
        cumulative.append(running_total)
        if running_total > peak:
            peak = running_total
        dd = peak - running_total
        if dd > max_dd:
            max_dd = dd
    
    # Exit reasons breakdown
    exit_reasons = {}
    for p in closed_positions:
        exit_reasons[p.exit_reason] = exit_reasons.get(p.exit_reason, 0) + 1
    
    print(f"Total Trades: {total_trades}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Total Profit: {total_profit_points:.0f} points")
    print(f"Avg Win: {avg_win:.0f} pts | Avg Loss: {avg_loss:.0f} pts")
    print(f"Max Drawdown: {max_dd:.0f} points")
    print(f"\nExit Reasons: {exit_reasons}")
    
    # Return results dict
    results = {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_profit_points': total_profit_points,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': max_dd,
        'exit_reasons': exit_reasons
    }
    
    return results


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest XAUUSD Scalping Bot")
    parser.add_argument("--model", default=os.path.join(PROJECT_ROOT, "models", "lightgbm_scalper.txt"),
                        help="Path to model file")
    parser.add_argument("--data", default=os.path.join(PROJECT_ROOT, "training_data", "XAUUSDm_m5.csv"),
                        help="Path to data file")
    parser.add_argument("--days", type=int, default=None,
                        help="Only use last N days of data")
    
    args = parser.parse_args()
    
    start_date = None
    if args.days:
        start_date = datetime.now() - timedelta(days=args.days)
    
    results = run_backtest(args.model, args.data, start_date=start_date)
    
    # Save results
    results_path = os.path.join(PROJECT_ROOT, "backtest_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Results saved to {results_path}")
