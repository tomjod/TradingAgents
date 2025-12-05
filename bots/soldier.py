import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from stockstats import wrap
import time
import json
import os
import sys
import asyncio
from datetime import datetime
import traceback
from dotenv import load_dotenv

import yaml

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Numba JIT-compiled functions for HFT optimization (must be after sys.path setup)
from bots.numba_indicators import (
    calculate_slopes,
    calculate_bb_metrics,
    calculate_vol_trend,
    calculate_trailing_stop,
    calculate_dynamic_lot
)
from bots.risk_guardian import RiskGuardian

# Load environment variables
load_dotenv()

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Load Config
config = load_config()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration
SYMBOL = config["symbol"]
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1
}
TIMEFRAME = TIMEFRAME_MAP.get(config["timeframe"], mt5.TIMEFRAME_M5)

MODEL_PATH = os.path.join(PROJECT_ROOT, config["model_path"])
BIAS_FILE = os.path.join(PROJECT_ROOT, config["bias_file"])

# Config file path for hot-reload
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
_config_mtime = os.path.getmtime(CONFIG_PATH)

def reload_config():
    """Hot-reload config if file changed"""
    global config, _config_mtime
    global TRAILING_STOP_START, TRAILING_STEP, TRAILING_TP_DISTANCE
    global MAX_POSITIONS, CUT_LOSS_POINTS, TRADE_COOLDOWN, MAX_SPREAD
    global TRADING_HOURS_ENABLED, TRADING_START_HOUR, TRADING_END_HOUR, TRADE_WEEKENDS
    
    try:
        current_mtime = os.path.getmtime(CONFIG_PATH)
        if current_mtime > _config_mtime:
            with open(CONFIG_PATH, "r") as f:
                config = yaml.safe_load(f)
            _config_mtime = current_mtime
            
            # Reload trading parameters
            TRAILING_STOP_START = config["trailing_stop_start"]
            TRAILING_STEP = config["trailing_step"]
            TRAILING_TP_DISTANCE = config.get("trailing_tp_distance", 500)
            MAX_POSITIONS = config["max_positions"]
            CUT_LOSS_POINTS = config.get("cut_loss_points", 2000)
            TRADE_COOLDOWN = config.get("trade_cooldown", 120)
            MAX_SPREAD = config["max_spread"]
            
            # Trading hours
            TRADING_HOURS = config.get("trading_hours", {})
            TRADING_HOURS_ENABLED = TRADING_HOURS.get("enabled", False)
            TRADING_START_HOUR = TRADING_HOURS.get("start_hour", 0)
            TRADING_END_HOUR = TRADING_HOURS.get("end_hour", 23)
            TRADE_WEEKENDS = TRADING_HOURS.get("trade_weekends", False)
            
            print("🔄 Config reloaded!")
            return True
    except Exception as e:
        print(f"⚠️ Config reload error: {e}")
    return False

# Initial values (will be hot-reloaded)
VOLUME = config["volume"]
SL_POINTS = config["sl_points"]
TP_POINTS = config["tp_points"]
MAX_SPREAD = config["max_spread"]
TRAILING_STOP_START = config["trailing_stop_start"]
TRAILING_STEP = config["trailing_step"]
TRAILING_TP_DISTANCE = config.get("trailing_tp_distance", 500)
MAX_POSITIONS = config["max_positions"]
CUT_LOSS_POINTS = config.get("cut_loss_points", 2000)
TRADE_COOLDOWN = config.get("trade_cooldown", 120)

# Trading Hours Config
TRADING_HOURS = config.get("trading_hours", {})
TRADING_HOURS_ENABLED = TRADING_HOURS.get("enabled", False)
TRADING_START_HOUR = TRADING_HOURS.get("start_hour", 0)
TRADING_END_HOUR = TRADING_HOURS.get("end_hour", 23)
TRADE_WEEKENDS = TRADING_HOURS.get("trade_weekends", False)

def is_trading_time():
    """Check if current time is within trading hours"""
    if not TRADING_HOURS_ENABLED:
        return True
    
    now = datetime.now()
    current_hour = now.hour
    day_of_week = now.weekday()  # 0=Monday, 6=Sunday
    
    # Check weekend
    if not TRADE_WEEKENDS and day_of_week >= 5:  # Saturday=5, Sunday=6
        return False
    
    # Check hours
    if TRADING_START_HOUR <= TRADING_END_HOUR:
        return TRADING_START_HOUR <= current_hour <= TRADING_END_HOUR
    else:  # Handles overnight like 22:00 to 06:00
        return current_hour >= TRADING_START_HOUR or current_hour <= TRADING_END_HOUR


def check_trailing_stop(position):
    """
    Updates Stop Loss to lock in profits.
    """
    tick = mt5.symbol_info_tick(position.symbol)
    point = mt5.symbol_info(position.symbol).point
    
    if position.type == mt5.ORDER_TYPE_BUY:
        # Current Profit distance in points
        dist = (tick.bid - position.price_open) / point
        
        if dist > TRAILING_STOP_START:
            new_sl = tick.bid - TRAILING_STOP_START * point
            if new_sl > position.sl + TRAILING_STEP * point:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": position.ticket,
                    "sl": new_sl,
                    "tp": position.tp,
                    "magic": 999000,
                }
                res = mt5.order_send(request)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"Trailing Stop Updated (BUY): {new_sl}")

    elif position.type == mt5.ORDER_TYPE_SELL:
        dist = (position.price_open - tick.ask) / point
        
        if dist > TRAILING_STOP_START:
            new_sl = tick.ask + TRAILING_STOP_START * point
            if position.sl == 0 or new_sl < position.sl - TRAILING_STEP * point:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": position.ticket,
                    "sl": new_sl,
                    "tp": position.tp,
                    "magic": 999000,
                }
                res = mt5.order_send(request)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"Trailing Stop Updated (SELL): {new_sl}")

def initialize_mt5():
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    
    max_retries = 3
    for i in range(max_retries):
        try:
            if not mt5.initialize():
                print(f"initialize() failed, error code = {mt5.last_error()}, retrying ({i+1}/{max_retries})...")
                time.sleep(2)
                continue
                
            if login and password and server:
                print(f"Attempting login: Account={login}, Server={server}")
                authorized = mt5.login(int(login), password=password, server=server)
                if authorized:
                    print(f"✅ Connected to MT5 account #{login}")
                    return True
                else:
                    error = mt5.last_error()
                    print(f"\n{'='*60}")
                    print(f"❌ MT5 CONNECTION FAILED")
                    print(f"{'='*60}")
                    print(f"   Account: {login}")
                    print(f"   Server: {server}")
                    print(f"   Error: {error}")
                    print(f"\n🔧 POSSIBLE SOLUTIONS:")
                    if "Invalid account" in str(error) or error[0] == -2:
                        print(f"   1. Demo account may have expired (Exness demos expire after ~30 days)")
                        print(f"   2. Create a new demo account at https://my.exness.com/")
                        print(f"   3. Update credentials in .env file")
                    elif "Authorization failed" in str(error):
                        print(f"   1. Check password in .env file")
                        print(f"   2. Check server name (should match MT5 terminal)")
                    else:
                        print(f"   1. Check internet connection")
                        print(f"   2. Restart MT5 terminal")
                        print(f"   3. Check broker server status")
                    print(f"{'='*60}\n")
                    mt5.shutdown()
                    return False
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize MT5: {e}")
            time.sleep(2)
            
    return True

def initialize_symbol():
    # Check if MT5 is actually connected
    terminal_info = mt5.terminal_info()
    if terminal_info is None:
        print(f"\n❌ MT5 not connected! Cannot initialize symbol.")
        print(f"   Run soldier.py again after fixing MT5 connection.\n")
        return False
    
    # Attempt to enable the symbol in Market Watch
    if not mt5.symbol_select(SYMBOL, True):
        print(f"⚠️ Failed to select {SYMBOL}, trying to find it...")
        
        # Check if symbols are available at all
        symbols = mt5.symbols_get()
        if symbols is None or len(symbols) == 0:
            print(f"\n❌ NO SYMBOLS AVAILABLE")
            print(f"   MT5 may not be fully connected to broker.")
            print(f"   1. Open MT5 terminal manually")
            print(f"   2. Wait for symbols to load")
            print(f"   3. Try again\n")
            return False
        
        # Check if symbol exists
        info = mt5.symbol_info(SYMBOL)
        if info is None:
            print(f"\n❌ Symbol '{SYMBOL}' not found!")
            similar = [s.name for s in symbols if "XAU" in s.name.upper() or "GOLD" in s.name.upper()]
            if similar:
                print(f"   Available Gold symbols: {similar[:5]}")
                print(f"   Update 'symbol' in config.yaml to one of these.\n")
            else:
                print(f"   No Gold symbols found. Check broker's available instruments.\n")
            return False
            
    # Check if data is available
    print(f"✅ Symbol {SYMBOL} selected. Checking data...")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
    if rates is None:
        print(f"⚠️ Still no data for {SYMBOL}. History might be syncing...")
        return False
        
    return True

def load_bias():
    try:
        if os.path.exists(BIAS_FILE):
            with open(BIAS_FILE, "r") as f:
                data = json.load(f)
                return data.get("bias", "NEUTRAL")
    except Exception as e:
        print(f"Error reading bias: {e}")
    return "NEUTRAL"

def get_data():
    # Fetch last 100 candles for indicators
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 100)
    if rates is None:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Rename tick_volume to volume for stockstats
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    
    return df

def get_higher_timeframe_trend():
    """
    Analyze H1 and M15 timeframes to determine overall trend.
    Returns: 'UP', 'DOWN', or 'NEUTRAL'
    """
    try:
        # Get H1 data (last 50 candles)
        h1_rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 50)
        if h1_rates is None:
            return 'NEUTRAL', {}
        
        h1_df = pd.DataFrame(h1_rates)
        
        # Calculate H1 EMAs
        h1_ema_10 = h1_df['close'].ewm(span=10).mean().iloc[-1]
        h1_ema_20 = h1_df['close'].ewm(span=20).mean().iloc[-1]
        h1_ema_50 = h1_df['close'].rolling(50).mean().iloc[-1]
        h1_close = h1_df['close'].iloc[-1]
        
        # Calculate H1 RSI
        delta = h1_df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        h1_rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # Get M15 data (last 50 candles)
        m15_rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 50)
        if m15_rates is None:
            return 'NEUTRAL', {}
        
        m15_df = pd.DataFrame(m15_rates)
        m15_ema_10 = m15_df['close'].ewm(span=10).mean().iloc[-1]
        m15_ema_20 = m15_df['close'].ewm(span=20).mean().iloc[-1]
        m15_close = m15_df['close'].iloc[-1]
        
        # Determine H1 trend
        h1_trend_score = 0
        if h1_close > h1_ema_10 > h1_ema_20:
            h1_trend_score = 2  # Strong UP
        elif h1_close > h1_ema_20:
            h1_trend_score = 1  # Weak UP
        elif h1_close < h1_ema_10 < h1_ema_20:
            h1_trend_score = -2  # Strong DOWN
        elif h1_close < h1_ema_20:
            h1_trend_score = -1  # Weak DOWN
        
        # Determine M15 trend
        m15_trend_score = 0
        if m15_close > m15_ema_10 > m15_ema_20:
            m15_trend_score = 1
        elif m15_close < m15_ema_10 < m15_ema_20:
            m15_trend_score = -1
        
        # Combined trend
        total_score = h1_trend_score + m15_trend_score
        
        trend_info = {
            'h1_trend': 'UP' if h1_trend_score > 0 else 'DOWN' if h1_trend_score < 0 else 'NEUTRAL',
            'm15_trend': 'UP' if m15_trend_score > 0 else 'DOWN' if m15_trend_score < 0 else 'NEUTRAL',
            'h1_rsi': round(h1_rsi, 1),
            'h1_ema_10': round(h1_ema_10, 2),
            'h1_ema_20': round(h1_ema_20, 2),
            'score': total_score
        }
        
        if total_score >= 2:
            return 'UP', trend_info
        elif total_score <= -2:
            return 'DOWN', trend_info
        else:
            return 'NEUTRAL', trend_info
            
    except Exception as e:
        print(f"MTF Error: {e}")
        return 'NEUTRAL', {}

def detect_market_regime():
    """
    Detect market regime: TRENDING, RANGING, or VOLATILE
    Returns regime and recommended strategy adjustments
    """
    try:
        # Get H1 data for regime detection
        h1_rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 30)
        if h1_rates is None:
            return 'UNKNOWN', {}
        
        h1_df = pd.DataFrame(h1_rates)
        
        # Calculate ATR (volatility)
        h1_df['tr'] = np.maximum(
            h1_df['high'] - h1_df['low'],
            np.maximum(
                abs(h1_df['high'] - h1_df['close'].shift(1)),
                abs(h1_df['low'] - h1_df['close'].shift(1))
            )
        )
        atr_14 = h1_df['tr'].rolling(14).mean().iloc[-1]
        atr_avg = h1_df['tr'].rolling(14).mean().mean()  # Historical average
        
        # ATR ratio (current vs average)
        atr_ratio = atr_14 / atr_avg if atr_avg > 0 else 1.0
        
        # Calculate ADX (trend strength)
        # Simplified ADX calculation
        h1_df['dm_plus'] = np.where(
            (h1_df['high'] - h1_df['high'].shift(1)) > (h1_df['low'].shift(1) - h1_df['low']),
            np.maximum(h1_df['high'] - h1_df['high'].shift(1), 0),
            0
        )
        h1_df['dm_minus'] = np.where(
            (h1_df['low'].shift(1) - h1_df['low']) > (h1_df['high'] - h1_df['high'].shift(1)),
            np.maximum(h1_df['low'].shift(1) - h1_df['low'], 0),
            0
        )
        
        smoothed_tr = h1_df['tr'].rolling(14).sum()
        smoothed_dm_plus = h1_df['dm_plus'].rolling(14).sum()
        smoothed_dm_minus = h1_df['dm_minus'].rolling(14).sum()
        
        di_plus = 100 * (smoothed_dm_plus / smoothed_tr)
        di_minus = 100 * (smoothed_dm_minus / smoothed_tr)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 0.001)
        adx = dx.rolling(14).mean().iloc[-1]
        
        # Determine regime
        regime_info = {
            'atr_ratio': round(atr_ratio, 2),
            'adx': round(adx, 1) if not np.isnan(adx) else 0,
            'trailing_multiplier': 1.0,
            'entry_threshold': 3
        }
        
        if atr_ratio > 1.5:
            # HIGH VOLATILITY - be cautious
            regime = 'VOLATILE'
            regime_info['trailing_multiplier'] = 1.5  # Wider trailing stop
            regime_info['entry_threshold'] = 4  # Stricter entry
        elif adx > 25:
            # TRENDING - ride the trend
            regime = 'TRENDING'
            regime_info['trailing_multiplier'] = 0.8  # Tighter trailing to lock profits
            regime_info['entry_threshold'] = 3
        else:
            # RANGING - mean reversion
            regime = 'RANGING'
            regime_info['trailing_multiplier'] = 1.2
            regime_info['entry_threshold'] = 4  # Stricter in ranges
        
        return regime, regime_info
        
    except Exception as e:
        print(f"Regime Detection Error: {e}")
        return 'UNKNOWN', {}

def calculate_features(df):
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
    
    # Convert back to DF
    df_calc = pd.DataFrame(stock)
    
    # 2. Price Action Features (using Numba where possible)
    rsi_array = df_calc['rsi_14'].to_numpy(dtype=np.float64)
    macd_array = df_calc['macd'].to_numpy(dtype=np.float64)
    volume_array = df_calc['volume'].to_numpy(dtype=np.float64)
    
    # JIT-compiled slope calculations
    rsi_slope, macd_slope = calculate_slopes(rsi_array, macd_array)
    df_calc.loc[df_calc.index[-1], 'rsi_slope'] = rsi_slope
    df_calc.loc[df_calc.index[-1], 'macd_slope'] = macd_slope
    
    # JIT-compiled Bollinger metrics
    last_idx = df_calc.index[-1]
    bb_width, dist_ma = calculate_bb_metrics(
        df_calc.loc[last_idx, 'close'],
        df_calc.loc[last_idx, 'boll'],
        df_calc.loc[last_idx, 'boll_ub'],
        df_calc.loc[last_idx, 'boll_lb']
    )
    df_calc.loc[last_idx, 'bb_width'] = bb_width
    df_calc.loc[last_idx, 'dist_ma'] = dist_ma
    
    # JIT-compiled volume trend
    df_calc.loc[last_idx, 'vol_trend'] = calculate_vol_trend(volume_array)
    
    # 3. NEW: Momentum Features
    df_calc['price_change_1'] = df_calc['close'].pct_change(1)
    df_calc['price_change_3'] = df_calc['close'].pct_change(3)
    df_calc['price_change_5'] = df_calc['close'].pct_change(5)
    df_calc['high_low_range'] = (df_calc['high'] - df_calc['low']) / df_calc['close']
    df_calc['close_to_high'] = (df_calc['high'] - df_calc['close']) / (df_calc['high'] - df_calc['low'] + 0.001)
    df_calc['close_to_low'] = (df_calc['close'] - df_calc['low']) / (df_calc['high'] - df_calc['low'] + 0.001)
    
    # 4. NEW: Trend Strength
    df_calc['ema_cross'] = (df_calc['close_5_ema'] - df_calc['close_20_ema']) / df_calc['close']
    df_calc['trend_strength'] = (df_calc['close'] - df_calc['close_50_sma']) / df_calc['close_50_sma']
    df_calc['adx_slope'] = df_calc['adx'] - df_calc['adx'].shift(3)
    
    # 5. NEW: Volatility Features
    df_calc['atr_ratio'] = df_calc['atr'] / df_calc['close']
    df_calc['vol_spike'] = df_calc['volume'] / df_calc['volume'].shift(1)
    df_calc['range_expansion'] = df_calc['high_low_range'] / df_calc['high_low_range'].rolling(10).mean()
    
    # 6. NEW: RSI Divergence Proxy
    df_calc['rsi_price_div'] = df_calc['rsi_14'].diff(5) - (df_calc['close'].pct_change(5) * 100)
    
    # 7. Time-based Features
    if 'time' in df_calc.columns:
        df_calc['time'] = pd.to_datetime(df_calc['time'])
        df_calc['hour'] = df_calc['time'].dt.hour
        df_calc['day_of_week'] = df_calc['time'].dt.dayofweek
        df_calc['london_session'] = ((df_calc['hour'] >= 8) & (df_calc['hour'] <= 16)).astype(int)
        df_calc['ny_session'] = ((df_calc['hour'] >= 13) & (df_calc['hour'] <= 21)).astype(int)
        df_calc['overlap_session'] = ((df_calc['hour'] >= 13) & (df_calc['hour'] <= 16)).astype(int)
    
    # 8. H1 Features (for v3 model with 42 features)
    try:
        h1_rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 20)
        if h1_rates is not None:
            h1_df = pd.DataFrame(h1_rates)
            
            # H1 RSI
            delta = h1_df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            h1_rsi = (100 - (100 / (1 + rs))).iloc[-1]
            df_calc.loc[df_calc.index[-1], 'h1_rsi'] = h1_rsi if not np.isnan(h1_rsi) else 50
            
            # H1 ADX (simplified)
            h1_df['tr'] = np.maximum(h1_df['high'] - h1_df['low'],
                np.maximum(abs(h1_df['high'] - h1_df['close'].shift(1)),
                           abs(h1_df['low'] - h1_df['close'].shift(1))))
            h1_atr = h1_df['tr'].rolling(14).mean().iloc[-1]
            df_calc.loc[df_calc.index[-1], 'h1_adx'] = 25  # Placeholder, needs full ADX calc
            
            # H1 Trend
            h1_ema_10 = h1_df['close'].ewm(span=10).mean().iloc[-1]
            h1_ema_20 = h1_df['close'].ewm(span=20).mean().iloc[-1]
            h1_close = h1_df['close'].iloc[-1]
            
            if h1_close > h1_ema_10 > h1_ema_20:
                h1_trend = 2
            elif h1_close > h1_ema_20:
                h1_trend = 1
            elif h1_close < h1_ema_10 < h1_ema_20:
                h1_trend = -2
            elif h1_close < h1_ema_20:
                h1_trend = -1
            else:
                h1_trend = 0
            df_calc.loc[df_calc.index[-1], 'h1_trend'] = h1_trend
            
            # Derived H1 features
            m5_rsi = df_calc['rsi_14'].iloc[-1]
            m5_ema_10 = df_calc['close_10_ema'].iloc[-1]
            m5_atr = df_calc['atr'].iloc[-1]
            
            df_calc.loc[df_calc.index[-1], 'h1_rsi_diff'] = m5_rsi - h1_rsi if not np.isnan(h1_rsi) else 0
            df_calc.loc[df_calc.index[-1], 'm5_h1_ema_ratio'] = m5_ema_10 / h1_ema_10 if h1_ema_10 > 0 else 1.0
            df_calc.loc[df_calc.index[-1], 'atr_ratio_h1'] = m5_atr / h1_atr if h1_atr > 0 else 1.0
        else:
            # Fallback values if H1 data unavailable
            df_calc.loc[df_calc.index[-1], 'h1_rsi'] = 50
            df_calc.loc[df_calc.index[-1], 'h1_adx'] = 25
            df_calc.loc[df_calc.index[-1], 'h1_trend'] = 0
            df_calc.loc[df_calc.index[-1], 'h1_rsi_diff'] = 0
            df_calc.loc[df_calc.index[-1], 'm5_h1_ema_ratio'] = 1.0
            df_calc.loc[df_calc.index[-1], 'atr_ratio_h1'] = 1.0
    except Exception as e:
        # Fallback values on error
        df_calc.loc[df_calc.index[-1], 'h1_rsi'] = 50
        df_calc.loc[df_calc.index[-1], 'h1_adx'] = 25
        df_calc.loc[df_calc.index[-1], 'h1_trend'] = 0
        df_calc.loc[df_calc.index[-1], 'h1_rsi_diff'] = 0
        df_calc.loc[df_calc.index[-1], 'm5_h1_ema_ratio'] = 1.0
        df_calc.loc[df_calc.index[-1], 'atr_ratio_h1'] = 1.0
    
    # Feature list matching training (42 features for v3 model)
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
    if 'hour' in df_calc.columns:
        features += ['hour', 'day_of_week', 'london_session', 'ny_session', 'overlap_session']
    
    # Add H1 features (v3 model)
    features += ['h1_rsi', 'h1_adx', 'h1_trend', 'h1_rsi_diff', 'm5_h1_ema_ratio', 'atr_ratio_h1']
    
    # Get last row, fill NaN with 0
    last_row = df_calc.iloc[[-1]][features].fillna(0)
    return last_row, df_calc.iloc[-1]

def close_position(position):
    tick = mt5.symbol_info_tick(position.symbol)
    price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
    type_close = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": type_close,
        "position": position.ticket,
        "price": price,
        "deviation": 20,
        "magic": 999000,
        "comment": "Soldier Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Close failed: {result.retcode}")
    else:
        print(f"Position #{position.ticket} closed")
    return result

def calculate_dynamic_volume(sl_points):
    """
    Calculates dynamic lot size based on risk percentage of equity.
    Uses Numba JIT-compiled function for C-speed calculation.
    """
    RISK_PERCENT = config.get("risk_percent", 0.01)  # Default 1% Risk per trade
    
    account_info = mt5.account_info()
    if account_info is None:
        return VOLUME  # Fallback to fixed
        
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        return VOLUME
    
    # Use JIT-compiled function for fast calculation
    calc_volume = calculate_dynamic_lot(
        equity=float(account_info.equity),
        risk_percent=RISK_PERCENT,
        sl_points=float(sl_points),
        point=float(symbol_info.point),
        tick_value=float(symbol_info.trade_tick_value),
        tick_size=float(symbol_info.trade_tick_size),
        volume_step=float(symbol_info.volume_step),
        volume_min=float(symbol_info.volume_min),
        volume_max=float(symbol_info.volume_max)
    )
    
    # Debug log (first time only)
    if not hasattr(calculate_dynamic_volume, '_logged'):
        risk_amount = account_info.equity * RISK_PERCENT
        print(f"📊 Dynamic Lot: Equity=${account_info.equity:.2f} | Risk {RISK_PERCENT*100}%=${risk_amount:.2f} | Vol={calc_volume:.2f}")
        calculate_dynamic_volume._logged = True
    
    return float(f"{calc_volume:.2f}")

def save_state(bias, rsi, prob, price, lower_band, upper_band, interpretation, positions):
    """
    Saves the current bot state to state.json for the dashboard.
    """
    state_file = os.path.join(PROJECT_ROOT, "state.json")
    
    pos_list = []
    if positions:
        for p in positions:
            pos_list.append({
                "ticket": int(p.ticket),
                "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": float(p.volume),
                "profit": float(p.profit),
                "open_price": float(p.price_open)
            })
            
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": SYMBOL,
        "price": float(price),
        "bias": bias,
        "rsi": float(rsi),
        "prob_up": float(prob),
        "lower_band": float(lower_band),
        "upper_band": float(upper_band),
        "interpretation": interpretation,
        "positions": pos_list
    }
    
    try:
        # Write to temp file then rename to avoid read conflicts
        temp_file = state_file + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(data, f)
        os.replace(temp_file, state_file)
    except Exception as e:
        print(f"Error saving state: {e}")

def interpret_state(bias, rsi, prob, price, lower_band, upper_band, ema):
    """
    Returns a human-readable explanation of the current state.
    """
    explanations = []
    
    # 1. Bias
    if bias == "BULLISH_SCALPING":
        explanations.append("Strategy: BUY ONLY (Looking for dips or breakouts)")
    elif bias == "BEARISH_SCALPING":
        explanations.append("Strategy: SELL ONLY (Looking for peaks or breakdowns)")
    else:
        explanations.append("Strategy: NEUTRAL (Waiting for direction)")
        
    # 2. RSI
    if rsi < 30:
        explanations.append(f"RSI: {rsi:.1f} (Oversold - Price is cheap)")
    elif rsi > 70:
        explanations.append(f"RSI: {rsi:.1f} (Overbought - Price is expensive)")
    else:
        explanations.append(f"RSI: {rsi:.1f} (Neutral)")
        
    # 3. Probability
    if prob > 0.6:
        explanations.append(f"AI Prediction: STRONG UP ({prob:.2f})")
    elif prob > 0.5:
        explanations.append(f"AI Prediction: WEAK UP ({prob:.2f})")
    elif prob < 0.4:
        explanations.append(f"AI Prediction: STRONG DOWN ({prob:.2f})")
    else:
        explanations.append(f"AI Prediction: WEAK DOWN ({prob:.2f})")
        
    # 4. Price Action
    if price < lower_band:
        explanations.append("Price: Below Lower Band (Dip)")
    elif price > upper_band:
        explanations.append("Price: Above Upper Band (Peak)")
    elif price > ema:
        explanations.append("Price: Above EMA (Uptrend)")
    else:
        explanations.append("Price: Below EMA (Downtrend)")
        
    return " | ".join(explanations)

# Global for ATR tracking
last_atr = 0

def execute_trade(action, lot_multiplier=1.0):
    # Check Spread
    tick = mt5.symbol_info_tick(SYMBOL)
    spread = tick.ask - tick.bid
    point = mt5.symbol_info(SYMBOL).point
    spread_points = spread / point
    
    if spread_points > MAX_SPREAD:
        print(f"Spread too high: {spread_points:.1f} > {MAX_SPREAD}. Skipping trade.")
        return None

    price = tick.ask if action == "BUY" else tick.bid
    
    # Use FIXED SL from config (500 points = safe distance)
    sl = price - SL_POINTS * point if action == "BUY" else price + SL_POINTS * point
    # NO TP - Exit on signal change instead (more natural/variable profits)
    
    # Calculate dynamic volume based on SL
    base_volume = calculate_dynamic_volume(SL_POINTS)
    volume = round(base_volume * lot_multiplier, 2)
    
    # Ensure minimum volume
    symbol_info = mt5.symbol_info(SYMBOL)
    if volume < symbol_info.volume_min:
        volume = symbol_info.volume_min
    if volume > 0.05:  # Cap at 0.05 for smaller trades
        volume = 0.05
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "deviation": 10,  # Reducido de 20 -> mejor precisión de entrada
        "magic": 999000, # Soldier Magic Number
        "comment": "Soldier Scalp",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # No TP - will exit on signal change (variable profits)
    
    result = mt5.order_send(request)
    multiplier_info = f" (x{lot_multiplier})" if lot_multiplier < 1.0 else ""
    
    # Check if order_send returned None
    if result is None:
        print(f"❌ Order Failed: {action} | Error: order_send returned None - {mt5.last_error()}")
        return None
    
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"\n{'='*60}")
        print(f"✅ ORDER EXECUTED: {action}")
        print(f"   Entry: {price:.2f} | Vol: {volume}{multiplier_info}")
        print(f"   SL: {sl:.2f} (500pts) | TP: None (exit on signal)")
        print(f"   Ticket: #{result.order}")
        print(f"{'='*60}")
    else:
        print(f"❌ Order Failed: {action} | Code: {result.retcode}")
    
    return result, price, sl

# ============ SHARED STATE ============
class SharedState:
    def __init__(self, model, guardian):
        self.model = model
        self.guardian = guardian
        self.last_trade_time = 0
        self.last_known_tickets = set()
        self.last_known_positions = {}
        self.bot_closed_tickets = set()
        self.running = True

# ============ ASYNC TASK: POSITION MONITOR ============
async def position_monitor(state):
    """Monitors open positions: trailing stop, smart exits, closure detection"""
    print("📊 Position Monitor started")
    
    while state.running:
        try:
            if not mt5.terminal_info():
                await asyncio.sleep(5)
                continue
            
            bias = load_bias()
            positions = mt5.positions_get(symbol=SYMBOL)
            current_tickets = set()
            
            if positions:
                current_tickets = {p.ticket for p in positions}
                
                for p in positions:
                    # Track position details
                    if p.ticket not in state.last_known_positions:
                        state.last_known_positions[p.ticket] = {
                            'type': 'BUY' if p.type == mt5.ORDER_TYPE_BUY else 'SELL',
                            'open_price': p.price_open,
                            'volume': p.volume
                        }
                    
                    tick = mt5.symbol_info_tick(SYMBOL)
                    symbol_info = mt5.symbol_info(SYMBOL)
                    point = symbol_info.point
                    
                    pos_type = 'BUY' if p.type == mt5.ORDER_TYPE_BUY else 'SELL'
                    
                    if pos_type == 'BUY':
                        current_profit_points = (tick.bid - p.price_open) / point
                    else:
                        current_profit_points = (p.price_open - tick.ask) / point
                    
                    # TRAILING STOP
                    if current_profit_points > TRAILING_STOP_START:
                        if pos_type == 'BUY':
                            new_sl = tick.bid - TRAILING_STOP_START * point
                            if new_sl > p.sl + TRAILING_STEP * point:
                                request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "symbol": SYMBOL,
                                    "position": p.ticket,
                                    "sl": new_sl,
                                    "tp": p.tp,
                                }
                                result = mt5.order_send(request)
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    print(f"📈 Trailing SL: #{p.ticket} → {new_sl:.2f}")
                        else:
                            new_sl = tick.ask + TRAILING_STOP_START * point
                            if p.sl == 0 or new_sl < p.sl - TRAILING_STEP * point:
                                request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "symbol": SYMBOL,
                                    "position": p.ticket,
                                    "sl": new_sl,
                                    "tp": p.tp,
                                }
                                result = mt5.order_send(request)
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    print(f"📉 Trailing SL: #{p.ticket} → {new_sl:.2f}")
                    
                    # TRAILING TP (Dynamic Take Profit)
                    if current_profit_points > 50:  # Only start trailing TP after 50 points profit
                        if pos_type == 'BUY':
                            new_tp = tick.bid + TRAILING_TP_DISTANCE * point
                            # Only move TP if it's higher than current TP (or no TP set)
                            if p.tp == 0 or new_tp > p.tp + 50 * point:
                                request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "symbol": SYMBOL,
                                    "position": p.ticket,
                                    "sl": p.sl,
                                    "tp": new_tp,
                                }
                                result = mt5.order_send(request)
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    print(f"🎯 Trailing TP: #{p.ticket} → {new_tp:.2f}")
                        else:  # SELL
                            new_tp = tick.ask - TRAILING_TP_DISTANCE * point
                            # Only move TP if it's lower than current TP (or no TP set)
                            if p.tp == 0 or new_tp < p.tp - 50 * point:
                                request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "symbol": SYMBOL,
                                    "position": p.ticket,
                                    "sl": p.sl,
                                    "tp": new_tp,
                                }
                                result = mt5.order_send(request)
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    print(f"🎯 Trailing TP: #{p.ticket} → {new_tp:.2f}")
                    
                    # SMART EXIT
                    should_close = False
                    close_reason = ""
                    
                    if pos_type == 'BUY':
                        if bias == "BEARISH_SCALPING" and current_profit_points > 50:
                            should_close = True
                            close_reason = f"Take profit +{current_profit_points:.0f}pts"
                        elif current_profit_points < -CUT_LOSS_POINTS:
                            should_close = True
                            close_reason = f"Cut loss {current_profit_points:.0f}pts"
                    elif pos_type == 'SELL':
                        if bias == "BULLISH_SCALPING" and current_profit_points > 50:
                            should_close = True
                            close_reason = f"Take profit +{current_profit_points:.0f}pts"
                        elif current_profit_points < -CUT_LOSS_POINTS:
                            should_close = True
                            close_reason = f"Cut loss {current_profit_points:.0f}pts"
                    
                    if should_close:
                        close_result = close_position(p)
                        if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                            emoji = "💰" if current_profit_points > 0 else "💸"
                            print(f"{emoji} Closed #{p.ticket}: {close_reason}")
                            state.bot_closed_tickets.add(p.ticket)
            
            # Detect external closures
            missing_tickets = state.last_known_tickets - current_tickets
            external_closures = missing_tickets - state.bot_closed_tickets
            
            if external_closures:
                for ticket in external_closures:
                    pos_info = state.last_known_positions.get(ticket, {})
                    deals = mt5.history_deals_get(position=ticket)
                    if deals and len(deals) > 0:
                        close_deal = deals[-1]
                        profit = close_deal.profit
                        emoji = "💰" if profit > 0 else "💸"
                        print(f"{emoji} Position #{ticket} closed: ${profit:.2f}")
                    state.last_known_positions.pop(ticket, None)
            
            state.bot_closed_tickets.clear()
            state.last_known_tickets = current_tickets
            
            await asyncio.sleep(1)  # Check every second
            
        except Exception as e:
            print(f"Position Monitor Error: {e}")
            await asyncio.sleep(5)

# ============ ASYNC TASK: SIGNAL FINDER ============
async def signal_finder(state):
    """Finds trading signals and executes entries"""
    print("🔍 Signal Finder started")
    
    while state.running:
        try:
            # Hot-reload config
            reload_config()
            
            if not mt5.terminal_info():
                await asyncio.sleep(5)
                continue
            
            # Kill Switch Check
            can_trade, reason = state.guardian.can_trade()
            if not can_trade:
                print(f"🛑 KILL SWITCH: {reason}")
                await asyncio.sleep(60)
                continue
            
            # Cooldown Check
            cooldown_remaining = (state.last_trade_time + TRADE_COOLDOWN) - time.time()
            if cooldown_remaining > 0:
                await asyncio.sleep(1)
                continue
            
            # Get Data & Features
            df = get_data()
            if df is None:
                await asyncio.sleep(1)
                continue
            
            X_pred, last_candle = calculate_features(df)
            bias = load_bias()
            prob = state.model.predict(X_pred)[0]
            
            current_price = last_candle['close']
            rsi = last_candle['rsi_14']
            upper_band = last_candle['boll_ub']
            lower_band = last_candle['boll_lb']
            ema = last_candle['close_10_ema']
            
            positions = mt5.positions_get(symbol=SYMBOL)
            num_positions = len(positions) if positions else 0
            
            # Print status
            print(f"Bias: {bias} | Pos: {num_positions} | Price: {current_price:.2f} | RSI: {rsi:.2f} | Prob(UP): {prob:.2f}", end="\r")
            
            # Check trading hours
            if not is_trading_time():
                now = datetime.now()
                print(f"\n⏰ Outside trading hours ({now.strftime('%H:%M')} - Weekend or closed)", end="\r")
                await asyncio.sleep(60)  # Check every minute
                continue
            
            if num_positions >= MAX_POSITIONS:
                await asyncio.sleep(5)
                continue
            
            # Don't open new positions if any current position is in loss
            has_losing_position = False
            if positions:
                for p in positions:
                    if p.profit < -1000:
                        has_losing_position = True
                        break
            
            if has_losing_position:
                print(f"\n⚠️ Position in loss - waiting before opening new trades", end="\r")
                await asyncio.sleep(10)
                continue
            
            # Get higher timeframe trend (H1 + M15)
            htf_trend, trend_info = get_higher_timeframe_trend()
            
            # Get market regime (TRENDING/RANGING/VOLATILE)
            regime, regime_info = detect_market_regime()
            
            # Dynamic entry threshold based on regime
            ENTRY_THRESHOLD = regime_info.get('entry_threshold', 3)
            
            # Entry Logic with Multi-Timeframe Filter
            action = "HOLD"
            
            # Multi-timeframe alignment check
            # BUY only if H1 trend is UP or NEUTRAL
            # SELL only if H1 trend is DOWN or NEUTRAL
            
            if bias == "BULLISH_SCALPING" and htf_trend == "DOWN":
                # H1 is bearish, skip BUY signals on M5
                await asyncio.sleep(5)
                continue
            
            if bias == "BEARISH_SCALPING" and htf_trend == "UP":
                # H1 is bullish, skip SELL signals on M5
                await asyncio.sleep(5)
                continue
            
            if bias == "BULLISH_SCALPING":
                score = 0
                reasons = []
                # Descarte más estricto: probabilidad muy baja = no comprar
                if prob < 0.40:
                    score = -99
                else:
                    if current_price <= lower_band * 1.005:
                        score += 2
                        reasons.append("DIP")
                    if rsi < 40:  # RSI más estricto (era 45)
                        score += 1
                        reasons.append(f"RSI:{rsi:.0f}")
                    if prob > 0.55:  # Umbral más alto (era 0.48)
                        score += 1
                        reasons.append(f"PROB:{prob:.2f}")
                    if prob > 0.62:  # Bonus por probabilidad fuerte
                        score += 1
                        reasons.append("STRONG_PROB")
                    if current_price > ema:
                        score += 1
                        reasons.append("TREND")
                
                if score >= ENTRY_THRESHOLD:
                    print(f"\nSIGNAL: BUY (Score: {score}) | {' + '.join(reasons)}")
                    action = "BUY"
                    
            elif bias == "BEARISH_SCALPING":
                score = 0
                reasons = []
                # Descarte más estricto: probabilidad muy alta = no vender
                if prob > 0.60:
                    score = -99
                else:
                    if current_price >= upper_band * 0.995:
                        score += 2
                        reasons.append("PEAK")
                    if rsi > 60:  # RSI más estricto (era 55)
                        score += 1
                        reasons.append(f"RSI:{rsi:.0f}")
                    if prob < 0.45:  # Umbral más estricto (era 0.52)
                        score += 1
                        reasons.append(f"PROB:{prob:.2f}")
                    if current_price < ema:
                        score += 1
                        reasons.append("TREND")
                    if prob < 0.38:  # Bonus por probabilidad muy baja (era 0.45)
                        score += 1
                        reasons.append("STRONG_PROB")
                
                if score >= ENTRY_THRESHOLD:
                    print(f"\nSIGNAL: SELL (Score: {score}) | {' + '.join(reasons)}")
                    action = "SELL"
            
            # Execute Trade
            if action != "HOLD":
                lot_multiplier = state.guardian.get_lot_multiplier()
                result = execute_trade(action, lot_multiplier)
                if result:
                    res, entry_price, sl = result
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        state.last_trade_time = time.time()
            
            await asyncio.sleep(2)  # Check signals every 2 seconds
            
        except Exception as e:
            print(f"Signal Finder Error: {e}")
            traceback.print_exc()
            await asyncio.sleep(5)

# ============ MAIN ASYNC ENTRY ============
async def main_async():
    print("Soldier Agent Starting (Async Mode)...")
    
    # Initialize MT5
    if not initialize_mt5():
        print("MT5 Init Failed")
        return
    
    # Initialize Symbol
    if not initialize_symbol():
        print("Symbol Init Failed. Exiting.")
        mt5.shutdown()
        return
    
    # Load Model
    model = lgb.Booster(model_file=MODEL_PATH)
    print("LightGBM Model Loaded")
    
    # Initialize Risk Guardian
    guardian = RiskGuardian(config)
    
    # Create Shared State
    state = SharedState(model, guardian)
    
    print("Starting async tasks...")
    
    # Run both tasks concurrently
    await asyncio.gather(
        position_monitor(state),
        signal_finder(state)
    )

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        mt5.shutdown()
        print("\nSoldier Stopped")
