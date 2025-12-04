import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from stockstats import wrap
import time
import json
import os
import sys
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
VOLUME = config["volume"]
SL_POINTS = config["sl_points"]
TP_POINTS = config["tp_points"]
MAX_SPREAD = config["max_spread"]
TRAILING_STOP_START = config["trailing_stop_start"]
TRAILING_STEP = config["trailing_step"]
MAX_POSITIONS = config["max_positions"]

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
    
    # Feature list matching training (36 features)
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
    
    sl = price - SL_POINTS * point if action == "BUY" else price + SL_POINTS * point
    tp = price + TP_POINTS * point if action == "BUY" else price - TP_POINTS * point
    
    # Calculate dynamic volume with lot multiplier (drawdown protection)
    base_volume = calculate_dynamic_volume(SL_POINTS)
    volume = round(base_volume * lot_multiplier, 2)
    
    # Ensure minimum volume
    symbol_info = mt5.symbol_info(SYMBOL)
    if volume < symbol_info.volume_min:
        volume = symbol_info.volume_min
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 999000, # Soldier Magic Number
        "comment": "Soldier Scalp",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    multiplier_info = f" (x{lot_multiplier})" if lot_multiplier < 1.0 else ""
    
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"\n{'='*60}")
        print(f"✅ ORDER EXECUTED: {action}")
        print(f"   Entry: {price:.2f} | Vol: {volume}{multiplier_info}")
        print(f"   SL: {sl:.2f} | TP: {tp:.2f}")
        print(f"   Ticket: #{result.order}")
        print(f"{'='*60}")
    else:
        print(f"❌ Order Failed: {action} | Code: {result.retcode}")
    
    return result, price, sl, tp

def main():
    print("Soldier Agent Starting...")
    
    # Initialize MT5
    if not initialize_mt5():
        print("MT5 Init Failed")
        return
        
    # Initialize Symbol
    if not initialize_symbol():
        print("Symbol Init Failed. Exiting.")
        mt5.shutdown()
        return

    # Load Model (LightGBM)
    model = lgb.Booster(model_file=MODEL_PATH)
    print("LightGBM Model Loaded")
    
    # Initialize Risk Guardian (Kill Switch)
    guardian = RiskGuardian(config)
    
    # Track last known positions to detect closures
    last_known_tickets = set()
    last_known_positions = {}  # Store position details for closure logging
    bot_closed_tickets = set()

    print("Entering main loop...")
    while True:
        try:
            # Check connection
            if not mt5.terminal_info():
                print("Connection lost, reconnecting...")
                if not initialize_mt5():
                    time.sleep(5)
                    continue
            
            # === KILL SWITCH CHECK ===
            can_trade, reason = guardian.can_trade()
            if not can_trade:
                print(f"🛑 KILL SWITCH: {reason}")
                time.sleep(60)  # Check again in 1 minute
                continue
            
            # 1. Read General's Bias
            bias = load_bias()
            
            # 2. Check Open Positions & Manage Them
            positions = mt5.positions_get(symbol=SYMBOL)
            current_tickets = set()
            
            if positions:
                current_tickets = {p.ticket for p in positions}
                # Track position details for closure logging
                for p in positions:
                    if p.ticket not in last_known_positions:
                        last_known_positions[p.ticket] = {
                            'type': 'BUY' if p.type == mt5.ORDER_TYPE_BUY else 'SELL',
                            'open_price': p.price_open,
                            'volume': p.volume,
                            'sl': p.sl,
                            'tp': p.tp
                        }
            
            # Detect closed positions (SL/TP or Manual)
            missing_tickets = last_known_tickets - current_tickets
            
            # Filter out ones we closed ourselves
            external_closures = missing_tickets - bot_closed_tickets
            
            # Log closed positions with details
            if external_closures:
                for ticket in external_closures:
                    pos_info = last_known_positions.get(ticket, {})
                    pos_type = pos_info.get('type', '?')
                    open_price = pos_info.get('open_price', 0)
                    
                    # Get deal history to find profit
                    deals = mt5.history_deals_get(position=ticket)
                    if deals and len(deals) > 0:
                        close_deal = deals[-1]
                        profit = close_deal.profit
                        close_price = close_deal.price
                        
                        emoji = "💰" if profit > 0 else "💸"
                        result = "PROFIT" if profit > 0 else "LOSS"
                        
                        print(f"\n{'='*60}")
                        print(f"{emoji} POSITION CLOSED: {pos_type} #{ticket}")
                        print(f"   Open: {open_price:.2f} → Close: {close_price:.2f}")
                        print(f"   Result: {result} ${profit:.2f}")
                        print(f"{'='*60}")
                    else:
                        print(f"\n📍 Position #{ticket} closed (details unavailable)")
                    
                    # Clean up tracking
                    last_known_positions.pop(ticket, None)
            
            bot_closed_tickets.clear()
                
            last_known_tickets = current_tickets
            
            # 3. Get Data & Features
            df = get_data()
            if df is None:
                print("Waiting for data (mt5.copy_rates returned None)...", end="\r")
                time.sleep(1)
                continue
                
            X_pred, last_candle = calculate_features(df)
            
            # 4. LightGBM Prediction
            prob = model.predict(X_pred)[0]  # Probability of UP
            
            # 5. Logic Variables
            current_price = last_candle['close']
            rsi = last_candle['rsi_14']
            lower_band = last_candle['boll_lb']
            upper_band = last_candle['boll_ub']
            ema = last_candle['close_10_ema']
            
            # Detailed Logging
            pos_count = len(positions) if positions else 0
            print(f"Bias: {bias} | Pos: {pos_count} | Price: {current_price:.2f} | RSI: {rsi:.2f} | BB_L: {lower_band:.2f} | BB_U: {upper_band:.2f} | Prob(UP): {prob:.2f}")
            
            # Print Interpretation
            explanation = interpret_state(bias, rsi, prob, current_price, lower_band, upper_band, ema)
            print(f"--> {explanation}", end="\r", flush=True)
            
            # Save State for Dashboard
            save_state(bias, rsi, prob, current_price, lower_band, upper_band, explanation, positions)

            # --- MANAGE OPEN POSITIONS (Exit Logic) ---
            if positions:
                for pos in positions:
                    profit = pos.profit
                    pos_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                    
                    # Apply Trailing Stop
                    check_trailing_stop(pos)
                    
                    should_close = False
                    reason = ""
                    
                    if pos_type == "BUY":
                        if current_price >= upper_band:
                            should_close = True
                            reason = "Target (Upper Band) Hit"
                        elif rsi > 75:
                            should_close = True
                            reason = "RSI Overbought"
                        elif prob < 0.4:
                            should_close = True
                            reason = "Model Reversal"
                            
                    elif pos_type == "SELL":
                        if current_price <= lower_band:
                            should_close = True
                            reason = "Target (Lower Band) Hit"
                        elif rsi < 25:
                            should_close = True
                            reason = "RSI Oversold"
                        elif prob > 0.6:
                            should_close = True
                            reason = "Model Reversal"
                    
                    if should_close:
                        outcome = "PROFIT" if profit > 0 else "LOSS"
                        symbol_str = "+++" if profit > 0 else "---"
                        print(f"\n{symbol_str} CLOSING {pos_type} ({outcome}) | Profit: {profit:.2f} | Reason: {reason}")
                        res = close_position(pos)
                        if res.retcode == mt5.TRADE_RETCODE_DONE:
                            bot_closed_tickets.add(pos.ticket)
                        time.sleep(1)
                
                # Stack Logic: If we have positions but less than MAX, we can still enter
                if len(positions) >= MAX_POSITIONS:
                    time.sleep(0.5)
                    continue 

            # --- ENTRY LOGIC (Score-Based System) ---
            action = "HOLD"
            ema = last_candle['close_10_ema']
            macd_slope = last_candle.get('rsi_slope', 0)  # Using RSI slope as momentum proxy
            
            ENTRY_THRESHOLD = 3  # Minimum score to enter (out of max ~5-6)
            
            if bias == "BULLISH_SCALPING":
                score = 0
                reasons = []
                
                # MODEL VETO: If model strongly disagrees, don't enter
                if prob < 0.35:
                    score = -99  # Veto any entry
                    reasons.append("MODEL_VETO")
                else:
                    # Score conditions for BUY
                    if current_price <= lower_band * 1.005:  # Near lower band (dip)
                        score += 2
                        reasons.append("DIP")
                    if rsi < 45:  # RSI not overbought
                        score += 1
                        reasons.append(f"RSI:{rsi:.0f}")
                    if prob > 0.48:  # Model slightly bullish
                        score += 1
                        reasons.append(f"PROB:{prob:.2f}")
                    if current_price > ema:  # Above trend
                        score += 1
                        reasons.append("TREND")
                    if prob > 0.55:  # Strong model signal (bonus)
                        score += 1
                        reasons.append("STRONG_PROB")
                    
                if score >= ENTRY_THRESHOLD:
                    print(f"\nSIGNAL: BUY (Score: {score}) | {' + '.join(reasons)} | Price: {current_price:.2f}")
                    action = "BUY"
                    
            elif bias == "BEARISH_SCALPING":
                score = 0
                reasons = []
                
                # MODEL VETO: If model strongly disagrees, don't enter
                if prob > 0.65:
                    score = -99  # Veto any entry
                    reasons.append("MODEL_VETO")
                else:
                    # Score conditions for SELL
                    if current_price >= upper_band * 0.995:  # Near upper band (peak)
                        score += 2
                        reasons.append("PEAK")
                    if rsi > 55:  # RSI not oversold
                        score += 1
                        reasons.append(f"RSI:{rsi:.0f}")
                    if prob < 0.52:  # Model slightly bearish
                        score += 1
                        reasons.append(f"PROB:{prob:.2f}")
                    if current_price < ema:  # Below trend
                        score += 1
                        reasons.append("TREND")
                    if prob < 0.45:  # Strong model signal (bonus)
                        score += 1
                        reasons.append("STRONG_PROB")
                    
                if score >= ENTRY_THRESHOLD:
                    print(f"\nSIGNAL: SELL (Score: {score}) | {' + '.join(reasons)} | Price: {current_price:.2f}")
                    action = "SELL"
            
            # 6. Execute Entry
            if action != "HOLD":
                # Get lot multiplier from Risk Guardian (drawdown protection)
                lot_multiplier = guardian.get_lot_multiplier()
                result = execute_trade(action, lot_multiplier)
                if result:
                    res, entry_price, sl, tp = result
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        pass  # Logging already done in execute_trade
                time.sleep(60) # Wait 1 min after trade to avoid double entry
            
            time.sleep(0.2) # Tick loop
            
        except Exception as e:
            print(f"\nError in loop: {e}")
            traceback.print_exc()
            # Try to reconnect on error
            initialize_mt5()
            time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        mt5.shutdown()
        print("\nSoldier Stopped")
