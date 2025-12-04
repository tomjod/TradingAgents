import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import xgboost as xgb
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
                    print(f"Connected to MT5 account #{login}")
                    return True
                else:
                    print(f"failed to connect at account #{login}, error code: {mt5.last_error()}")
                    mt5.shutdown()
                    return False
            return True
            
        except Exception as e:
            print(f"Failed to login to MT5: {e}")
            time.sleep(2)
            
    return True

def initialize_symbol():
    # Attempt to enable the symbol in Market Watch
    if not mt5.symbol_select(SYMBOL, True):
        print(f"Failed to select {SYMBOL}, trying to find it...")
        
        # Check if it exists but is not visible
        info = mt5.symbol_info(SYMBOL)
        if info is None:
            print(f"Symbol {SYMBOL} not found!")
            # Try to find similar symbols
            symbols = mt5.symbols_get()
            similar = [s.name for s in symbols if "XAU" in s.name or "GOLD" in s.name]
            if similar:
                print(f"Did you mean one of these? {similar}")
            else:
                print("No similar symbols found. Check your broker's symbol list.")
            return False
            
    # Check if data is available
    print(f"Symbol {SYMBOL} selected. Checking data...")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
    if rates is None:
        print(f"Still no data for {SYMBOL}. History might be syncing...")
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
    _ = stock['close_10_ema']  # For Trend Logic
    
    # Convert back to DF to ensure columns exist for custom calc
    df_calc = pd.DataFrame(stock)
    
    # 2. Advanced Features using Numba JIT (C-speed calculations)
    # Extract numpy arrays for JIT functions
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
    
    # Return the last row as a DataFrame for prediction (keeping feature names)
    features = [
        'rsi_14', 'rsi_slope',
        'boll', 'boll_ub', 'boll_lb', 'bb_width', 'dist_ma',
        'macd', 'macds', 'macdh', 'macd_slope',
        'atr', 'cci', 'adx',
        'close', 'volume', 'vol_trend'
    ]
    
    # Get last row
    last_row = df_calc.iloc[[-1]][features]
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
    RISK_PERCENT = 0.01  # 1% Risk per trade
    
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

def execute_trade(action):
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
    
    # Calculate dynamic volume
    volume = calculate_dynamic_volume(SL_POINTS)
    
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
    print(f"Order Sent: {action} | Vol: {volume} | Result: {result.retcode}")
    return result

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

    # Load Model
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    print("XGBoost Model Loaded")
    
    # Track last known positions to detect closures
    last_known_tickets = set()
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
            
            # 1. Read General's Bias
            bias = load_bias()
            
            # 2. Check Open Positions & Manage Them
            positions = mt5.positions_get(symbol=SYMBOL)
            current_tickets = set()
            
            if positions:
                current_tickets = {p.ticket for p in positions}
            
            # Detect closed positions (SL/TP or Manual)
            missing_tickets = last_known_tickets - current_tickets
            
            # Filter out ones we closed ourselves
            external_closures = missing_tickets - bot_closed_tickets
            
            if external_closures:
                print(f"\nPositions closed externally (SL/TP/Manual): {external_closures}")
            
            bot_closed_tickets.clear()
                
            last_known_tickets = current_tickets
            
            # 3. Get Data & Features
            df = get_data()
            if df is None:
                print("Waiting for data (mt5.copy_rates returned None)...", end="\r")
                time.sleep(1)
                continue
                
            X_pred, last_candle = calculate_features(df)
            
            # 4. XGBoost Prediction
            dmatrix = xgb.DMatrix(X_pred)
            prob = model.predict(dmatrix)[0] # Probability of UP
            
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
                res = execute_trade(action)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"\n>>> POSITION OPENED: {action} | Ticket: {res.order}")
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
