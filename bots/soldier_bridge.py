"""
Soldier Bot using Socket Bridge to MQL5
Uses TradingBridge for ultra-low latency execution.
Full V4 Model Implementation with Regime Detection.
"""

import asyncio
import time
import json
import os
import sys
import yaml
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import lightgbm as lgb
from stockstats import wrap
from datetime import datetime, timedelta

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from bots.bridge_server import TradingBridge
from bots.risk_guardian import RiskGuardian
from bots.news_filter import NewsFilter


def load_config():
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Load config
config = load_config()
SYMBOL = config["symbol"]
BIAS_FILE = os.path.join(PROJECT_ROOT, config["bias_file"])
MODEL_PATH = os.path.join(PROJECT_ROOT, config["model_path"])

# Trading params
SL_POINTS = config.get("sl_points", 800)
MAX_SPREAD = config.get("max_spread", 120)
MAX_POSITIONS = config.get("max_positions", 4)
TRADE_COOLDOWN = config.get("trade_cooldown", 300)


class CandleManager:
    """aggregates ticks into M5 candles for feature calc"""
    def __init__(self, timeframe_minutes=5):
        self.tf_min = timeframe_minutes
        self.candles = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        self.current_candle = None
        self.last_clean_time = time.time()
        
        # Pre-load some dummy data or wait for accumulation?
        # Ideally we'd load history from CSV or MT5, for now we accumulation
        pass

    def on_tick(self, tick):
        symbol = tick.get('symbol')
        if symbol != SYMBOL: return

        price = tick.get('bid')
        tick_time_str = tick.get('time') # "2023-10-27 10:00:00"
        try:
            tick_dt = datetime.strptime(tick_time_str, "%Y.%m.%d %H:%M:%S")
        except:
            # Fallback format
             tick_dt = datetime.now()

        # Determine candle start time (floor to 5min)
        minute = tick_dt.minute
        minute_floor = (minute // self.tf_min) * self.tf_min
        candle_start = tick_dt.replace(minute=minute_floor, second=0, microsecond=0)
        
        if self.current_candle is None:
            self.current_candle = {
                'time': candle_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 1
            }
        elif self.current_candle['time'] != candle_start:
            # New candle started, finish previous
            # Append to history
            self.candles.loc[len(self.candles)] = self.current_candle
            if len(self.candles) > 500: # Keep limited history
                self.candles = self.candles.iloc[-500:].reset_index(drop=True)
                
            # Start new
            self.current_candle = {
                'time': candle_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 1
            }
        else:
            # Update current
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price
            self.current_candle['volume'] += 1
            
    def load_history(self, candles_list):
        """Load historical candles from EA"""
        data = []
        for c in candles_list:
            try:
                dt = datetime.strptime(c['time'], "%Y.%m.%d %H:%M")
                data.append({
                    'time': dt,
                    'open': c['open'],
                    'high': c['high'],
                    'low': c['low'],
                    'close': c['close'],
                    'volume': c['tick_vol']
                })
            except Exception as e:
                print(f"Error parsing candle: {e}")
                
        if data:
            self.candles = pd.DataFrame(data).sort_values('time').reset_index(drop=True)
            print(f"✅ Loaded {len(self.candles)} historical candles")

    def get_dataframe(self):
        """Return history + current partial candle"""
        if self.current_candle is None:
            return self.candles.copy()
        
        # Combine
        # If candles is empty, just return current
        if self.candles.empty:
            return pd.DataFrame([self.current_candle])

        df = self.candles.copy()
        
        # Check if current candle time is newer than last history candle
        last_hist_time = df.iloc[-1]['time']
        if self.current_candle['time'] > last_hist_time:
            df.loc[len(df)] = self.current_candle
        elif self.current_candle['time'] == last_hist_time:
             # Update last candle (overwrite)
             df.iloc[-1] = self.current_candle
             
        return df


class BridgeSoldier:
    """
    Trading bot using socket bridge to MQL5 for execution.
    Features: V4 Model, Regime Detection, Dynamic Scoring.
    """
    
    def __init__(self):
        self.bridge = TradingBridge()
        self.model = None
        self.candle_manager = CandleManager(timeframe_minutes=5)
        
        # Initialize MT5 for RiskGuardian (Read-Only usage)
        if not mt5.initialize():
             print("⚠️ MT5 Init failed, RiskGuardian might not work")
        
        # Initialize Risk Guardian
        self.guardian = RiskGuardian(config)
        
        # Initialize News Filter
        self.news_filter = NewsFilter(config)
        
        # Dynamic Lot Sizing Parameters
        self.risk_percent = config.get("risk_percent", 0.01)  # 1% risk per trade
        self.min_lot = 0.01
        self.max_lot = 1.0
        
        self.running = False
        self.last_trade_time = 0
        self.last_analysis_time = 0
        self.positions = []
        self.pending_modifications = set() # Avoid spamming requests
        
        # Setup callbacks
        self.bridge.on_tick_callback = self.on_tick
        self.bridge.on_execution_callback = self.on_execution
        self.bridge.on_connect_callback = self.on_connect
        self.bridge.on_disconnect_callback = self.on_disconnect
        
        # Handle history response (custom callback logic needed in bridge_server or handle via wait)
        # For simplicity, we check response queue or add specific callback logic if we modify bridge_server
        # Let's modify bridge_server to allow custom message handlers? 
        # Actually bridge_server puts history in response_queue. We need to poll it?
        # Better: let's invoke a method when history arrives if implemented.
        # But bridge_server is generic. Let's just create a thread to poll responses or handle in on_tick?
        # Actually bridge_server doesn't have on_message callback. 
        # We can just check the response queue in the main loop or dedicated thread.
        # OR we can monkey-patch _process_message? No. 
        # Let's use a background waiter.
    
    def load_model(self):
        """Load LightGBM model"""
        try:
            self.model = lgb.Booster(model_file=MODEL_PATH)
            print(f"✅ Model loaded: {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def load_bias(self):
        try:
            with open(BIAS_FILE, "r") as f:
                data = json.load(f)
            return data.get("bias", "NEUTRAL")
        except:
            return "NEUTRAL"
    
    def calculate_dynamic_lot(self, sl_points: int) -> float:
        """
        Calculate dynamic lot size based on risk percentage.
        
        Formula:
        Risk Amount = Equity * Risk %
        Lot Size = Risk Amount / (SL Points * Point Value per Lot)
        
        For XAUUSDm: 1 point = $0.01 per lot (0.01 * 100 XAU)
        """
        try:
            # Get current equity from MT5
            account = mt5.account_info()
            if not account:
                print("⚠️ Cannot get account info, using min lot")
                return self.min_lot
            
            equity = account.equity
            
            # Calculate risk amount
            risk_amount = equity * self.risk_percent
            
            # Point value for XAUUSD: Each point = $0.01 per 0.01 lot
            # So for 1.0 lot, each point = $1.00
            # For SL of 500 points with 1.0 lot = $500 risk
            point_value_per_lot = 1.0  # $1 per point per lot for XAUUSD
            
            # Apply lot multiplier from RiskGuardian (reduces lot in drawdown)
            lot_multiplier = self.guardian.get_lot_multiplier()
            
            # Calculate lot size
            if sl_points > 0:
                lot_size = risk_amount / (sl_points * point_value_per_lot)
            else:
                lot_size = self.min_lot
            
            # Apply lot multiplier
            lot_size *= lot_multiplier
            
            # Clamp to min/max
            lot_size = max(self.min_lot, min(self.max_lot, lot_size))
            
            # Round to 2 decimal places
            lot_size = round(lot_size, 2)
            
            print(f"📊 Dynamic Lot: {lot_size} (Risk: ${risk_amount:.2f}, SL: {sl_points}pts, Mult: {lot_multiplier})")
            
            return lot_size
            
        except Exception as e:
            print(f"⚠️ Lot calc error: {e}, using min lot")
            return self.min_lot

    
    def on_connect(self, address):
        print(f"🔗 EA Connected from {address}")
        # Request history immediately
        print("📥 Requesting 500 M5 candles...")
        self.bridge.get_history(count=500)
        self.bridge.get_positions()
    
    def on_disconnect(self):
        print("❌ EA Disconnected")
    
    def on_tick(self, tick):
        """Callback from Bridge Thread - pushes to Async Queue"""
        if hasattr(self, 'loop') and self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.tick_queue.put_nowait, tick)
            
    def manage_positions(self, tick):
        """Manage open positions: Trailing Stop & Take Profit"""
        if not self.positions: return
        
        try:
            bid = tick.get('bid')
            ask = tick.get('ask')
            
            # Point value (Dynamic or Config)
            point = 0.01 if "JPY" in SYMBOL or "XAU" in SYMBOL else 0.00001
            
            # Trailing Config from Global Config
            TS_START = config.get("trailing_stop_start", 350)
            TS_STEP = config.get("trailing_step", 100)
            TP_DIST = config.get("trailing_tp_distance", 500)
            CUT_LOSS = config.get("cut_loss_points", 5000)
            
            bias = self.load_bias() # For Smart Exit
            
            for p in self.positions:
                ticket = p['ticket']
                
                # Skip if pending action
                if ticket in self.pending_modifications:
                    continue
                    
                pos_type = p['type'] # "BUY" or "SELL"
                open_price = p['price']
                sl = p['sl']
                tp = p['tp']
                
                should_close = False
                close_reason = ""
                
                # BUY Logic
                if pos_type == 'BUY':
                    current_profit = (bid - open_price) / point
                    
                    # 0. Safety Enforcement (SL/TP Check)
                    # If SL is 0 or too far (old config), enforce new SL
                    target_sl = bid - SL_POINTS * point # Use BID for SL/TP base? No, usually Open Price for static SL, but let's use current config relative to OPEN
                    # Actually standard is Open Price - SL. But if price moved, maybe we want it relative to Ask/Bid? 
                    # Standard: Fixed SL relative to Open Price.
                    calc_sl = open_price - SL_POINTS * point
                    
                    # If no SL or SL is significantly larger than configured (e.g. > 10% diff), tighten it.
                    # But don't widen it if it's already tighter (trailing).
                    if sl == 0 or (open_price - sl) > (SL_POINTS * point * 1.05):
                         # Only modify if we are not already in profit trailing zone (which handles its own SL)
                         # Simple check: enforce max risk
                         new_sl_enforced = calc_sl
                         if sl == 0 or new_sl_enforced > sl: # Closer to price
                             print(f"🛡️ Enforcing Safety SL #{ticket} to {new_sl_enforced:.2f}")
                             self.bridge.modify_position(ticket, new_sl_enforced, tp)
                             self.pending_modifications.add(ticket)
                             sl = new_sl_enforced
                    
                    # 1. Trailing SL
                    new_sl = bid - TS_START * point
                    if current_profit > TS_START:
                        if new_sl > sl + TS_STEP * point:
                             print(f"📈 Trailing SL #{ticket} to {new_sl:.2f}")
                             self.bridge.modify_position(ticket, new_sl, tp)
                             self.pending_modifications.add(ticket)
                             sl = new_sl # Optimistic update for next logic in same loop
                    
                    # 2. Trailing TP
                    if current_profit > 50:
                        new_tp = bid + TP_DIST * point
                        if tp == 0 or new_tp > tp + 50 * point:
                            print(f"🎯 Trailing TP #{ticket} to {new_tp:.2f}")
                            self.bridge.modify_position(ticket, sl, new_tp)
                            self.pending_modifications.add(ticket)

                    # 3. Smart Exit (Reversal or Cut Loss)
                    if bias == "BEARISH_SCALPING" and current_profit > 50:
                        should_close = True
                        close_reason = f"Reversal (Bearish) +{current_profit:.0f}pts"
                    elif current_profit < -CUT_LOSS:
                        should_close = True
                        close_reason = f"Cut Loss {current_profit:.0f}pts"

                # SELL Logic        
                elif pos_type == 'SELL':
                    current_profit = (open_price - ask) / point
                    
                    # 0. Safety Enforcement
                    calc_sl = open_price + SL_POINTS * point
                    if sl == 0 or (sl - open_price) > (SL_POINTS * point * 1.05):
                         new_sl_enforced = calc_sl
                         if sl == 0 or new_sl_enforced < sl: # Closer to price (lower for Sell)
                             print(f"🛡️ Enforcing Safety SL #{ticket} to {new_sl_enforced:.2f}")
                             self.bridge.modify_position(ticket, new_sl_enforced, tp)
                             self.pending_modifications.add(ticket)
                             sl = new_sl_enforced
                    
                    # 1. Trailing SL
                    new_sl = ask + TS_START * point
                    if current_profit > TS_START:
                        if sl == 0 or new_sl < sl - TS_STEP * point:
                             print(f"📉 Trailing SL #{ticket} to {new_sl:.2f}")
                             self.bridge.modify_position(ticket, new_sl, tp)
                             self.pending_modifications.add(ticket)
                             sl = new_sl
                             
                    # 2. Trailing TP
                    if current_profit > 50:
                        new_tp = ask - TP_DIST * point
                        if tp == 0 or new_tp < tp - 50 * point:
                            print(f"🎯 Trailing TP #{ticket} to {new_tp:.2f}")
                            self.bridge.modify_position(ticket, sl, new_tp)
                            self.pending_modifications.add(ticket)

                    # 3. Smart Exit
                    if bias == "BULLISH_SCALPING" and current_profit > 50:
                        should_close = True
                        close_reason = f"Reversal (Bullish) +{current_profit:.0f}pts"
                    elif current_profit < -CUT_LOSS:
                        should_close = True
                        close_reason = f"Cut Loss {current_profit:.0f}pts"
                
                # Execute Close
                if should_close and ticket not in self.pending_modifications:
                    print(f"💰 Closing #{ticket}: {close_reason}")
                    self.bridge.close_position(ticket)
                    self.pending_modifications.add(ticket)
                    
        except Exception as e:
            print(f"Manage positions error: {e}")
    def on_execution(self, result):
        """Callback from Bridge Thread - Schedule update on Async Loop"""
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._handle_execution_async, result)
            
    def _handle_execution_async(self, result):
        """Handle execution in Async Loop"""
        status = result.get('status')
        action = result.get('action')
        ticket = result.get('ticket')
        
        # Clear pending flag
        if ticket and ticket in self.pending_modifications:
            self.pending_modifications.remove(ticket)
        
        if status == 'filled':
            self.last_trade_time = time.time()
            
            if action == 'MODIFY':
                sl = result.get('sl')
                tp = result.get('tp')
                print(f"✅ Modified #{ticket} | SL: {sl} | TP: {tp}")
                
                # Update local position instantly
                for p in self.positions:
                     if p['ticket'] == ticket:
                         p['sl'] = sl
                         p['tp'] = tp
            
            elif action in ['BUY', 'SELL']:
                price = result.get('price')
                print(f"✅ Order filled: {action} @ {price}")
                self.bridge.get_positions()
                
            elif action in ['CLOSE', 'CLOSE_ALL']:
                print(f"✅ Position Closed: {result.get('ticket', 'ALL')}")
                self.bridge.get_positions()
                
        elif status == 'error':
            ret = result.get('retcode', 'N/A')
            desc = result.get('error', 'Unknown')
            
            # Handle "False Error" codes (10009=DONE, 10008=PLACED)
            if ret in [10009, 10008]:
                 print(f"✅ Order confirmed (RetCode: {ret})")
                 self.bridge.get_positions()
                 return # Treated as success

            print(f"❌ Order failed: {desc} (RetCode: {ret})")
            
            # If No Changes (10025), just update local to match request to stop loop
            if ret == 10025:
                 pass # Wait for next position sync

    def detect_regime(self, df):
        """Detect market regime (reused from soldier.py)"""
        try:
            if len(df) < 50:
                 return 'UNKNOWN', self.get_default_regime_params()

            # ATR ratio
            atr_14 = df['atr'].iloc[-14:].mean()
            atr_avg = df['atr'].mean()
            atr_ratio = atr_14 / atr_avg if atr_avg > 0 else 1.0
            
            # ADX
            adx = df['adx'].iloc[-1]
            
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
        except IndexError:
             return 'UNKNOWN', self.get_default_regime_params()

    def get_default_regime_params(self):
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

    def calculate_features(self, df):
        """Feature engineering pipeline (matches train_lightgbm_v4.py)"""
        stock = wrap(df.copy())
        
        # 1. Basic Indicators
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
        
        df_calc = pd.DataFrame(stock)
        
        # 2. Price Action
        df_calc['rsi_slope'] = df_calc['rsi_14'] - df_calc['rsi_14'].shift(3)
        df_calc['macd_slope'] = df_calc['macd'] - df_calc['macd'].shift(3)
        df_calc['bb_width'] = (df_calc['boll_ub'] - df_calc['boll_lb']) / df_calc['boll']
        df_calc['dist_ma'] = (df_calc['close'] - df_calc['boll']) / df_calc['boll']
        df_calc['vol_trend'] = df_calc['volume'] / df_calc['volume'].rolling(20).mean()
        
        # 3. Momentum
        df_calc['price_change_1'] = df_calc['close'].pct_change(1)
        df_calc['price_change_3'] = df_calc['close'].pct_change(3)
        df_calc['price_change_5'] = df_calc['close'].pct_change(5)
        df_calc['high_low_range'] = (df_calc['high'] - df_calc['low']) / df_calc['close']
        df_calc['close_to_high'] = (df_calc['high'] - df_calc['close']) / (df_calc['high'] - df_calc['low'] + 0.001)
        df_calc['close_to_low'] = (df_calc['close'] - df_calc['low']) / (df_calc['high'] - df_calc['low'] + 0.001)
        
        # 4. Trend Strength
        df_calc['ema_cross'] = (df_calc['close_5_ema'] - df_calc['close_20_ema']) / df_calc['close']
        df_calc['trend_strength'] = (df_calc['close'] - df_calc['close_50_sma']) / df_calc['close_50_sma']
        df_calc['adx_slope'] = df_calc['adx'] - df_calc['adx'].shift(3)
        
        # 5. Volatility
        df_calc['atr_ratio'] = df_calc['atr'] / df_calc['close']
        df_calc['vol_spike'] = df_calc['volume'] / df_calc['volume'].shift(1)
        df_calc['range_expansion'] = df_calc['high_low_range'] / df_calc['high_low_range'].rolling(10).mean()
        df_calc['rsi_price_div'] = df_calc['rsi_14'].diff(5) - (df_calc['close'].pct_change(5) * 100)
        
        # 6. Time Features
        if 'time' in df_calc.columns:
            df_calc['time'] = pd.to_datetime(df_calc['time'])
            df_calc['hour'] = df_calc['time'].dt.hour
            df_calc['day_of_week'] = df_calc['time'].dt.dayofweek
            df_calc['london_session'] = ((df_calc['hour'] >= 8) & (df_calc['hour'] <= 16)).astype(int)
            df_calc['ny_session'] = ((df_calc['hour'] >= 13) & (df_calc['hour'] <= 21)).astype(int)
            df_calc['overlap_session'] = ((df_calc['hour'] >= 13) & (df_calc['hour'] <= 16)).astype(int)
            
            # V4 NEW
            df_calc['peak_hours'] = (
                ((df_calc['hour'] >= 8) & (df_calc['hour'] <= 10)) | 
                ((df_calc['hour'] >= 13) & (df_calc['hour'] <= 16))
            ).astype(int)
            df_calc['off_peak'] = ((df_calc['hour'] >= 21) | (df_calc['hour'] <= 6)).astype(int)
        
        # 7. Regime Features
        df_calc['atr_14'] = df_calc['atr'].rolling(14).mean()
        df_calc['atr_avg'] = df_calc['atr'].rolling(50).mean()
        df_calc['atr_regime_ratio'] = df_calc['atr_14'] / df_calc['atr_avg']
        
        # Regime one-hot
        # Simplified for now, just 0 unless calc'd
        df_calc['regime_trending'] = 0
        df_calc['regime_volatile'] = 0
        
        # 8. Intraday Position
        df_calc['daily_high'] = df_calc['high'].rolling(288).max()
        df_calc['daily_low'] = df_calc['low'].rolling(288).min()
        df_calc['intraday_position'] = (df_calc['close'] - df_calc['daily_low']) / (df_calc['daily_high'] - df_calc['daily_low'] + 0.001)

        # 9. H1 Features (Real Calculation)
        # Resample M5 to H1
        try:
            # Ensure time index
            df_h1_src = stock.copy()
            if 'time' in df_h1_src.columns:
                df_h1_src.set_index('time', inplace=True)
            
            # Resample calculation
            df_h1 = df_h1_src.resample('1h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(df_h1) > 14:
                 h1_stock = wrap(df_h1)
                 _ = h1_stock['rsi_14']
                 _ = h1_stock['adx']
                 _ = h1_stock['close_50_sma']
                 _ = h1_stock['close_200_sma']
                 _ = h1_stock['close_8_ema']
                 _ = h1_stock['close_21_ema']
                 _ = h1_stock['atr']
                 
                 # Get latest H1 values (broadcast to M5 length)
                 # We take the last closed H1 candle (-2) to avoid repainting/noise.
                 # Using -1 (current forming) repaints. -2 is stable.
                 last_h1 = h1_stock.iloc[-2]
                 
                 df_calc['h1_rsi'] = last_h1['rsi_14']
                 df_calc['h1_adx'] = last_h1['adx']
                 
                 # H1 Trend: 1 if EMA8 > EMA21, -1 else
                 trend_val = 1 if last_h1['close_8_ema'] > last_h1['close_21_ema'] else -1
                 df_calc['h1_trend'] = trend_val
                 
                 # RSI diff (H1 - M5)
                 df_calc['h1_rsi_diff'] = last_h1['rsi_14'] - df_calc['rsi_14']
                 
                 # Ratio M5 EMA / H1 EMA
                 df_calc['m5_h1_ema_ratio'] = df_calc['close_5_ema'] / last_h1['close_8_ema'] if last_h1['close_8_ema'] else 1.0
                 
                 # ATR Ratio
                 atr_m5 = df_calc['atr']
                 atr_h1 = last_h1['atr']
                 df_calc['atr_ratio_h1'] = atr_m5 / atr_h1 if atr_h1 > 0 else 1.0
                 
            else:
                 # Fallback if not enough history
                 df_calc['h1_rsi'] = 50
                 df_calc['h1_adx'] = 25
                 df_calc['h1_trend'] = 0
                 df_calc['h1_rsi_diff'] = 0
                 df_calc['m5_h1_ema_ratio'] = 1.0
                 df_calc['atr_ratio_h1'] = 0.2
                 
        except Exception as e:
            print(f"H1 Calc Error: {e}")
            df_calc['h1_rsi'] = 50
            df_calc['h1_adx'] = 25
            df_calc['h1_trend'] = 0
            df_calc['h1_rsi_diff'] = 0
            df_calc['m5_h1_ema_ratio'] = 1.0
            df_calc['atr_ratio_h1'] = 0.2

        # Features List (48 features)
        features = [
            'rsi_14', 'rsi_6', 'rsi_slope', 'boll', 'boll_ub', 'boll_lb', 'bb_width', 'dist_ma',
            'macd', 'macds', 'macdh', 'macd_slope', 'atr', 'atr_ratio', 'cci', 'adx', 'adx_slope',
            'close', 'volume', 'vol_trend', 'vol_spike', 'price_change_1', 'price_change_3', 'price_change_5',
            'high_low_range', 'close_to_high', 'close_to_low', 'ema_cross', 'trend_strength',
            'range_expansion', 'rsi_price_div'
        ]
        
        if 'hour' in df_calc.columns:
            features += ['hour', 'day_of_week', 'london_session', 'ny_session', 'overlap_session', 
                         'peak_hours', 'off_peak']
        
        features += ['atr_regime_ratio', 'regime_trending', 'regime_volatile']
        features += ['intraday_position']
        features += ['h1_rsi', 'h1_adx', 'h1_trend', 'h1_rsi_diff', 'm5_h1_ema_ratio', 'atr_ratio_h1']
        
        last_row = df_calc.iloc[[-1]][features].fillna(0)
        # Return dataframe with indicators for regime usage
        return last_row, df_calc

    def calculate_score(self, row, prob, bias, regime_params):
        """Scoring logic"""
        score = 0
        reasons = []
        
        if bias == "BULLISH_SCALPING":
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
                
        elif bias == "BEARISH_SCALPING":
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


    def is_market_open(self):
        """
        Check if market is open based on XAUUSDm schedule (Image provided).
        Schedule (Server Time):
        - Mon-Thu: 00:00-21:58, 23:00-24:00 (Break 21:58-23:00)
        - Fri: 00:00-21:58
        - Sat: Closed
        - Sun: 23:00-24:00
        
        Assumed Server Time = Local Time + 3 Hours (based on 19:13 Local being closed)
        """
        try:
            # Current Local Time
            now_local = datetime.now()
            
            # Estimated Server Time
            now_server = now_local + timedelta(hours=3)
            
            weekday = now_server.weekday() # 0=Mon, 6=Sun
            hour = now_server.hour
            minute = now_server.minute
            
            # Total minutes from start of day
            total_mins = hour * 60 + minute
            
            # Break Start: 21:58 -> 1318 mins
            # Break End: 23:00 -> 1380 mins
            BREAK_START = 21 * 60 + 58
            BREAK_END = 23 * 60
            
            if weekday >= 0 and weekday <= 3: # Mon-Thu
                if total_mins >= BREAK_START and total_mins < BREAK_END:
                    return False, f"Daily Break ({now_server.strftime('%H:%M')} Server)"
                return True, "Open"
                
            elif weekday == 4: # Fri
                if total_mins >= BREAK_START:
                    return False, "Weekend Close"
                return True, "Open"
                
            elif weekday == 5: # Sat
                return False, "Weekend Close"
                
            elif weekday == 6: # Sun
                if total_mins < BREAK_END:
                    return False, "Weekend Close"
                return True, "Open"
                
            return True, "Open"
            
        except Exception as e:
            print(f"Time check error: {e}")
            return True, "Error" # Default to open on error


    async def monitor_positions_task(self):
        """High-frequency position monitoring (runs on every tick event)"""
        print("⚡ Position Monitor Async Task Started")
        
        # Watchdog
        self.last_tick_recv = time.time()
        MAX_FRAME_SIZE = 500
        
        while self.running:
            try:
                # 0. Watchdog Check (Timeout 10s loop)
                try:
                    tick = await asyncio.wait_for(self.tick_queue.get(), timeout=10.0)
                    self.last_tick_recv = time.time()
                except asyncio.TimeoutError:
                    elapsed = time.time() - self.last_tick_recv
                    if elapsed > 60:
                        # Check if Market is Closed before restarting
                        is_open, reason = self.is_market_open()
                        if not is_open:
                             print(f"\r💤 Market Closed ({reason}). Sleeping... ({int(elapsed)}s)", end="")
                             # Reset watchdog to prevent restart loop immediately when it opens
                             # self.last_tick_recv = time.time() # Optional: Don't reset, just don't restart
                             await asyncio.sleep(10)
                             continue
                        
                        print(f"\n🚨 WATCHDOG ALERT: No ticks for {int(elapsed)}s! Restarting...")
                        os.execv(sys.executable, ['python'] + sys.argv)
                    # print(f"\r⏳ Waiting... ({int(elapsed)}s)", end="")
                    continue
                
                # 1. Update Candles (Fast)
                self.candle_manager.on_tick(tick)
                
                # Limit DataFrame size protection
                if len(self.candle_manager.candles) > MAX_FRAME_SIZE:
                     self.candle_manager.candles = self.candle_manager.candles[-MAX_FRAME_SIZE:]
                
                # 2. Manage Positions (Fast - Trailing Stop, etc)
                self.manage_positions(tick)
                
                # 3. Check Responses (positions, history)
                try:
                    while not self.bridge.response_queue.empty():
                        resp = self.bridge.response_queue.get_nowait()
                        if resp.get('type') == 'history_data':
                            self.candle_manager.load_history(resp.get('data'))
                            self.history_loaded.set()
                        elif resp.get('type') == 'positions':
                            self.positions = resp.get('data', [])
                except Exception as e:
                    print(f"Resp queue error: {e}")
                
                # 4. Trigger Strategy Analysis (Throttled)
                now = time.time()
                if now - self.last_analysis_time >= 0.5: # 500ms throttle for ML
                     # Snapshot data for strategy
                     df_copy = self.candle_manager.get_dataframe()
                     asyncio.create_task(self.run_strategy(tick, df_copy))
                     self.last_analysis_time = now
                     
                self.tick_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Monitor error: {e}")
                await asyncio.sleep(0.1)

    async def run_strategy(self, tick, df):
        """Run ML Strategy (Offloaded to avoid blocking critical path)"""
        try:
            # We run this in current loop but could use executor if calc is very heavy
            # For now direct call is fine as long as we yield
            await self.process_signal_async(tick, df)
        except Exception as e:
            print(f"Strategy error: {e}")

    async def process_signal_async(self, tick, df):
        """Async version of process_signal"""
        try:
            if len(df) < 30:
                print(f"\r⏳ Warming up... {len(df)}/30 candles", end="")
                return

            bias = self.load_bias()
            spread = tick.get('spread', 999)
            bid = tick.get('bid', 0)
            
            # Stats
            status = self.guardian.get_status()
            pnl = status.get('daily_pnl_percent', 0) * 100
            dd = status.get('drawdown_percent', 0) * 100
            
            # Log status immediately
            print(f"\r📊 {bias} | P: {bid:.2f} | PnL: {pnl:+.1f}% | DD: {dd:.1f}% | ", end="")

            # Check cooldown
            if time.time() - self.last_trade_time < TRADE_COOLDOWN:
                print(f"Cooldown ({int(TRADE_COOLDOWN - (time.time() - self.last_trade_time))}s)", end="")
                return
            
            # Risk Guardian Checks
            can_trade, reason = self.guardian.can_trade()
            if not can_trade:
                print(f"\r🛡️ Risk Guardian: {reason}", end="")
                return
                
            check_spread, reason = self.guardian.check_volatility(spread)
            if not check_spread:
                print(f"\r🛡️ High Spread: {spread} > {self.guardian.volatility_spread}", end="")
                return
            
            # Calc features (Sync CPU bound - runs in task)
            X, df_enriched = self.calculate_features(df)
            
            # Prediction
            prob = self.model.predict(X)[0]
            
            # Regime
            regime, params = self.detect_regime(df_enriched)
            
            # Score
            last_row_series = df_enriched.iloc[-1]
            score, reasons = self.calculate_score(last_row_series, prob, bias, params)
            
            # Update log
            print(f"Prob: {prob:.2f} | Score: {score} | Regime: {regime}", end="")
            
            if bias == "NEUTRAL":
                return
            
            # Signal
            if score >= params['entry_threshold']:
                # News Filter Check
                can_trade_news, news_reason = self.news_filter.can_trade()
                if not can_trade_news:
                    print(f"\n📰 News Filter blocked: {news_reason}")
                    return
                
                # Calculate dynamic lot size
                lot_size = self.calculate_dynamic_lot(SL_POINTS)
                
                if bias == "BULLISH_SCALPING":
                    print(f"\n🚀 BUY SIGNAL! Score: {score} | Lot: {lot_size} | {reasons}")
                    self.bridge.buy(lot_size, SL_POINTS, 0, "BridgeSoldier")
                    self.last_trade_time = time.time()
                elif bias == "BEARISH_SCALPING":
                    print(f"\n🚀 SELL SIGNAL! Score: {score} | Lot: {lot_size} | {reasons}")
                    self.bridge.sell(lot_size, SL_POINTS, 0, "BridgeSoldier")
                    self.last_trade_time = time.time()

        except Exception as e:
            print(f"Error in process_signal_async: {e}")

    def stop(self):
        self.running = False
        self.bridge.stop()
        if hasattr(self, 'loop') and self.loop.is_running():
            self.loop.stop()
        print("🛑 Bridge Soldier stopped")

    def start_async(self):
         self.loop = asyncio.new_event_loop()
         asyncio.set_event_loop(self.loop)
         self.tick_queue = asyncio.Queue()
         self.history_loaded = asyncio.Event()
         
         if not self.load_model(): return

         self.bridge.start()
         self.running = True
         
         # Connect & Request
         print("\n⏳ Waiting for MQL5 EA...")
         # Bridge connection happens in background thread
         
         try:
             self.loop.create_task(self.monitor_positions_task())
             self.loop.run_forever()
         except KeyboardInterrupt:
             pass
         finally:
             self.stop()
             
if __name__ == "__main__":
    soldier = BridgeSoldier()
    soldier.start_async()
