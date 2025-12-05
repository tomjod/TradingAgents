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
import lightgbm as lgb
from stockstats import wrap
from datetime import datetime, timedelta

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from bots.bridge_server import TradingBridge
from bots.risk_guardian import RiskGuardian


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
        self.guardian = None
        self.candle_manager = CandleManager(timeframe_minutes=5)
        
        self.running = False
        self.last_trade_time = 0
        self.last_analysis_time = 0
        self.positions = []
        
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
    
    def on_connect(self, address):
        print(f"🔗 EA Connected from {address}")
        # Request history immediately
        print("📥 Requesting 500 M5 candles...")
        self.bridge.get_history(count=500)
        self.bridge.get_positions()
    
    def on_disconnect(self):
        print("❌ EA Disconnected")
    
    def on_tick(self, tick):
        # Update candles
        self.candle_manager.on_tick(tick)
        
        # Check for history / responses
        try:
            while not self.bridge.response_queue.empty():
                resp = self.bridge.response_queue.get_nowait()
                if resp.get('type') == 'history_data':
                    self.candle_manager.load_history(resp.get('data'))
                elif resp.get('type') == 'positions':
                    # Update positions if needed
                    pass
        except:
            pass
        
        # Analyze periodically
        if time.time() - self.last_analysis_time > 1.0:
            self.last_analysis_time = time.time()
            self.process_signal(tick)
    
    def on_execution(self, result):
        if result.get('status') == 'filled':
            self.last_trade_time = time.time()
            print(f"✅ Order filled: {result.get('action')} @ {result.get('price')}")
        else:
            print(f"❌ Order failed: {result.get('error')}")

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

        # 9. H1 Features (Placeholder/Approximation)
        # Ideally fetch H1 data, here we approximate or zerofill
        df_calc['h1_rsi'] = 50
        df_calc['h1_adx'] = 25
        df_calc['h1_trend'] = 0
        df_calc['h1_rsi_diff'] = 0
        df_calc['m5_h1_ema_ratio'] = 1.0
        df_calc['atr_ratio_h1'] = 1.0

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
        return last_row, df_calc.iloc[-1]

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

    def process_signal(self, tick):
        try:
            # Stats
            df = self.candle_manager.get_dataframe()
            if len(df) < 30:
                print(f"\r⏳ Warming up... {len(df)}/30 candles", end="")
                return

            bias = self.load_bias()
            spread = tick.get('spread', 999)
            bid = tick.get('bid', 0)
            
            # Log status immediately (throttled by on_tick timer)
            print(f"\r📊 {bias} | P: {bid:.2f} | Spread: {spread:.0f} | ", end="")

            # Check cooldown
            if time.time() - self.last_trade_time < TRADE_COOLDOWN:
                print(f"Cooldown ({int(TRADE_COOLDOWN - (time.time() - self.last_trade_time))}s)", end="")
                return
            
            # Check spread
            if spread > MAX_SPREAD:
                print(f"Spread > {MAX_SPREAD}", end="")
                return
            
            # Calc features
            X, row = self.calculate_features(df)
            
            # Prediction
            prob = self.model.predict(X)[0]
            
            # Regime
            regime, params = self.detect_regime(df)
            
            # Score
            score, reasons = self.calculate_score(row, prob, bias, params)
            
            # Update log with analysis
            print(f"Prob: {prob:.2f} | Score: {score} | Regime: {regime}", end="")
            
            if bias == "NEUTRAL":
                return
            
            # Signal
            if score >= params['entry_threshold']:
                if bias == "BULLISH_SCALPING":
                    print(f"\n🚀 BUY SIGNAL! Score: {score} | {reasons}")
                    self.bridge.buy(0.01, SL_POINTS, 0, "BridgeSoldier")
                    self.last_trade_time = time.time()
                elif bias == "BEARISH_SCALPING":
                    print(f"\n🚀 SELL SIGNAL! Score: {score} | {reasons}")
                    self.bridge.sell(0.01, SL_POINTS, 0, "BridgeSoldier")
                    self.last_trade_time = time.time()

        except Exception as e:
            print(f"Error in process_signal: {e}")

    def start(self):
        print("="*60)
        print("🤖 Bridge Soldier Starting (V4 Full Features)")
        print("="*60)
        
        if not self.load_model(): return False
        
        self.bridge.start()
        self.running = True
        
        print("\n⏳ Waiting for MQL5 EA to connect...")
        return True
    
    def stop(self):
        self.running = False
        self.bridge.stop()
        print("🛑 Bridge Soldier stopped")
    
    def run(self):
        if not self.start(): return
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

if __name__ == "__main__":
    soldier = BridgeSoldier()
    soldier.run()
