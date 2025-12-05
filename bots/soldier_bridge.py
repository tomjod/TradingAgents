"""
Soldier Bot using Socket Bridge to MQL5
Uses TradingBridge for ultra-low latency execution.

This replaces soldier.py for socket-based trading.
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
from datetime import datetime

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


class BridgeSoldier:
    """
    Trading bot using socket bridge to MQL5 for execution.
    Python handles: AI, ML model, signal generation, risk management
    MQL5 EA handles: Order execution, position monitoring, tick data
    """
    
    def __init__(self):
        self.bridge = TradingBridge()
        self.model = None
        self.guardian = None
        
        self.running = False
        self.last_trade_time = 0
        self.positions = []
        
        # Setup callbacks
        self.bridge.on_tick_callback = self.on_tick
        self.bridge.on_execution_callback = self.on_execution
        self.bridge.on_connect_callback = self.on_connect
        self.bridge.on_disconnect_callback = self.on_disconnect
    
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
        """Load current bias from file"""
        try:
            with open(BIAS_FILE, "r") as f:
                data = json.load(f)
            return data.get("bias", "NEUTRAL")
        except:
            return "NEUTRAL"
    
    def on_connect(self, address):
        """Called when MQL5 EA connects"""
        print(f"🔗 EA Connected from {address}")
        # Request current positions
        self.bridge.get_positions()
    
    def on_disconnect(self):
        """Called when MQL5 EA disconnects"""
        print("❌ EA Disconnected")
    
    def on_tick(self, tick):
        """Called on each tick from MQL5"""
        # This runs frequently, process signals here
        self.process_signal(tick)
    
    def on_execution(self, result):
        """Called when order is executed"""
        if result.get('status') == 'filled':
            self.last_trade_time = time.time()
            print(f"✅ Order filled: {result.get('action')} @ {result.get('price')}")
        else:
            print(f"❌ Order failed: {result.get('error')}")
    
    def calculate_features(self, tick):
        """
        Calculate features from tick data.
        Note: In socket mode, we get tick-by-tick data, not OHLC bars.
        For full feature calculation, we'd need to build bars from ticks.
        
        For now, use a simplified approach based on tick data.
        """
        # In a full implementation, you'd collect ticks and build M5 bars
        # For demo, return simplified features
        return None
    
    def process_signal(self, tick):
        """Process tick and decide on trading action"""
        try:
            # Check cooldown
            if time.time() - self.last_trade_time < TRADE_COOLDOWN:
                return
            
            # Check spread
            spread = tick.get('spread', 999)
            if spread > MAX_SPREAD:
                return
            
            # Get bias
            bias = self.load_bias()
            if bias == "NEUTRAL":
                return
            
            # For full implementation, calculate features and use model
            # For now, use simple price action from tick
            bid = tick.get('bid', 0)
            ask = tick.get('ask', 0)
            
            # Simple logic based on bias
            # In production, use ML model prediction
            
            if bias == "BULLISH_SCALPING":
                # Execute BUY
                print(f"\n📈 Signal: BUY (Bias: {bias}, Spread: {spread})")
                self.bridge.buy(
                    volume=0.01,
                    sl_points=SL_POINTS,
                    comment="BridgeSoldier"
                )
                
            elif bias == "BEARISH_SCALPING":
                # Execute SELL
                print(f"\n📉 Signal: SELL (Bias: {bias}, Spread: {spread})")
                self.bridge.sell(
                    volume=0.01,
                    sl_points=SL_POINTS,
                    comment="BridgeSoldier"
                )
                
        except Exception as e:
            print(f"Signal processing error: {e}")
    
    def start(self):
        """Start the soldier bot"""
        print("="*60)
        print("🤖 Bridge Soldier Starting")
        print("="*60)
        
        # Load model
        if not self.load_model():
            return False
        
        # Initialize risk guardian
        self.guardian = RiskGuardian(config.get('kill_switch', {}))
        
        # Start bridge server
        self.bridge.start()
        self.running = True
        
        print("\n⏳ Waiting for MQL5 EA to connect...")
        print("   Start TradingBridge.mq5 EA in MetaTrader 5")
        print("="*60)
        
        return True
    
    def stop(self):
        """Stop the soldier bot"""
        self.running = False
        self.bridge.stop()
        print("🛑 Bridge Soldier stopped")
    
    def run(self):
        """Main run loop"""
        if not self.start():
            return
        
        try:
            while self.running:
                # Status display every 30 seconds
                if self.bridge.connected:
                    tick = self.bridge.get_tick()
                    bias = self.load_bias()
                    
                    if tick:
                        print(f"\r📊 {bias} | Bid: {tick.get('bid'):.2f} | Spread: {tick.get('spread'):.0f}", end="")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n")
        finally:
            self.stop()


# ========================================
# Main
# ========================================

if __name__ == "__main__":
    soldier = BridgeSoldier()
    soldier.run()
