"""
Tactical Agent - Fast BIAS Generator using Technical Analysis Only
Updates bias every 30-60 seconds without LLM calls for rapid market response.

HYBRID MODE: Works alongside general.py (AI agents)
- Reads strategic bias from AI agents
- Only updates when aligned with AI bias OR when AI bias is stale (>10 min)
- Provides faster tactical execution while respecting AI strategy
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
import yaml

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
BIAS_FILE = os.path.join(PROJECT_ROOT, config["bias_file"])

# Tactical Agent Config
TACTICAL_CONFIG = config.get("tactical_agent", {})
INTERVAL_SECONDS = TACTICAL_CONFIG.get("interval_seconds", 30)
STRONG_SIGNAL_THRESHOLD = TACTICAL_CONFIG.get("strong_signal_threshold", 5)
USE_MTF = TACTICAL_CONFIG.get("use_mtf", True)
RESPECT_AI_BIAS = TACTICAL_CONFIG.get("respect_ai_bias", True)  # Hybrid mode
AI_BIAS_STALE_MINUTES = TACTICAL_CONFIG.get("ai_bias_stale_minutes", 10)  # Override AI if stale

# Separate files for AI and tactical output
AI_BIAS_FILE = os.path.join(PROJECT_ROOT, "ai_bias.json")  # Read AI bias from here


def get_ai_strategic_bias():
    """
    Read the current bias from AI agents (general.py).
    Reads from ai_bias.json (separate from tactical output).
    Returns: (bias, source, age_minutes, reasoning)
    """
    try:
        # Read from AI-specific file (written by general.py)
        if os.path.exists(AI_BIAS_FILE):
            with open(AI_BIAS_FILE, "r") as f:
                data = json.load(f)
            
            bias = data.get("bias", "NEUTRAL")
            source = data.get("source", "unknown")
            last_update = data.get("last_update", "")
            reasoning = data.get("reasoning", "")
            
            # Calculate age in minutes
            if last_update:
                try:
                    update_time = datetime.strptime(last_update, "%Y-%m-%d %H:%M:%S")
                    age_minutes = (datetime.now() - update_time).total_seconds() / 60
                except:
                    age_minutes = 999  # If parsing fails, consider stale
            else:
                age_minutes = 999
            
            return bias, source, age_minutes, reasoning
    except Exception as e:
        print(f"⚠️ Error reading AI bias: {e}")
    
    return "NEUTRAL", "none", 999, ""


def is_aligned(tactical_bias, ai_bias):
    """
    Check if tactical bias aligns with AI strategic bias.
    Returns: (aligned, reason)
    """
    if ai_bias == "NEUTRAL":
        return True, "AI neutral - tactical can lead"
    
    if tactical_bias == "NEUTRAL":
        return True, "Tactical neutral - no conflict"
    
    if tactical_bias == ai_bias:
        return True, "Fully aligned"
    
    # Conflicting biases
    return False, f"CONFLICT: Tactical={tactical_bias} vs AI={ai_bias}"


def initialize_mt5():
    """Initialize MT5 connection"""
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    
    if not mt5.initialize():
        print(f"❌ MT5 initialize() failed: {mt5.last_error()}")
        return False
    
    if login and password and server:
        authorized = mt5.login(int(login), password=password, server=server)
        if not authorized:
            print(f"❌ MT5 login failed: {mt5.last_error()}")
            return False
        print(f"✅ Connected to MT5 account #{login}")
    
    return True


def get_timeframe_data(timeframe, count=100):
    """Get OHLCV data for a specific timeframe"""
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, count)
    if rates is None:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def calculate_rsi(df, period=14):
    """Calculate RSI"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_ema(df, period):
    """Calculate EMA"""
    return df['close'].ewm(span=period, adjust=False).mean()


def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger(df, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return sma, upper, lower


def calculate_atr(df, period=14):
    """Calculate ATR"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def analyze_timeframe(df):
    """
    Analyze a single timeframe and return a score + reasoning.
    Returns: (score, reasons) where score is -10 to +10
    """
    if df is None or len(df) < 50:
        return 0, ["No data"]
    
    score = 0
    reasons = []
    
    # Calculate indicators
    rsi = calculate_rsi(df).iloc[-1]
    ema_10 = calculate_ema(df, 10).iloc[-1]
    ema_20 = calculate_ema(df, 20).iloc[-1]
    ema_50 = calculate_ema(df, 50).iloc[-1]
    macd, signal, hist = calculate_macd(df)
    macd_val, signal_val, hist_val = macd.iloc[-1], signal.iloc[-1], hist.iloc[-1]
    sma, upper, lower = calculate_bollinger(df)
    close = df['close'].iloc[-1]
    atr = calculate_atr(df).iloc[-1]
    
    # 1. RSI Analysis
    if rsi < 30:
        score += 2
        reasons.append(f"RSI oversold ({rsi:.0f})")
    elif rsi < 40:
        score += 1
        reasons.append(f"RSI low ({rsi:.0f})")
    elif rsi > 70:
        score -= 2
        reasons.append(f"RSI overbought ({rsi:.0f})")
    elif rsi > 60:
        score -= 1
        reasons.append(f"RSI high ({rsi:.0f})")
    
    # 2. EMA Alignment
    if close > ema_10 > ema_20 > ema_50:
        score += 3
        reasons.append("Strong uptrend (EMA aligned)")
    elif close > ema_10 > ema_20:
        score += 2
        reasons.append("Uptrend (EMA 10>20)")
    elif close > ema_20:
        score += 1
        reasons.append("Above EMA20")
    elif close < ema_10 < ema_20 < ema_50:
        score -= 3
        reasons.append("Strong downtrend (EMA aligned)")
    elif close < ema_10 < ema_20:
        score -= 2
        reasons.append("Downtrend (EMA 10<20)")
    elif close < ema_20:
        score -= 1
        reasons.append("Below EMA20")
    
    # 3. MACD Analysis
    if macd_val > signal_val and hist_val > 0:
        if hist_val > hist.iloc[-2]:  # Histogram increasing
            score += 2
            reasons.append("MACD bullish + momentum")
        else:
            score += 1
            reasons.append("MACD bullish")
    elif macd_val < signal_val and hist_val < 0:
        if hist_val < hist.iloc[-2]:  # Histogram decreasing
            score -= 2
            reasons.append("MACD bearish + momentum")
        else:
            score -= 1
            reasons.append("MACD bearish")
    
    # 4. Bollinger Band Position
    bb_position = (close - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
    if bb_position < 0.1:
        score += 2
        reasons.append("Near BB lower (oversold)")
    elif bb_position < 0.3:
        score += 1
        reasons.append("Low in BB range")
    elif bb_position > 0.9:
        score -= 2
        reasons.append("Near BB upper (overbought)")
    elif bb_position > 0.7:
        score -= 1
        reasons.append("High in BB range")
    
    # 5. Price Momentum (last 3 candles)
    price_change = (close - df['close'].iloc[-4]) / df['close'].iloc[-4] * 100
    if price_change > 0.3:
        score += 1
        reasons.append(f"Momentum up ({price_change:.2f}%)")
    elif price_change < -0.3:
        score -= 1
        reasons.append(f"Momentum down ({price_change:.2f}%)")
    
    return score, reasons


def get_tactical_bias():
    """
    Analyze multiple timeframes and generate a tactical bias.
    Uses M5, M15, H1 timeframes with different weights.
    
    Returns: (bias, confidence, reasoning)
    """
    # Get data for each timeframe
    m5_df = get_timeframe_data(mt5.TIMEFRAME_M5, 100)
    m15_df = get_timeframe_data(mt5.TIMEFRAME_M15, 100)
    h1_df = get_timeframe_data(mt5.TIMEFRAME_H1, 50)
    
    # Analyze each timeframe
    m5_score, m5_reasons = analyze_timeframe(m5_df)
    m15_score, m15_reasons = analyze_timeframe(m15_df)
    h1_score, h1_reasons = analyze_timeframe(h1_df)
    
    # Weighted score (H1 has more weight for trend direction)
    # M5: 30%, M15: 30%, H1: 40%
    total_score = (m5_score * 0.3) + (m15_score * 0.3) + (h1_score * 0.4)
    
    # Determine bias based on score
    if total_score >= STRONG_SIGNAL_THRESHOLD:
        bias = "BULLISH_SCALPING"
        confidence = min(total_score / 10, 1.0)
    elif total_score <= -STRONG_SIGNAL_THRESHOLD:
        bias = "BEARISH_SCALPING"
        confidence = min(abs(total_score) / 10, 1.0)
    else:
        bias = "NEUTRAL"
        confidence = 0.5
    
    # Build reasoning
    reasoning = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_score": round(total_score, 2),
        "timeframes": {
            "M5": {"score": m5_score, "signals": m5_reasons[:3]},
            "M15": {"score": m15_score, "signals": m15_reasons[:3]},
            "H1": {"score": h1_score, "signals": h1_reasons[:3]}
        },
        "confidence": round(confidence, 2)
    }
    
    return bias, confidence, reasoning


def update_bias(bias, reasoning):
    """Save bias to file"""
    data = {
        "bias": bias,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "tactical_agent",
        "reasoning": reasoning
    }
    
    try:
        temp_file = BIAS_FILE + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_file, BIAS_FILE)
    except Exception as e:
        print(f"❌ Error saving bias: {e}")


def print_status(bias, score, reasoning):
    """Print formatted status"""
    emoji = "🟢" if bias == "BULLISH_SCALPING" else "🔴" if bias == "BEARISH_SCALPING" else "⚪"
    
    m5 = reasoning["timeframes"]["M5"]
    m15 = reasoning["timeframes"]["M15"]
    h1 = reasoning["timeframes"]["H1"]
    
    print(f"\n{'='*60}")
    print(f"⚡ TACTICAL BIAS: {emoji} {bias}")
    print(f"{'='*60}")
    print(f"   Score: {score:.1f} | Confidence: {reasoning['confidence']*100:.0f}%")
    print(f"   M5:  {m5['score']:+d} | {', '.join(m5['signals'][:2])}")
    print(f"   M15: {m15['score']:+d} | {', '.join(m15['signals'][:2])}")
    print(f"   H1:  {h1['score']:+d} | {', '.join(h1['signals'][:2])}")
    print(f"{'='*60}")


def main():
    print("⚡ Tactical Agent Starting (HYBRID MODE)...")
    print(f"   Symbol: {SYMBOL}")
    print(f"   Interval: {INTERVAL_SECONDS}s")
    print(f"   Signal Threshold: {STRONG_SIGNAL_THRESHOLD}")
    print(f"   Respect AI Bias: {RESPECT_AI_BIAS}")
    print(f"   AI Stale After: {AI_BIAS_STALE_MINUTES} min")
    
    if not initialize_mt5():
        print("❌ Failed to initialize MT5")
        return
    
    # Verify symbol
    if not mt5.symbol_select(SYMBOL, True):
        print(f"❌ Failed to select symbol {SYMBOL}")
        mt5.shutdown()
        return
    
    print(f"✅ Symbol {SYMBOL} ready")
    print(f"🔄 Starting tactical analysis loop...\n")
    
    last_written_bias = None
    iteration = 0
    
    while True:
        try:
            iteration += 1
            
            # 1. Get AI strategic bias (from general.py)
            ai_bias, ai_source, ai_age_min, ai_reasoning = get_ai_strategic_bias()
            
            # 2. Get tactical technical bias
            tactical_bias, confidence, reasoning = get_tactical_bias()
            score = reasoning["total_score"]
            
            # 3. Determine if we should update bias
            should_update = False
            update_reason = ""
            final_bias = tactical_bias
            
            if not RESPECT_AI_BIAS:
                # Pure tactical mode - always update
                should_update = True
                update_reason = "Pure tactical mode"
            elif ai_age_min > AI_BIAS_STALE_MINUTES:
                # AI bias is stale - tactical takes over
                should_update = True
                update_reason = f"AI stale ({ai_age_min:.0f}m > {AI_BIAS_STALE_MINUTES}m)"
            elif ai_source == "none":
                # No AI bias exists - tactical leads
                should_update = True
                update_reason = "No AI bias - tactical leads"
            else:
                # Check alignment
                aligned, align_reason = is_aligned(tactical_bias, ai_bias)
                if aligned:
                    should_update = True
                    update_reason = f"Aligned: {align_reason}"
                else:
                    # NOT aligned - respect AI, don't update
                    should_update = False
                    update_reason = align_reason
                    final_bias = ai_bias  # Keep AI bias
            
            # 4. Status display
            emoji_tactical = "🟢" if tactical_bias == "BULLISH_SCALPING" else "🔴" if tactical_bias == "BEARISH_SCALPING" else "⚪"
            emoji_ai = "🟢" if ai_bias == "BULLISH_SCALPING" else "🔴" if ai_bias == "BEARISH_SCALPING" else "⚪"
            
            if should_update and final_bias != last_written_bias:
                # Bias changing - print full status
                print_status(final_bias, score, reasoning)
                print(f"   🤖 AI Bias: {emoji_ai} {ai_bias} (age: {ai_age_min:.0f}m, src: {ai_source})")
                print(f"   ⚡ Tactical: {emoji_tactical} {tactical_bias} | {update_reason}")
                print(f"   📝 WRITING BIAS: {final_bias}")
                last_written_bias = final_bias
                update_bias(final_bias, reasoning)
            elif iteration % 10 == 0:
                # Every 10 iterations, show status
                print(f"\n⚡ {emoji_tactical} Tact:{tactical_bias} | 🤖 {emoji_ai} AI:{ai_bias} ({ai_age_min:.0f}m) | Score:{score:+.1f}")
                if not should_update:
                    print(f"   ⏸️  NOT updating: {update_reason}")
            else:
                # Compact one-liner
                status = "✅" if should_update else "⏸️"
                print(f"{status} Tact:{tactical_bias} | AI:{ai_bias} ({ai_age_min:.0f}m) | Score:{score:+.1f}", end="\r")
            
            # 5. Only write if should update
            if should_update:
                update_bias(final_bias, reasoning)
            
            # Wait for next iteration
            time.sleep(INTERVAL_SECONDS)
            
        except KeyboardInterrupt:
            print("\n\n⛔ Tactical Agent stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)
    
    mt5.shutdown()
    print("👋 Tactical Agent shutdown complete")


if __name__ == "__main__":
    main()
