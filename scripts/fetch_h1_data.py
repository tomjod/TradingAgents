"""
Fetch H1 (hourly) training data from MT5
Run this before training the v3 model
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_H1
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "training_data", "XAUUSDm_h1.csv")

def main():
    print("Fetching H1 data from MT5...")
    
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
    
    # Login
    login = int(os.getenv("MT5_LOGIN") or os.getenv("MT5_ACCOUNT"))
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    
    if not mt5.login(login, password=password, server=server):
        print(f"Login failed: {mt5.last_error()}")
        return
    
    print(f"Connected to MT5")
    
    # Get last 6 months of H1 data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start_date, end_date)
    
    if rates is None:
        print(f"Failed to get data: {mt5.last_error()}")
        return
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved {len(df)} H1 candles to {OUTPUT_PATH}")
    
    mt5.shutdown()

if __name__ == "__main__":
    main()
