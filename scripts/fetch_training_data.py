import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os
import sys
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def initialize_mt5():
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
        
    if login and password and server:
        try:
            authorized = mt5.login(int(login), password=password, server=server)
            if authorized:
                print(f"Connected to MT5 account #{login}")
            else:
                print(f"failed to connect at account #{login}, error code: {mt5.last_error()}")
                return False
        except Exception as e:
            print(f"Failed to login to MT5: {e}")
            return False
    return True

def fetch_data(symbol="XAUUSDm", timeframe=mt5.TIMEFRAME_M5, n_candles=50000):
    if not initialize_mt5():
        return
    
    # Ensure symbol is selected
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
        return

    print(f"Fetching last {n_candles} candles for {symbol}...")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
    
    mt5.shutdown()
    
    if rates is None:
        print("No data found")
        return
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Save to CSV
    DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    output_file = os.path.join(DATA_DIR, "XAUUSDm_m5.csv")
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    print(df.head())
    print(df.tail())

if __name__ == "__main__":
    fetch_data()
