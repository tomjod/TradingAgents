import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz
from typing import Annotated

def initialize_mt5():
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    return True

def shutdown_mt5():
    mt5.shutdown()

def get_mt5_data(
    symbol: Annotated[str, "ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    if not initialize_mt5():
        return "Error initializing MT5"

    # Ensure symbol is valid (e.g., XAUUSD)
    # Some brokers use suffixes, might need handling but for now assume exact match
    
    # Convert dates
    timezone = pytz.timezone("Etc/UTC")
    utc_from = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone)
    utc_to = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone)
    
    # Get rates
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, utc_from, utc_to)
    
    if rates is None:
        return f"No data found for {symbol} from {start_date} to {end_date}"
        
    # Create DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Format for output
    return f"## Market Data for {symbol} from {start_date} to {end_date}:\n\n" + df.to_string()

def get_mt5_account_info() -> str:
    if not initialize_mt5():
        return "Error initializing MT5"
        
    account_info = mt5.account_info()
    if account_info is None:
        return "Failed to get account info"
        
    df = pd.DataFrame(list(account_info._asdict().items()), columns=['Property', 'Value'])
    return "## MT5 Account Info:\n\n" + df.to_string()

def execute_mt5_order(
    symbol: str,
    action_type: str, # BUY or SELL
    volume: float,
    sl_points: int = 0,
    tp_points: int = 0
) -> str:
    if not initialize_mt5():
        return "Error initializing MT5"
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return f"{symbol} not found"
        
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            return f"symbol_select({symbol}) failed"
            
    point = symbol_info.point
    
    order_type = mt5.ORDER_TYPE_BUY if action_type.upper() == "BUY" else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if action_type.upper() == "BUY" else mt5.symbol_info_tick(symbol).bid
    
    sl = 0.0
    tp = 0.0
    
    if action_type.upper() == "BUY":
        if sl_points > 0: sl = price - sl_points * point
        if tp_points > 0: tp = price + tp_points * point
    else:
        if sl_points > 0: sl = price + sl_points * point
        if tp_points > 0: tp = price - tp_points * point

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "TradingAgents python script",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return f"Order failed: {result.comment} (retcode: {result.retcode})"
        
    return f"Order executed: {result}"

def _get_mt5_data_df(symbol, start_date, end_date):
    if not initialize_mt5():
        return None

    timezone = pytz.timezone("Etc/UTC")
    utc_from = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone)
    utc_to = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone)
    
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, utc_from, utc_to)
    
    if rates is None:
        return None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['date'] = df['time'].dt.strftime('%Y-%m-%d')
    # Rename columns to match what stockstats/yfinance might expect if needed, 
    # but stockstats works with lowercase open, high, low, close.
    # MT5 columns: time, open, high, low, close, tick_volume, spread, real_volume
    # We might need 'volume' column for some indicators
    df['volume'] = df['tick_volume'] 
    return df

def get_mt5_indicators(
    symbol: Annotated[str, "ticker symbol"],
    indicator: Annotated[str, "technical indicator"],
    curr_date: Annotated[str, "current date yyyy-mm-dd"],
    look_back_days: Annotated[int, "look back days"],
) -> str:
    from stockstats import wrap
    from dateutil.relativedelta import relativedelta
    
    # Calculate start date for data fetching (go back enough for indicators)
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date_dt = curr_date_dt - relativedelta(days=look_back_days + 200) # Extra buffer for SMA200 etc
    start_date = start_date_dt.strftime("%Y-%m-%d")
    
    df = _get_mt5_data_df(symbol, start_date, curr_date)
    
    if df is None or df.empty:
        return f"No data found for {symbol}"
        
    # Use stockstats
    stock = wrap(df)
    
    # Calculate indicator
    # stockstats handles many indicators by accessing the column
    # e.g. stock['rsi_14']
    
    # Map common names if necessary, or rely on stockstats parsing
    # The system passes names like 'close_50_sma', 'rsi', 'macd'
    
    try:
        # Accessing the column triggers calculation
        _ = stock[indicator]
    except KeyError:
        # Try to map or just fail
        return f"Indicator {indicator} not supported or calculation failed"
        
    # Filter for the requested window
    before = curr_date_dt - relativedelta(days=look_back_days)
    
    # df has 'date' column from helper
    mask = (df['date'] >= before.strftime("%Y-%m-%d")) & (df['date'] <= curr_date)
    filtered_df = df.loc[mask]
    
    ind_string = ""
    for _, row in filtered_df.iterrows():
        val = row[indicator]
        date_str = row['date']
        ind_string += f"{date_str}: {val}\n"
        
    return f"## {indicator} values for {symbol} from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n{ind_string}"

