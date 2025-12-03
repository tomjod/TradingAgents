import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz
from typing import Annotated

def initialize_mt5():
    import os
    from dotenv import load_dotenv
    
    # Load env vars if not already loaded
    load_dotenv()
    
    # Check if we have credentials
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
        
    # If credentials are provided, try to login
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

def shutdown_mt5():
    mt5.shutdown()

def get_mt5_data(
    symbol: Annotated[str, "ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    if not initialize_mt5():
        return "Error initializing MT5"

    # Ensure symbol is valid (e.g., XAUUSDm)
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
    
    # Validate action type
    action = action_type.upper()
    if action not in ["BUY", "SELL"]:
        print(f"DEBUG: execute_mt5_order - Ignoring action '{action_type}'")
        return f"No order executed. Action '{action_type}' is not a valid trade action (BUY/SELL)."

    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if action == "BUY" else mt5.symbol_info_tick(symbol).bid
    
    sl = 0.0
    tp = 0.0
    
    if action_type.upper() == "BUY":
        if sl_points > 0: sl = price - sl_points * point
        if tp_points > 0: tp = price + tp_points * point
    else:
        if sl_points > 0: sl = price + sl_points * point
        if tp_points > 0: tp = price - tp_points * point

    print(f"DEBUG: execute_mt5_order - Symbol: {symbol}, Action: {action_type}, Volume: {volume}")
    print(f"DEBUG: SL Points: {sl_points}, TP Points: {tp_points}, Point: {point}, Price: {price}")
    print(f"DEBUG: Calculated SL: {sl}, TP: {tp}")

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
    
    # Add 1 day to utc_to to ensure we include the end_date data
    from datetime import timedelta
    utc_to += timedelta(days=1)
    
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
    try:
        from stockstats import wrap
        from dateutil.relativedelta import relativedelta
        
        # Calculate start date for data fetching (go back enough for indicators)
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_date_dt = curr_date_dt - relativedelta(days=look_back_days + 200) # Extra buffer for SMA200 etc
        start_date = start_date_dt.strftime("%Y-%m-%d")
        
        df = _get_mt5_data_df(symbol, start_date, curr_date)
        
        if df is None or df.empty:
            print(f"DEBUG: No data found for {symbol} in get_mt5_indicators")
            return f"No data found for {symbol}"
            
        # Use stockstats
        try:
            stock = wrap(df)
        except Exception as e:
            print(f"Failed to wrap dataframe: {e}")
            return f"Error calculating indicators: {e}"
        
        # Handle indicator argument which might be a list or a string representation of a list
        indicators_list = []
        if isinstance(indicator, list):
            indicators_list = indicator
        elif isinstance(indicator, str):
            indicator = indicator.strip()
            if indicator.startswith('[') and indicator.endswith(']'):
                try:
                    # Safe parsing of list string
                    import ast
                    indicators_list = ast.literal_eval(indicator)
                except:
                    # Fallback if parsing fails, treat as single string
                    indicators_list = [indicator]
            elif ',' in indicator:
                indicators_list = [i.strip() for i in indicator.split(',')]
            else:
                indicators_list = [indicator]
                
        # Calculate indicators
        results = {}
        for ind in indicators_list:
            try:
                ind_lower = ind.lower()
                # Accessing the column triggers calculation
                _ = stock[ind_lower]
                results[ind] = ind_lower # Map original name to calculated name
            except KeyError:
                results[ind] = None
            except Exception as e:
                print(f"Error calculating {ind}: {e}")
                results[ind] = None
            
        # Filter for the requested window
        # Convert stockstats object back to DataFrame to ensure alignment
        df_calc = pd.DataFrame(stock)
        
        # Filter for the requested window using the calculated dataframe
        before = curr_date_dt - relativedelta(days=look_back_days)
        
        # Ensure 'date' column exists and is used for filtering
        if 'date' not in df_calc.columns:
             # If date is index, reset it
             df_calc = df_calc.reset_index()
             
        mask = (df_calc['date'] >= before.strftime("%Y-%m-%d")) & (df_calc['date'] <= curr_date)
        filtered_df = df_calc.loc[mask]

        # OPTIMIZATION: Only return the last 15 rows to save context window
        if len(filtered_df) > 15:
            filtered_df = filtered_df.tail(15)
        
        ind_string = ""
        # Header
        ind_string += f"Date"
        valid_inds = [ind for ind, calc_name in results.items() if calc_name is not None]
        for ind in valid_inds:
            ind_string += f", {ind}"
        ind_string += "\n"
        
        for _, row in filtered_df.iterrows():
            date_str = row['date']
            ind_string += f"{date_str}"
            for ind in valid_inds:
                calc_name = results[ind]
                val = row[calc_name]
                ind_string += f", {val}"
            ind_string += "\n"
            
        return f"## Indicators values for {symbol} from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n{ind_string}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error in get_mt5_indicators: {e}"

def get_mt5_positions(symbol: str = None) -> str:
    if not initialize_mt5():
        return "Error initializing MT5"
        
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()
        
    if positions is None:
        return f"Failed to get positions, error code: {mt5.last_error()}"
        
    if not positions:
        return "No open positions"
        
    # Convert to DataFrame for readable output
    df = pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
    
    # Select relevant columns
    cols = ['ticket', 'time', 'type', 'magic', 'identifier', 'reason', 'volume', 'price_open', 'sl', 'tp', 'price_current', 'swap', 'profit', 'symbol', 'comment']
    # Filter columns that exist
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    # Map type to readable string (0=BUY, 1=SELL)
    df['type'] = df['type'].map({mt5.ORDER_TYPE_BUY: 'BUY', mt5.ORDER_TYPE_SELL: 'SELL'})
    
    return f"## Open Positions:\n\n{df.to_string()}"

def get_mt5_history(
    date_from: Annotated[str, "Start date yyyy-mm-dd"] = None,
    date_to: Annotated[str, "End date yyyy-mm-dd"] = None,
    group: Annotated[str, "Filter by group (optional)"] = None
) -> str:
    if not initialize_mt5():
        return "Error initializing MT5"
        
    timezone = pytz.timezone("Etc/UTC")
    
    # Default to last 30 days if not specified
    if not date_from:
        date_from_dt = datetime.now(timezone) - pd.Timedelta(days=30)
    else:
        date_from_dt = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone)
        
    if not date_to:
        date_to_dt = datetime.now(timezone) + pd.Timedelta(days=1) # Tomorrow to include today
    else:
        date_to_dt = datetime.strptime(date_to, "%Y-%m-%d").replace(tzinfo=timezone) + pd.Timedelta(days=1)

    if group:
        deals = mt5.history_deals_get(date_from_dt, date_to_dt, group=group)
    else:
        deals = mt5.history_deals_get(date_from_dt, date_to_dt)
        
    if deals is None:
        return f"Failed to get history, error code: {mt5.last_error()}"
        
    if not deals:
        return "No history found"
        
    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    
    # Select relevant columns
    cols = ['ticket', 'order', 'time', 'type', 'entry', 'magic', 'reason', 'volume', 'price', 'commission', 'swap', 'profit', 'symbol', 'comment']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    # Convert time
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Map type (0=BUY, 1=SELL) - Note: Deal types are different from Order types
    # DEAL_TYPE_BUY=0, DEAL_TYPE_SELL=1
    df['type'] = df['type'].map({mt5.DEAL_TYPE_BUY: 'BUY', mt5.DEAL_TYPE_SELL: 'SELL'})
    
    return f"## Trade History ({date_from_dt.strftime('%Y-%m-%d')} to {date_to_dt.strftime('%Y-%m-%d')}):\n\n{df.to_string()}"
