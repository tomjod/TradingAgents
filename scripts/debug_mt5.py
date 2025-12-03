    
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
        
    if login and password and server:
        authorized = mt5.login(int(login), password=password, server=server)
        if not authorized:
            print(f"failed to connect at account #{login}, error code: {mt5.last_error()}")
            return False
    return True

def debug_data():
    if not initialize_mt5():
        return

    symbol = "XAUUSDm"
    timezone = pytz.timezone("Etc/UTC")
    
    # Test 1: Data for today
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"--- Debugging Data for {today} ---")
    
    utc_from = datetime.strptime(today, "%Y-%m-%d").replace(tzinfo=timezone)
    # Current logic uses same date for to
    utc_to = datetime.strptime(today, "%Y-%m-%d").replace(tzinfo=timezone)
    
    print(f"Querying from {utc_from} to {utc_to}")
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, utc_from, utc_to)
    print(f"Rates (same day): {rates}")
    
    # Test 2: Data for today + 1 day for 'to'
    utc_to_plus = utc_to + timedelta(days=1)
    
    # Fetch longer range like mt5.py
    utc_from_long = utc_to - timedelta(days=230)
    print(f"Querying long range from {utc_from_long} to {utc_to_plus}")
    rates_plus = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, utc_from_long, utc_to_plus)
    print(f"Rates count: {len(rates_plus) if rates_plus is not None else 0}")
    
    if rates_plus is not None:
        df = pd.DataFrame(rates_plus)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['date'] = df['time'].dt.strftime('%Y-%m-%d')
        df['volume'] = df['tick_volume']
        print("\nDataFrame Columns:", df.columns)
        print("DataFrame Head:\n", df.head())
        
        # Test 3: Stockstats
        print("\n--- Debugging Stockstats ---")
        stock = wrap(df)
        try:
            print("Calculating RSI...")
            print(stock['rsi'])
            print("Calculating VWMA...")
            print(stock['vwma'])
        except Exception as e:
            print(f"Stockstats error: {e}")

    mt5.shutdown()

if __name__ == "__main__":
    debug_data()
