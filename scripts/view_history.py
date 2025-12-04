"""
Script to view trade history from MT5
"""
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

def get_trade_history():
    # Initialize MT5
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return
    
    # Login
    login = int(os.getenv("MT5_LOGIN") or os.getenv("MT5_ACCOUNT"))
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    
    if not mt5.login(login, password=password, server=server):
        print(f"Login failed: {mt5.last_error()}")
        mt5.shutdown()
        return
    
    print(f"✅ Connected to account #{login}")
    print(f"{'='*80}")
    
    # Get history for today
    from_date = datetime.now() - timedelta(days=1)
    to_date = datetime.now() + timedelta(days=1)
    
    # Get deals (completed trades)
    deals = mt5.history_deals_get(from_date, to_date)
    
    if deals is None or len(deals) == 0:
        print("No trades found in history")
        mt5.shutdown()
        return
    
    # Filter only our bot's trades (magic = 999000)
    bot_deals = [d for d in deals if d.magic == 999000 and d.entry == 1]  # entry=1 means OUT (closed)
    
    print(f"\n📊 SOLDIER BOT TRADE HISTORY (Last 24h)")
    print(f"{'='*80}")
    print(f"{'Time':<20} {'Type':<6} {'Volume':<8} {'Open':<12} {'Close':<12} {'Profit':<12}")
    print(f"{'-'*80}")
    
    total_profit = 0
    wins = 0
    losses = 0
    
    for deal in bot_deals:
        time_str = datetime.fromtimestamp(deal.time).strftime("%Y-%m-%d %H:%M")
        deal_type = "BUY" if deal.type == 0 else "SELL"
        profit = deal.profit
        total_profit += profit
        
        if profit > 0:
            wins += 1
            emoji = "💰"
        else:
            losses += 1
            emoji = "💸"
        
        print(f"{time_str:<20} {deal_type:<6} {deal.volume:<8.2f} {deal.price:<12.2f} {deal.price:<12.2f} {emoji} ${profit:<10.2f}")
    
    print(f"{'-'*80}")
    print(f"\n📈 SUMMARY:")
    print(f"   Total Trades: {wins + losses}")
    print(f"   Wins: {wins} | Losses: {losses}")
    print(f"   Win Rate: {wins/(wins+losses)*100:.1f}%" if wins+losses > 0 else "   Win Rate: N/A")
    
    total_emoji = "💰" if total_profit > 0 else "💸"
    print(f"   Total P/L: {total_emoji} ${total_profit:.2f}")
    
    # Account info
    account = mt5.account_info()
    print(f"\n💵 ACCOUNT:")
    print(f"   Balance: ${account.balance:.2f}")
    print(f"   Equity: ${account.equity:.2f}")
    print(f"   Profit: ${account.profit:.2f}")
    
    mt5.shutdown()

if __name__ == "__main__":
    get_trade_history()
