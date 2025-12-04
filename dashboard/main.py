from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json
import os
import sys
from datetime import datetime, timedelta

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI()

# Setup Templates
templates = Jinja2Templates(directory="dashboard/templates")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.path.join(PROJECT_ROOT, "state.json")
MAGIC_NUMBER = 999000  # Soldier bot magic number

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/state")
async def get_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
    return {"error": "State file not found. Is Soldier running?"}

@app.get("/api/history")
async def get_trade_history():
    """Get trade history from MT5"""
    try:
        import MetaTrader5 as mt5
        from dotenv import load_dotenv
        
        load_dotenv()
        
        if not mt5.initialize():
            return {"error": "MT5 not initialized", "trades": []}
        
        # Login
        login = int(os.getenv("MT5_LOGIN") or os.getenv("MT5_ACCOUNT"))
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        
        if not mt5.login(login, password=password, server=server):
            return {"error": "MT5 login failed", "trades": []}
        
        # Get deals from last 7 days
        from_date = datetime.now() - timedelta(days=7)
        to_date = datetime.now() + timedelta(days=1)
        
        deals = mt5.history_deals_get(from_date, to_date)
        
        trades = []
        total_profit = 0
        wins = 0
        losses = 0
        
        if deals:
            for deal in deals:
                # Filter by magic number and entry/exit deals only
                if deal.magic == MAGIC_NUMBER and deal.entry in [1, 2]:
                    trade_type = "BUY" if deal.type == 0 else "SELL"
                    profit = deal.profit
                    total_profit += profit
                    
                    if profit > 0:
                        wins += 1
                    elif profit < 0:
                        losses += 1
                    
                    trades.append({
                        "ticket": deal.ticket,
                        "time": datetime.fromtimestamp(deal.time).strftime("%Y-%m-%d %H:%M"),
                        "type": trade_type,
                        "volume": deal.volume,
                        "price": deal.price,
                        "profit": round(profit, 2),
                        "symbol": deal.symbol
                    })
        
        # Get account info
        account = mt5.account_info()
        
        return {
            "trades": trades[-50:],  # Last 50 trades
            "summary": {
                "total_trades": wins + losses,
                "wins": wins,
                "losses": losses,
                "win_rate": round((wins / (wins + losses) * 100) if (wins + losses) > 0 else 0, 1),
                "total_profit": round(total_profit, 2),
                "balance": round(account.balance, 2) if account else 0,
                "equity": round(account.equity, 2) if account else 0
            }
        }
        
    except Exception as e:
        return {"error": str(e), "trades": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
