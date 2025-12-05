# Python-MQL5 Socket Bridge

## Setup Instructions

### 1. Install MQL5 EA

Copy `mql5/TradingBridge.mq5` to your MetaTrader 5 Experts folder:
```
C:\Users\[YOUR_USERNAME]\AppData\Roaming\MetaQuotes\Terminal\[TERMINAL_ID]\MQL5\Experts\
```

Then compile it in MetaEditor (F7).

### 2. Configure EA Inputs

In MT5, attach the EA to a chart with these settings:
- **InpHost**: 127.0.0.1 (localhost)
- **InpPort**: 5555
- **InpSymbol**: XAUUSDm
- **InpMagic**: 999001

### 3. Start Python Server FIRST

```bash
cd C:\Users\alexi\Proyectos\TradingAgents
conda activate tradingagents
python bots/bridge_server.py
```

### 4. Then Start EA in MT5

Enable "AutoTrading" in MT5 and the EA will connect to Python.

## Architecture

```
Python (Brain)                    MQL5 EA (Muscle)
┌────────────────┐               ┌────────────────┐
│ bridge_server  │◄────TCP────► │ TradingBridge  │
│                │   5555        │                │
│ • AI/ML model  │               │ • Order exec   │
│ • Signals      │               │ • Tick stream  │
│ • Risk mgmt    │               │ • Positions    │
└────────────────┘               └────────────────┘
```

## Commands (Interactive Mode)

| Key | Action |
|-----|--------|
| b | Send BUY order |
| s | Send SELL order |
| c | Close all positions |
| p | Ping (measure latency) |
| t | Show last tick |
| a | Get account info |
| q | Quit |

## JSON Protocol

### Python → MQL5
```json
{"action": "BUY", "volume": 0.01, "sl_points": 800}
{"action": "SELL", "volume": 0.01, "sl_points": 800}
{"action": "CLOSE", "ticket": 12345}
{"action": "CLOSE_ALL"}
```

### MQL5 → Python
```json
{"type": "tick", "bid": 2650.50, "ask": 2650.80, "spread": 30}
{"type": "execution", "action": "BUY", "status": "filled", "ticket": 12345}
{"type": "positions", "data": [...]}
```
