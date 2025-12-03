import time
import json
import os
import sys
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import yaml

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Load Config
config = load_config()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration
BIAS_FILE = os.path.join(PROJECT_ROOT, config["bias_file"])
SYMBOL = config["symbol"]
INTERVAL_SECONDS = config.get("interval_seconds", 900)

def update_bias(bias, reasoning):
    data = {
        "bias": bias,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reasoning": reasoning
    }
    
    with open(BIAS_FILE, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Updated Bias to: {bias}")

def get_general_decision(graph):
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"General is thinking... (Date: {today})")
    
    # Run the graph
    # The graph returns a final state. We need to extract the "Bias" from it.
    # Currently, the graph produces a "final_trade_decision" (BUY/SELL/HOLD).
    # We can map this to our Scalping Bias.
    
    try:
        final_state, signal = graph.propagate(SYMBOL, today)
        decision = final_state.get("final_trade_decision", "HOLD").upper()
        
        # Map decision to Bias
        if "BUY" in decision:
            return "BULLISH_SCALPING", decision
        elif "SELL" in decision:
            return "BEARISH_SCALPING", decision
        else:
            return "NEUTRAL", decision
            
    except Exception as e:
        print(f"Error in General's thought process: {e}")
        return "NEUTRAL", f"Error: {e}"

def main():
    print("General Agent Starting...")
    
    # Initialize Graph
    config = DEFAULT_CONFIG.copy()
    # Ensure we use the optimized settings
    ta = TradingAgentsGraph(debug=True, config=config)
    
    while True:
        try:
            bias, reasoning = get_general_decision(ta)
            update_bias(bias, reasoning)
            
            print(f"Sleeping for {INTERVAL_SECONDS} seconds...")
            time.sleep(INTERVAL_SECONDS)
            
        except KeyboardInterrupt:
            print("General Stopped")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
