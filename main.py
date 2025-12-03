from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
# config["deep_think_llm"] = "gpt-4o-mini"  # Use a different model
# config["quick_think_llm"] = "gpt-4o-mini"  # Use a different model
config["max_debate_rounds"] = 1  # Increase debate rounds

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
from datetime import datetime
today = datetime.now().strftime("%Y-%m-%d")
_, decision = ta.propagate("XAUUSDm", today)
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
