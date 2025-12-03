import time
import os
import sys
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

from tradingagents.dataflows.y_finance import get_stock_stats_indicators_window

print("Testing optimized implementation with 30-day lookback:")
start_time = time.time()
result = get_stock_stats_indicators_window("AAPL", "macd", "2024-11-01", 30)
end_time = time.time()

print(f"Execution time: {end_time - start_time:.2f} seconds")
print(f"Result length: {len(result)} characters")
print(result)
