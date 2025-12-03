from langchain_core.tools import tool
from typing import Annotated, Optional
from tradingagents.dataflows.interface import route_to_vendor

@tool
def execute_order(
    symbol: Annotated[str, "ticker symbol"],
    action_type: Annotated[str, "BUY or SELL"],
    volume: Annotated[float, "volume/lot size"],
    sl_points: Annotated[Optional[int], "stop loss points (optional)"] = 0,
    tp_points: Annotated[Optional[int], "take profit points (optional)"] = 0
):
    """
    Execute a market order (BUY or SELL) for a given symbol.
    """
    # Handle None values for points
    if sl_points is None: sl_points = 0
    if tp_points is None: tp_points = 0
    
    # Validate volume
    if volume <= 0:
        return "Error: Volume must be greater than 0"
        
    return route_to_vendor("execute_order", symbol, action_type, volume, sl_points, tp_points)

@tool
def get_open_positions(
    symbol: Annotated[str, "ticker symbol (optional)"] = None
):
    """
    Get current open positions. Can be filtered by symbol.
    """
    return route_to_vendor("get_open_positions", symbol)

@tool
def get_trade_history(
    date_from: Annotated[str, "Start date yyyy-mm-dd (optional)"] = None,
    date_to: Annotated[str, "End date yyyy-mm-dd (optional)"] = None
):
    """
    Get closed trade history. Defaults to last 30 days if dates not provided.
    """
    return route_to_vendor("get_trade_history", date_from, date_to)
