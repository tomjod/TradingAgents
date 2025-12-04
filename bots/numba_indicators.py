"""
Numba JIT-compiled functions for high-frequency trading calculations.
These functions run at near-C speed for latency-critical operations.
"""
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def calculate_slopes(rsi_array: np.ndarray, macd_array: np.ndarray) -> tuple:
    """
    Calculate RSI and MACD slopes over the last 3 periods.
    
    Args:
        rsi_array: Array of RSI values (at least 4 elements)
        macd_array: Array of MACD values (at least 4 elements)
    
    Returns:
        Tuple of (rsi_slope, macd_slope)
    """
    rsi_slope = rsi_array[-1] - rsi_array[-4]
    macd_slope = macd_array[-1] - macd_array[-4]
    return rsi_slope, macd_slope


@jit(nopython=True, cache=True)
def calculate_bb_metrics(
    close: float, 
    boll: float, 
    boll_ub: float, 
    boll_lb: float
) -> tuple:
    """
    Calculate Bollinger Band metrics.
    
    Args:
        close: Current close price
        boll: Middle Bollinger Band (SMA)
        boll_ub: Upper Bollinger Band
        boll_lb: Lower Bollinger Band
    
    Returns:
        Tuple of (bb_width, dist_ma)
    """
    bb_width = (boll_ub - boll_lb) / boll
    dist_ma = (close - boll) / boll
    return bb_width, dist_ma


@jit(nopython=True, cache=True)
def calculate_vol_trend(volume_array: np.ndarray) -> float:
    """
    Calculate volume trend as ratio to 20-period moving average.
    
    Args:
        volume_array: Array of volume values (at least 20 elements)
    
    Returns:
        Volume trend ratio
    """
    vol_ma_20 = np.mean(volume_array[-20:])
    if vol_ma_20 == 0:
        return 1.0
    return volume_array[-1] / vol_ma_20


@jit(nopython=True, cache=True)
def calculate_trailing_stop(
    pos_type: int,  # 0 = BUY, 1 = SELL
    current_bid: float,
    current_ask: float,
    open_price: float,
    current_sl: float,
    point: float,
    trailing_start: float,
    trailing_step: float
) -> tuple:
    """
    Calculate new trailing stop level if applicable.
    
    Args:
        pos_type: 0 for BUY, 1 for SELL
        current_bid: Current bid price
        current_ask: Current ask price
        open_price: Position open price
        current_sl: Current stop loss
        point: Symbol point value
        trailing_start: Points profit before trailing starts
        trailing_step: Minimum points between SL updates
    
    Returns:
        Tuple of (should_update, new_sl)
    """
    if pos_type == 0:  # BUY
        dist = (current_bid - open_price) / point
        if dist > trailing_start:
            new_sl = current_bid - trailing_start * point
            if new_sl > current_sl + trailing_step * point:
                return True, new_sl
    else:  # SELL
        dist = (open_price - current_ask) / point
        if dist > trailing_start:
            new_sl = current_ask + trailing_start * point
            if current_sl == 0 or new_sl < current_sl - trailing_step * point:
                return True, new_sl
    
    return False, 0.0


@jit(nopython=True, cache=True)
def calculate_dynamic_lot(
    equity: float,
    risk_percent: float,
    sl_points: float,
    point: float,
    tick_value: float,
    tick_size: float,
    volume_step: float,
    volume_min: float,
    volume_max: float
) -> float:
    """
    Calculate dynamic lot size based on risk percentage.
    
    Args:
        equity: Account equity
        risk_percent: Risk percentage (e.g., 0.01 for 1%)
        sl_points: Stop loss in points
        point: Symbol point value
        tick_value: Value per tick
        tick_size: Size of one tick
        volume_step: Minimum volume increment
        volume_min: Minimum allowed volume
        volume_max: Maximum allowed volume
    
    Returns:
        Calculated lot size
    """
    if tick_size == 0 or tick_value == 0:
        return volume_min
    
    risk_amount = equity * risk_percent
    loss_per_lot = (sl_points * point) * (tick_value / tick_size)
    
    if loss_per_lot == 0:
        return volume_min
    
    calc_volume = risk_amount / loss_per_lot
    
    # Round to step
    calc_volume = round(calc_volume / volume_step) * volume_step
    
    # Clamp to min/max
    if calc_volume < volume_min:
        calc_volume = volume_min
    if calc_volume > volume_max:
        calc_volume = volume_max
    
    return calc_volume


# Pre-compile functions on import (warm-up)
def _warmup():
    """Pre-compile JIT functions with dummy data."""
    rsi = np.array([50.0, 51.0, 52.0, 53.0])
    macd = np.array([0.1, 0.2, 0.3, 0.4])
    calculate_slopes(rsi, macd)
    calculate_bb_metrics(100.0, 100.0, 102.0, 98.0)
    calculate_vol_trend(np.ones(20))
    calculate_trailing_stop(0, 100.0, 100.1, 99.0, 98.5, 0.01, 50.0, 10.0)
    calculate_dynamic_lot(10000.0, 0.01, 100.0, 0.01, 1.0, 0.01, 0.01, 0.01, 100.0)


# Run warmup on import
_warmup()
