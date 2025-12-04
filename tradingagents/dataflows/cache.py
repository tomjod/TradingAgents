"""
Simple in-memory cache for API responses with TTL support.
Reduces redundant API calls and saves tokens.
"""

import time
import hashlib
from typing import Any, Optional, Dict

# Cache storage: {cache_key: (value, expiry_timestamp)}
_cache: Dict[str, tuple] = {}

# TTL in seconds for different data types
TTL_CONFIG = {
    "get_stock_data": 300,       # 5 minutes - prices change
    "get_indicators": 300,       # 5 minutes
    "get_fundamentals": 3600,    # 1 hour - rarely changes
    "get_balance_sheet": 3600,
    "get_cashflow": 3600,
    "get_income_statement": 3600,
    "get_news": 600,             # 10 minutes
    "get_global_news": 600,
    "get_insider_sentiment": 1800,
    "get_insider_transactions": 1800,
    "default": 300,
}


def _make_cache_key(method: str, args: tuple, kwargs: dict) -> str:
    """Create a unique cache key from method name and arguments."""
    # Convert args and kwargs to a string and hash it
    args_str = str(args) + str(sorted(kwargs.items()))
    args_hash = hashlib.md5(args_str.encode()).hexdigest()[:12]
    return f"{method}:{args_hash}"


def get_cached(method: str, args: tuple, kwargs: dict = None) -> Optional[Any]:
    """
    Get cached value if it exists and hasn't expired.
    
    Returns:
        Cached value or None if not found/expired
    """
    if kwargs is None:
        kwargs = {}
    
    cache_key = _make_cache_key(method, args, kwargs)
    
    if cache_key in _cache:
        value, expiry = _cache[cache_key]
        if time.time() < expiry:
            print(f"CACHE HIT: {method} (key={cache_key[:20]}...)")
            return value
        else:
            # Expired, remove it
            del _cache[cache_key]
            print(f"CACHE EXPIRED: {method}")
    
    print(f"CACHE MISS: {method}")
    return None


def set_cached(method: str, args: tuple, kwargs: dict, value: Any) -> None:
    """
    Store a value in the cache with appropriate TTL.
    """
    if kwargs is None:
        kwargs = {}
    
    cache_key = _make_cache_key(method, args, kwargs)
    ttl = TTL_CONFIG.get(method, TTL_CONFIG["default"])
    expiry = time.time() + ttl
    
    _cache[cache_key] = (value, expiry)
    print(f"CACHE SET: {method} (TTL={ttl}s)")


def clear_cache() -> None:
    """Clear all cached values."""
    global _cache
    _cache = {}
    print("CACHE CLEARED")


def get_cache_stats() -> Dict:
    """Get cache statistics."""
    now = time.time()
    total = len(_cache)
    expired = sum(1 for _, (_, exp) in _cache.items() if exp < now)
    return {
        "total_entries": total,
        "expired_entries": expired,
        "active_entries": total - expired,
    }
