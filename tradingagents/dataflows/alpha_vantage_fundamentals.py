from .alpha_vantage_common import _make_api_request

# Commodity tickers that don't have traditional fundamentals
COMMODITY_PATTERNS = {"XAUUSD", "XAGUSD", "GOLD", "SILVER", "GC", "SI", "CL", "NG"}


def _is_commodity(ticker: str) -> bool:
    """Check if ticker is a commodity."""
    # Remove common suffixes like 'm' for micro contracts
    clean_ticker = ticker.upper().rstrip("M")
    return any(pattern in clean_ticker for pattern in COMMODITY_PATTERNS)


def _get_commodity_fundamentals(ticker: str) -> str:
    """Return macro context for commodity tickers."""
    if "XAU" in ticker.upper() or "GOLD" in ticker.upper():
        return """## XAUUSD (Gold) - Macro Fundamentals

Gold is a commodity, not a stock. Traditional fundamentals don't apply. Key drivers:

**1. USD Strength (DXY Index)**
- Inverse correlation: DXY up → Gold down
- Watch Fed policy, Treasury yields

**2. Interest Rates**
- Higher rates = lower gold (opportunity cost of holding non-yielding asset)
- Real rates (nominal - inflation) matter most

**3. Inflation Expectations**
- Gold is a hedge against inflation
- CPI data and inflation breakevens are key

**4. Geopolitical Risk**
- Safe-haven asset during crises, wars, uncertainty
- Flight to quality during market stress

**5. Central Bank Buying**
- Central banks are net buyers (especially China, Russia, India)
- Supports long-term demand

**Current Macro Regime Factors to Consider:**
- Fed rate path (hawkish/dovish)
- US economic data (jobs, GDP, PMI)
- Geopolitical tensions
- Dollar liquidity conditions"""
    else:
        return f"""## {ticker} - Commodity Fundamentals

This is a commodity ticker. Traditional stock fundamentals (P/E, revenue, etc.) don't apply.
Key drivers: Supply/demand, USD strength, interest rates, and macro conditions."""


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol using Alpha Vantage.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (not used for Alpha Vantage)

    Returns:
        str: Company overview data including financial ratios and key metrics
    """
    # Handle commodities specially
    if _is_commodity(ticker):
        return _get_commodity_fundamentals(ticker)
    
    params = {
        "symbol": ticker,
    }

    return _make_api_request("OVERVIEW", params)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """
    Retrieve balance sheet data for a given ticker symbol using Alpha Vantage.

    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly) - not used for Alpha Vantage
        curr_date (str): Current date you are trading at, yyyy-mm-dd (not used for Alpha Vantage)

    Returns:
        str: Balance sheet data with normalized fields
    """
    params = {
        "symbol": ticker,
    }

    return _make_api_request("BALANCE_SHEET", params)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """
    Retrieve cash flow statement data for a given ticker symbol using Alpha Vantage.

    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly) - not used for Alpha Vantage
        curr_date (str): Current date you are trading at, yyyy-mm-dd (not used for Alpha Vantage)

    Returns:
        str: Cash flow statement data with normalized fields
    """
    params = {
        "symbol": ticker,
    }

    return _make_api_request("CASH_FLOW", params)


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """
    Retrieve income statement data for a given ticker symbol using Alpha Vantage.

    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly) - not used for Alpha Vantage
        curr_date (str): Current date you are trading at, yyyy-mm-dd (not used for Alpha Vantage)

    Returns:
        str: Income statement data with normalized fields
    """
    params = {
        "symbol": ticker,
    }

    return _make_api_request("INCOME_STATEMENT", params)

