from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
from .googlenews_utils import getNewsData


def get_google_news(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"] = None,
    end_date: Annotated[str, "End date in yyyy-mm-dd format"] = None,
) -> str:
    query = ticker.replace(" ", "+")

    # Default to today if end_date is missing
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    # Default to 7 days lookback if start_date is missing
    if not start_date:
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_date = (end_dt - relativedelta(days=7)).strftime("%Y-%m-%d")
        except ValueError:
             # Fallback if end_date format is weird
             start_date = (datetime.now() - relativedelta(days=7)).strftime("%Y-%m-%d")

    news_results = getNewsData(query, start_date, end_date)

    news_str = ""

    # Limit to 5 articles to save tokens
    for news in news_results[:5]:
        news_str += (
            f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        return ""

    return f"## {ticker} Google News, from {start_date} to {end_date}:\n\n{news_str}"


def get_global_google_news(
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"] = 7,
    limit: Annotated[int, "Maximum number of articles to return"] = 5,
) -> str:
    # Calculate start and end dates for get_google_news
    end_date = curr_date
    try:
        start_dt = datetime.strptime(curr_date, "%Y-%m-%d") - relativedelta(days=look_back_days)
        start_date = start_dt.strftime("%Y-%m-%d")
    except ValueError:
        # Fallback
        start_date = None
        
    return get_google_news("global financial market news", start_date, end_date)