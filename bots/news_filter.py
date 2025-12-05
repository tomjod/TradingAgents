"""
News Filter - Alpha Vantage News Sentiment API Integration
Pauses trading during high-impact news events that break technical patterns.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "news_cache.json")


class NewsFilter:
    """
    Filters trading based on high-impact news events.
    
    High-Impact Events for Gold (XAUUSD):
    - Federal Reserve Speeches (Jerome Powell)
    - Interest Rate Decisions
    - Non-Farm Payrolls (NFP)
    - Consumer Price Index (CPI)
    - GDP Reports
    """
    
    def __init__(self, config: dict):
        news_config = config.get("news_filter", {})
        
        self.enabled = news_config.get("enabled", True)
        self.pause_minutes_before = news_config.get("pause_minutes_before", 30)
        self.pause_minutes_after = news_config.get("pause_minutes_after", 30)
        self.cache_duration_minutes = news_config.get("cache_duration_minutes", 60)
        
        # Keywords that indicate high-impact events for Gold
        self.high_impact_keywords = news_config.get("high_impact_keywords", [
            "federal reserve",
            "interest rate",
            "fomc",
            "jerome powell",
            "inflation",
            "cpi",
            "consumer price",
            "non-farm",
            "nonfarm",
            "employment",
            "gdp",
            "treasury",
            "dollar",
            "usd"
        ])
        
        # Sentiment score thresholds
        self.bearish_threshold = news_config.get("bearish_threshold", 0.3)
        self.bullish_threshold = news_config.get("bullish_threshold", 0.7)
        
        self.cache = self._load_cache()
        
        if self.enabled:
            print(f"📰 News Filter initialized")
            print(f"   Pause: {self.pause_minutes_before}m before / {self.pause_minutes_after}m after events")
        else:
            print(f"📰 News Filter DISABLED")
    
    def _load_cache(self) -> dict:
        """Load cached news data"""
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ News cache load error: {e}")
        return {"last_update": None, "articles": [], "high_impact_active": False}
    
    def _save_cache(self):
        """Save news data to cache"""
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"⚠️ News cache save error: {e}")
    
    def _fetch_news(self) -> list:
        """Fetch news from Alpha Vantage API"""
        if not ALPHA_VANTAGE_API_KEY:
            print("⚠️ ALPHA_VANTAGE_API_KEY not set in .env")
            return []
        
        try:
            # Query for USD/Gold related news
            url = (
                f"https://www.alphavantage.co/query?"
                f"function=NEWS_SENTIMENT"
                f"&tickers=FOREX:XAU,FOREX:USD"
                f"&topics=economy_monetary,economy_fiscal,financial_markets"
                f"&sort=LATEST"
                f"&limit=50"
                f"&apikey={ALPHA_VANTAGE_API_KEY}"
            )
            
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if "feed" in data:
                return data["feed"]
            elif "Note" in data:
                print(f"⚠️ Alpha Vantage rate limit: {data['Note'][:50]}...")
            elif "Error Message" in data:
                print(f"⚠️ Alpha Vantage error: {data['Error Message'][:50]}...")
                
        except Exception as e:
            print(f"⚠️ News fetch error: {e}")
        
        return []
    
    def _is_high_impact(self, article: dict) -> bool:
        """Check if article is high-impact for Gold trading"""
        title = article.get("title", "").lower()
        summary = article.get("summary", "").lower()
        text = f"{title} {summary}"
        
        # Check for high-impact keywords
        for keyword in self.high_impact_keywords:
            if keyword.lower() in text:
                return True
        
        return False
    
    def _parse_time(self, time_str: str) -> datetime:
        """Parse Alpha Vantage time format (YYYYMMDDTHHMMSS)"""
        try:
            return datetime.strptime(time_str, "%Y%m%dT%H%M%S")
        except:
            return datetime.now()
    
    def update_news(self) -> bool:
        """Update news cache if stale"""
        if not self.enabled:
            return True
        
        # Check if cache is fresh
        if self.cache.get("last_update"):
            try:
                last_update = datetime.fromisoformat(self.cache["last_update"])
                if datetime.now() - last_update < timedelta(minutes=self.cache_duration_minutes):
                    return True  # Cache is fresh
            except:
                pass
        
        # Fetch fresh news
        articles = self._fetch_news()
        
        if articles:
            # Filter high-impact articles from the last hour
            now = datetime.now()
            recent_high_impact = []
            
            for article in articles:
                time_published = self._parse_time(article.get("time_published", ""))
                age_minutes = (now - time_published).total_seconds() / 60
                
                # Check if article is recent and high-impact
                if age_minutes < 120 and self._is_high_impact(article):
                    recent_high_impact.append({
                        "title": article.get("title", "")[:100],
                        "time": article.get("time_published", ""),
                        "sentiment": article.get("overall_sentiment_score", 0)
                    })
            
            self.cache = {
                "last_update": now.isoformat(),
                "articles": recent_high_impact,
                "high_impact_active": len(recent_high_impact) > 0
            }
            self._save_cache()
            
            if recent_high_impact:
                print(f"📰 Found {len(recent_high_impact)} high-impact news articles!")
        
        return True
    
    def can_trade(self) -> tuple:
        """
        Check if trading is allowed based on news.
        
        Returns: (allowed: bool, reason: str)
        """
        if not self.enabled:
            return True, "News filter disabled"
        
        # Update news if needed
        self.update_news()
        
        # Check if high-impact news is active
        if self.cache.get("high_impact_active", False):
            articles = self.cache.get("articles", [])
            if articles:
                latest_title = articles[0].get("title", "Unknown")[:50]
                return False, f"High-impact news: {latest_title}..."
        
        return True, "OK"
    
    def get_sentiment_bias(self) -> str:
        """
        Get overall sentiment from news.
        
        Returns: "BULLISH", "BEARISH", or "NEUTRAL"
        """
        if not self.enabled or not self.cache.get("articles"):
            return "NEUTRAL"
        
        # Calculate average sentiment
        sentiments = [a.get("sentiment", 0) for a in self.cache.get("articles", [])]
        
        if not sentiments:
            return "NEUTRAL"
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        if avg_sentiment >= self.bullish_threshold:
            return "BULLISH"
        elif avg_sentiment <= self.bearish_threshold:
            return "BEARISH"
        
        return "NEUTRAL"
    
    def get_status(self) -> dict:
        """Get current news filter status"""
        return {
            "enabled": self.enabled,
            "high_impact_active": self.cache.get("high_impact_active", False),
            "article_count": len(self.cache.get("articles", [])),
            "last_update": self.cache.get("last_update"),
            "sentiment_bias": self.get_sentiment_bias()
        }


# Standalone test
if __name__ == "__main__":
    import yaml
    
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Test news filter
    news = NewsFilter(config)
    
    can_trade, reason = news.can_trade()
    print(f"Can trade: {can_trade} - {reason}")
    
    status = news.get_status()
    print(f"Status: {json.dumps(status, indent=2)}")
