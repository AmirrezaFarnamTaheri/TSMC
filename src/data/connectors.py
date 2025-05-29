import os
import requests
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

logger = logging.getLogger(__name__)

class BaseConnector:
    """Enhanced base class for data source connectors with retry logic"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_url = config.get("api_url")
        self.params = self._process_params(config.get("params", {}))
        self.timeout = self.params.get("timeout", 30)
        self.retry_count = self.params.get("retry_count", 3)
        self.required = config.get("required", True)
        
        # Set up session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.retry_count,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def _process_params(self, params: Dict[str, str]) -> Dict[str, str]:
        """Process parameters, replacing environment variable placeholders"""
        processed = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                processed[key] = os.environ.get(env_var, "")
            else:
                processed[key] = value
        return processed
        
    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from source. Must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement get_data")
    
    def _make_request(self, url: str, params: Dict[str, Any]) -> requests.Response:
        """Make HTTP request with error handling and retries"""
        try:
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            if self.required:
                raise
            else:
                logger.warning(f"Non-required data source failed, continuing...")
                return None

class MarketDataConnector(BaseConnector):
    """Enhanced connector for Tehran Stock Exchange market data"""
    
    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        logger.info(f"Fetching market data from {start_date} to {end_date}")
        
        # If no API key provided, return sample data for testing
        if not self.params.get("api_key"):
            logger.warning("No market API key provided, returning sample data")
            return self._generate_sample_data(start_date, end_date)
        
        params = {
            **{k: v for k, v in self.params.items() if k not in ["timeout", "retry_count"]},
            "start_date": start_date,
            "end_date": end_date,
            "symbols": "all"
        }
        
        try:
            response = self._make_request(self.api_url, params)
            if response is None:
                return self._generate_sample_data(start_date, end_date)
                
            data = response.json()
            
            # Process the data into a DataFrame
            records = []
            for symbol_data in data.get("data", []):
                symbol = symbol_data.get("symbol")
                
                for price_data in symbol_data.get("prices", []):
                    record = {
                        "symbol": symbol,
                        "date": price_data.get("date"),
                        "open": float(price_data.get("open", 0)),
                        "high": float(price_data.get("high", 0)),
                        "low": float(price_data.get("low", 0)),
                        "close": float(price_data.get("close", 0)),
                        "volume": int(price_data.get("volume", 0)),
                        "value": float(price_data.get("value", 0))
                    }
                    records.append(record)
            
            if not records:
                logger.warning("No market data received, returning sample data")
                return self._generate_sample_data(start_date, end_date)
            
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            
            return self._validate_data(df)
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            if self.required:
                return self._generate_sample_data(start_date, end_date)
            else:
                return pd.DataFrame()

    def _generate_sample_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate sample market data for testing"""
        from src.data.config import DEFAULT_SYMBOLS
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        records = []
        
        for symbol in DEFAULT_SYMBOLS[:5]:  # Use first 5 symbols
            base_price = 10000 + hash(symbol) % 50000  # Pseudo-random base price
            
            for i, date in enumerate(date_range):
                # Generate realistic price movements
                price_change = (hash(f"{symbol}{date}") % 200 - 100) / 100.0 * 0.05  # ±5%
                price = base_price * (1 + price_change) * (1 + i * 0.001)  # Slight upward trend
                
                record = {
                    "symbol": symbol,
                    "date": date,
                    "open": price * (1 + (hash(f"o{symbol}{date}") % 20 - 10) / 1000),
                    "high": price * (1 + abs(hash(f"h{symbol}{date}") % 30) / 1000),
                    "low": price * (1 - abs(hash(f"l{symbol}{date}") % 30) / 1000),
                    "close": price,
                    "volume": abs(hash(f"v{symbol}{date}")) % 1000000 + 100000,
                    "value": price * (abs(hash(f"val{symbol}{date}")) % 1000000 + 100000)
                }
                records.append(record)
        
        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)
        return df

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean market data"""
        from src.data.config import VALIDATION_RULES
        
        # Check required columns
        required_cols = VALIDATION_RULES["required_columns"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
        
        # Remove invalid prices
        price_min = VALIDATION_RULES["price_range"]["min"]
        price_max = VALIDATION_RULES["price_range"]["max"]
        
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df = df[(df[col] >= price_min) & (df[col] <= price_max)]
        
        # Remove rows where high < low (data errors)
        if "high" in df.columns and "low" in df.columns:
            df = df[df["high"] >= df["low"]]
        
        return df

class NewsDataConnector(BaseConnector):
    """Enhanced connector for financial news data"""
    
    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        logger.info(f"Fetching news data from {start_date} to {end_date}")
        
        if not self.params.get("api_key"):
            logger.warning("No news API key provided, returning empty DataFrame")
            return pd.DataFrame()
        
        params = {
            **{k: v for k, v in self.params.items() if k not in ["timeout", "retry_count"]},
            "start_date": start_date,
            "end_date": end_date
        }
        
        try:
            response = self._make_request(self.api_url, params)
            if response is None:
                return pd.DataFrame()
                
            data = response.json()
            
            # Process the data into a DataFrame
            records = []
            for news_item in data.get("data", []):
                # Extract ticker mentions from news
                tickers = news_item.get("tickers", [])
                sentiment = news_item.get("sentiment", {})
                
                for ticker in tickers:
                    record = {
                        "symbol": ticker,
                        "date": news_item.get("date"),
                        "headline": news_item.get("headline"),
                        "sentiment_score": float(sentiment.get("score", 0)),
                        "sentiment_label": sentiment.get("label", "neutral"),
                        "news_id": news_item.get("id")
                    }
                    records.append(record)
            
            if not records:
                return pd.DataFrame()
                
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            
            # Aggregate sentiment by day and ticker
            df = df.groupby(["date", "symbol"]).agg({
                "sentiment_score": "mean",
                "news_id": "count"
            }).reset_index()
            
            df.rename(columns={"news_id": "news_count"}, inplace=True)
            df.set_index("date", inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return pd.DataFrame()

class EconomicDataConnector(BaseConnector):
    """Enhanced connector for macroeconomic indicators"""
    
    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        logger.info(f"Fetching economic data from {start_date} to {end_date}")
        
        if not self.params.get("api_key"):
            logger.warning("No economic API key provided, returning empty DataFrame")
            return pd.DataFrame()
        
        params = {
            **{k: v for k, v in self.params.items() if k not in ["timeout", "retry_count"]},
            "start_date": start_date,
            "end_date": end_date,
            "indicators": "inflation,interest_rate,exchange_rate,gdp_growth"
        }
        
        try:
            response = self._make_request(self.api_url, params)
            if response is None:
                return pd.DataFrame()
                
            data = response.json()
            
            # Process the data into a DataFrame
            records = []
            for indicator in data.get("data", []):
                indicator_name = indicator.get("name")
                
                for value_data in indicator.get("values", []):
                    record = {
                        "date": value_data.get("date"),
                        indicator_name: float(value_data.get("value", 0))
                    }
                    records.append(record)
            
            if not records:
                return pd.DataFrame()
                
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            
            # Pivot to get each indicator as a column
            df = df.groupby("date").first().reset_index()
            
            # Forward fill missing values
            df = df.fillna(method="ffill")
            df.set_index("date", inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching economic data: {e}")
            return pd.DataFrame()
