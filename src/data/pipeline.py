import pandas as pd
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

from src.data.connectors import MarketDataConnector, NewsDataConnector, EconomicDataConnector
from src.data.feature_factory import FeatureFactory

logger = logging.getLogger(__name__)

class DataPipeline:
    """End-to-end data pipeline for stock prediction"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data pipeline with configuration
        
        Args:
            config: Configuration dictionary with data sources
        """
        # Load configuration
        if config is None:
            from src.data.config import DATA_SOURCES
            config = DATA_SOURCES
        
        self.config = config
        
        # Initialize connectors
        self.connectors = {
            "market": MarketDataConnector(self.config["market"]),
            "news": NewsDataConnector(self.config["news"]),
            "economic": EconomicDataConnector(self.config["economic"])
        }
        
        # Initialize feature factory
        self.feature_factory = FeatureFactory()
        
    def fetch_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from all sources for the given date range
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary of DataFrames with data from each source
        """
        logger.info(f"Fetching data from {start_date} to {end_date}")
        
        data = {}
        for source_name, connector in self.connectors.items():
            try:
                source_data = connector.get_data(start_date, end_date)
                data[source_name] = source_data
                logger.info(f"Successfully fetched {source_name} data: {len(source_data)} records")
            except Exception as e:
                logger.error(f"Error fetching {source_name} data: {e}")
                # Continue with other data sources even if one fails
                data[source_name] = pd.DataFrame()
        
        return data
    
    def merge_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge data from different sources
        
        Args:
            data_dict: Dictionary of DataFrames from different sources
            
        Returns:
            Merged DataFrame with all data
        """
        if not data_dict.get("market", pd.DataFrame()).empty:
            # Market data is the primary source, others are merged in
            base_df = data_dict["market"].copy()
            logger.info(f"Base market data: {len(base_df)} rows")
            
            # Merge news data if available
            if not data_dict.get("news", pd.DataFrame()).empty:
                news_df = data_dict["news"]
                # Join on date and symbol
                base_df = base_df.reset_index()
                news_df = news_df.reset_index()
                
                base_df = pd.merge(
                    base_df, 
                    news_df, 
                    on=["date", "symbol"], 
                    how="left"
                )
                
                base_df = base_df.set_index("date")
                logger.info(f"Merged news data, now: {len(base_df)} rows")
            
            # Merge economic data if available (applies to all symbols)
            if not data_dict.get("economic", pd.DataFrame()).empty:
                econ_df = data_dict["economic"]
                
                base_df = base_df.reset_index()
                econ_df = econ_df.reset_index()
                
                base_df = pd.merge(
                    base_df,
                    econ_df,
                    on="date",
                    how="left"
                )
                
                base_df = base_df.set_index("date")
                logger.info(f"Merged economic data, now: {len(base_df)} rows")
            
            return base_df
        else:
            logger.error("No market data available, cannot create merged dataset")
            return pd.DataFrame()
    
    def process_ticker(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Process data for a specific ticker
        
        Args:
            data: Merged data from all sources
            ticker: Stock symbol to process
            
        Returns:
            DataFrame with processed features for the ticker
        """
        # Filter data for this ticker
        ticker_data = data[data["symbol"] == ticker].copy()
        
        if len(ticker_data) < 20:  # Need sufficient data for features
            logger.warning(f"Insufficient data for ticker {ticker}: {len(ticker_data)} rows")
            return pd.DataFrame()
        
        # Sort by date
        ticker_data = ticker_data.sort_index()
        
        # Generate features
        try:
            features = self.feature_factory.generate_features(ticker_data)
            return features
        except Exception as e:
            logger.error(f"Error generating features for ticker {ticker}: {e}")
            return pd.DataFrame()
    
    def process_data(self, start_date: str, end_date: str, tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        End-to-end data processing
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            tickers: List of tickers to process (None for all)
            
        Returns:
            Dictionary of processed DataFrames by ticker
        """
        # Fetch raw data
        raw_data = self.fetch_data(start_date, end_date)
        
        # Merge data sources
        merged_data = self.merge_data(raw_data)
        
        if merged_data.empty:
            logger.error("No data to process after merging")
            return {}
        
        # Get list of tickers if not provided
        if tickers is None:
            tickers = merged_data["symbol"].unique().tolist()
        
        # Process each ticker
        processed_data = {}
        for ticker in tickers:
            ticker_data = self.process_ticker(merged_data, ticker)
            if not ticker_data.empty:
                processed_data[ticker] = ticker_data
                logger.info(f"Processed {ticker}: {ticker_data.shape[1]} features, {len(ticker_data)} rows")
            
        return processed_data
