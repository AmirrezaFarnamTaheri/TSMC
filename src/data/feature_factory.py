import pandas as pd
import numpy as np
import ta
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class FeatureFactory:
    """Generate features for stock prediction"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature factory with configuration
        
        Args:
            config: Configuration dictionary with feature parameters
        """
        self.config = config or {}
        self.frequencies = self.config.get("frequencies", ["1D"])
        self.windows = self.config.get("windows", [5, 10, 20, 50])
        self.technical_indicators = self.config.get("technical_indicators", [])
        self.tsfresh_params = self.config.get("tsfresh_params", {"minimal": True})
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive feature set from raw data
        
        Args:
            df: DataFrame with OHLCV data, date as index
            
        Returns:
            DataFrame with all features
        """
        logger.info(f"Generating features for data with shape {df.shape}")
        
        # Ensure we have the required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure data is sorted by date
        df = df.sort_index()
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        # Add price-based features
        df = self._add_price_features(df)
        
        # Add volatility features
        df = self._add_volatility_features(df)
        
        # Add sentiment features if available
        if "sentiment_score" in df.columns:
            df = self._add_sentiment_features(df)
        
        # Add economic features if available
        economic_cols = ["inflation", "interest_rate", "exchange_rate", "gdp_growth"]
        has_economic = any(col in df.columns for col in economic_cols)
        if has_economic:
            df = self._add_economic_features(df, economic_cols)
        
        # Multi-frequency features
        multifreq_features = self._generate_multifrequency_features(df)
        if not multifreq_features.empty:
            # Resample back to daily and merge
            daily_index = df.index
            multifreq_features = multifreq_features.reindex(daily_index, method="ffill")
            df = pd.concat([df, multifreq_features], axis=1)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove duplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]
        
        logger.info(f"Generated {df.shape[1]} features")
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset"""
        result_df = df.copy()
        
        if "rsi" in self.technical_indicators:
            for window in self.windows:
                if window <= len(df):
                    result_df[f'rsi_{window}'] = ta.momentum.RSIIndicator(
                        close=df['close'], window=window
                    ).rsi()
                
        if "macd" in self.technical_indicators:
            macd = ta.trend.MACD(
                close=df['close'], window_slow=26, window_fast=12, window_sign=9
            )
            result_df['macd'] = macd.macd()
            result_df['macd_signal'] = macd.macd_signal()
            result_df['macd_diff'] = macd.macd_diff()
            
        if "bollinger_bands" in self.technical_indicators:
            for window in [20]:  # Standard window for Bollinger Bands
                if window <= len(df):
                    bollinger = ta.volatility.BollingerBands(
                        close=df['close'], window=window, window_dev=2
                    )
                    result_df[f'bollinger_mavg_{window}'] = bollinger.bollinger_mavg()
                    result_df[f'bollinger_hband_{window}'] = bollinger.bollinger_hband()
                    result_df[f'bollinger_lband_{window}'] = bollinger.bollinger_lband()
                    result_df[f'bollinger_width_{window}'] = bollinger.bollinger_wband()
                
        if "stoch" in self.technical_indicators:
            stoch = ta.momentum.StochasticOscillator(
                high=df['high'], low=df['low'], close=df['close'], 
                window=14, smooth_window=3
            )
            result_df['stoch_k'] = stoch.stoch()
            result_df['stoch_d'] = stoch.stoch_signal()
            
        if "adx" in self.technical_indicators:
            adx = ta.trend.ADXIndicator(
                high=df['high'], low=df['low'], close=df['close'], window=14
            )
            result_df['adx'] = adx.adx()
            result_df['adx_pos'] = adx.adx_pos()
            result_df['adx_neg'] = adx.adx_neg()
            
        if "obv" in self.technical_indicators:
            result_df['obv'] = ta.volume.OnBalanceVolumeIndicator(
                close=df['close'], volume=df['volume']
            ).on_balance_volume()
            
        if "atr" in self.technical_indicators:
            for window in [14]:  # Standard window for ATR
                if window <= len(df):
                    result_df[f'atr_{window}'] = ta.volatility.AverageTrueRange(
                        high=df['high'], low=df['low'], close=df['close'], window=window
                    ).average_true_range()
                
        if "williams_r" in self.technical_indicators:
            for window in [14]:  # Standard window for Williams %R
                if window <= len(df):
                    result_df[f'williams_r_{window}'] = ta.momentum.WilliamsRIndicator(
                        high=df['high'], low=df['low'], close=df['close'], lbp=window
                    ).williams_r()
        
        return result_df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        result_df = df.copy()
        
        # Returns over different periods
        for window in self.windows:
            if window <= len(df):
                result_df[f'return_{window}d'] = df['close'].pct_change(window)
        
        # Price momentum
        for window in self.windows:
            if window <= len(df):
                result_df[f'momentum_{window}d'] = df['close'] / df['close'].shift(window) - 1
        
        # Moving averages
        for window in self.windows:
            if window <= len(df):
                result_df[f'ma_{window}d'] = df['close'].rolling(window=window).mean()
                
        # Moving average convergence/divergence
        for window in self.windows:
            if window > 5 and window <= len(df):  # Avoid too short windows
                ma_col = f'ma_{window}d'
                if ma_col in result_df.columns:
                    result_df[f'close_to_ma_{window}d'] = df['close'] / result_df[ma_col] - 1
        
        # High-Low ranges
        result_df['daily_range'] = (df['high'] - df['low']) / df['close']
        for window in self.windows:
            if window <= len(df):
                result_df[f'range_{window}d_avg'] = result_df['daily_range'].rolling(window=window).mean()
            
        # Day of week, month features
        result_df['day_of_week'] = df.index.dayofweek
        result_df['month'] = df.index.month
        result_df['day_of_month'] = df.index.day
        
        return result_df
        
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        result_df = df.copy()
        
        # Historical volatility
        for window in self.windows:
            if window <= len(df):
                result_df[f'volatility_{window}d'] = df['close'].pct_change().rolling(window=window).std()
        
        # Normalized range
        for window in self.windows:
            if window <= len(df):
                result_df[f'norm_range_{window}d'] = (df['high'].rolling(window).max() - df['low'].rolling(window).min()) / df['close']
        
        # Garman-Klass volatility
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        result_df['gk_volatility'] = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        
        return result_df
    
    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment-based features if available"""
        result_df = df.copy()
        
        # Rolling sentiment metrics
        for window in self.windows:
            if window <= len(df):
                result_df[f'sentiment_{window}d_avg'] = df['sentiment_score'].rolling(window=window).mean()
            
        if 'news_count' in df.columns:
            # News volume features
            for window in self.windows:
                if window <= len(df):
                    result_df[f'news_volume_{window}d'] = df['news_count'].rolling(window=window).sum()
            
            # News impact: higher sentiment shift with higher news volume
            result_df['news_impact'] = result_df['sentiment_score'] * np.log1p(df['news_count'])
        
        return result_df
    
    def _add_economic_features(self, df: pd.DataFrame, economic_cols: List[str]) -> pd.DataFrame:
        """Add economic indicator features if available"""
        result_df = df.copy()
        
        # Calculate changes in economic indicators
        for col in economic_cols:
            if col in df.columns:
                # Monthly or quarterly change
                result_df[f'{col}_change'] = df[col].pct_change()
                
                # Longer-term trends (e.g., 6-month change)
                if len(df) > 126:  # ~6 months of data
                    result_df[f'{col}_change_6m'] = df[col].pct_change(126)
        
        return result_df
    
    def _generate_multifrequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features at multiple frequencies"""
        all_features = pd.DataFrame()
        
        # Skip daily as it's already processed
        non_daily_freqs = [f for f in self.frequencies if f != "1D"]
        
        for freq in non_daily_freqs:
            try:
                # Resample to frequency
                resampled = df.resample(freq).agg({
                    'open': 'first', 
                    'high': 'max', 
                    'low': 'min', 
                    'close': 'last',
                    'volume': 'sum'
                })
                
                if len(resampled) < 4:  # Need at least 4 periods
                    continue
                
                # Calculate simple derived features
                close_pct_change = resampled['close'].pct_change()
                
                # Volatility metrics
                volatility = close_pct_change.rolling(4).std()
                volatility.name = f'volatility_{freq}'
                
                # Return metrics
                returns = close_pct_change
                returns.name = f'return_{freq}'
                
                # Join frequency-specific features
                freq_features = pd.concat([volatility, returns], axis=1)
                freq_features.index.name = 'date'
                
                # Add to all features
                all_features = pd.concat([all_features, freq_features], axis=1)
                
            except Exception as e:
                logger.warning(f"Error generating {freq} features: {e}")
        
        return all_features
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        result_df = df.copy()
        
        # Check missing percentage
        missing_pct = result_df.isna().mean()
        
        # Remove features with too many missing values (>50%)
        cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
        if cols_to_drop:
            logger.warning(f"Dropping {len(cols_to_drop)} columns with >50% missing values")
            result_df = result_df.drop(columns=cols_to_drop)
        
        # Forward-fill remaining missing values (appropriate for time series)
        result_df = result_df.fillna(method='ffill')
        
        # If still have missing values at the beginning, fill with column median
        for col in result_df.select_dtypes(include=[np.number]).columns:
            if result_df[col].isna().any():
                result_df[col] = result_df[col].fillna(result_df[col].median())
        
        return result_df
