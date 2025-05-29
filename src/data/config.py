import os
from typing import Dict, Any

# Data source configurations
DATA_SOURCES = {
    "market": {
        "api_url": os.getenv("MARKET_API_URL", "https://api.tehranexchange.ir/v1/market"),
        "params": {
            "api_key": os.getenv("MARKET_API_KEY", ""),
            "timeout": 30,
            "retry_count": 3
        },
        "required": True
    },
    "news": {
        "api_url": os.getenv("NEWS_API_URL", "https://api.tehranscraper.ir/v1/news"),
        "params": {
            "api_key": os.getenv("NEWS_API_KEY", ""),
            "timeout": 30,
            "retry_count": 3
        },
        "required": False
    },
    "economic": {
        "api_url": os.getenv("ECONOMIC_API_URL", "https://api.cbi.ir/v1/indicators"),
        "params": {
            "api_key": os.getenv("ECON_API_KEY", ""),
            "timeout": 30,
            "retry_count": 3
        },
        "required": False
    }
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "technical_indicators": [
        "rsi", "macd", "bollinger_bands", "stoch", 
        "adx", "obv", "atr", "williams_r"
    ],
    "windows": [5, 10, 20, 50, 100, 200],
    "frequencies": ["1D", "1W", "1M"],
    "tsfresh_params": {
        "enabled": True,
        "minimal": True,
        "n_jobs": 4
    },
    "missing_value_threshold": 0.5,
    "outlier_detection": {
        "enabled": True,
        "method": "iqr",
        "threshold": 3.0
    }
}

# Default stock symbols for Tehran Stock Exchange
DEFAULT_SYMBOLS = [
    "TEDPIX",    # Tehran Stock Exchange Price Index
    "KHODRO",    # Iran Khodro
    "FOLD",      # Foolad Mobarakeh Isfahan
    "VBMELLAT",  # Bank Mellat
    "SHBANDAR",  # Bandar Abbas
    "PICICO",    # Petro Iran
    "BMELLAT",   # Bank Mellat
    "SHPARS",    # Pars Oil and Gas
]

# Validation rules
VALIDATION_RULES = {
    "min_data_points": 100,
    "max_missing_ratio": 0.3,
    "required_columns": ["open", "high", "low", "close", "volume"],
    "date_format": "%Y-%m-%d",
    "price_range": {
        "min": 0,
        "max": 1e12  # 1 trillion IRR
    }
}
