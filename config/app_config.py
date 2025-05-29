import os
from typing import Dict, Any

# Application configuration
APP_CONFIG = {
    "debug": os.getenv("DEBUG", "False").lower() == "true",
    "api_port": int(os.getenv("API_PORT", 8000)),
    "dashboard_port": int(os.getenv("DASHBOARD_PORT", 8050)),
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
    "model_dir": os.getenv("MODEL_DIR", "models/trained"),
    "data_dir": os.getenv("DATA_DIR", "data"),
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
}

# Database configuration (if needed for user management)
DATABASE_CONFIG = {
    "url": os.getenv("DATABASE_URL", "sqlite:///app.db"),
    "pool_size": int(os.getenv("DB_POOL_SIZE", 5)),
    "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", 10)),
}

# Security configuration
SECURITY_CONFIG = {
    "secret_key": os.getenv("SECRET_KEY", "your-secret-key-change-in-production"),
    "jwt_algorithm": "HS256",
    "access_token_expire_minutes": int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30)),
}

# Feature flags
FEATURE_FLAGS = {
    "enable_authentication": os.getenv("ENABLE_AUTH", "False").lower() == "true",
    "enable_caching": os.getenv("ENABLE_CACHING", "True").lower() == "true",
    "enable_rate_limiting": os.getenv("ENABLE_RATE_LIMITING", "True").lower() == "true",
    "enable_monitoring": os.getenv("ENABLE_MONITORING", "True").lower() == "true",
}
