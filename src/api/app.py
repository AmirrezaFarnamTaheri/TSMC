from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from pydantic import BaseModel, Field
import logging
import time
import asyncio
import redis.asyncio as redis
from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
import os

from src.api.model_pool import ModelPool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API models
class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    horizon: int = Field(5, ge=1, le=30, description="Prediction horizon in days")
    include_quantiles: bool = Field(True, description="Include prediction quantiles (uncertainty)")
    include_regime: bool = Field(True, description="Include market regime information")

class HistoricalRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    days: int = Field(30, ge=1, le=365, description="Number of days of historical data")
    include_features: bool = Field(False, description="Include engineered features")

# API setup
app = FastAPI(
    title="Tehran Stock Market Prediction API",
    description="API for predicting stock prices using transformer models",
    version="1.0.0"
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# Initialize model pool
model_pool = None

@app.on_event("startup")
async def startup_event():
    global model_pool
    
    # Initialize Redis if available
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(redis_url, encoding="utf8", decode_responses=True)
        
        # Test Redis connection
        await redis_client.ping()
        
        # Initialize FastAPI cache
        FastAPICache.init(RedisBackend(redis_client), prefix="stockapi-cache:")
        logger.info("Redis cache initialized successfully")
    except Exception as e:
        logger.warning(f"Redis not available, caching disabled: {e}")
    
    # Initialize model pool
    model_pool = ModelPool()
    await model_pool.initialize()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/api/v1/tickers", 
    summary="List available tickers",
    response_description="List of available tickers with metadata"
)
@limiter.limit("60/minute")
async def list_tickers(request: Request):
    """List all available stock tickers that can be predicted"""
    if not model_pool:
        raise HTTPException(status_code=503, detail="Model pool not initialized")
    
    available_tickers = []
    for ticker, metadata in model_pool.available_models.items():
        available_tickers.append({
            "ticker": ticker,
            "last_updated": metadata.get("last_updated", "unknown"),
            "metrics": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in metadata.get("metrics", {}).items()
            }
        })
    
    return {"tickers": available_tickers}

@app.post("/api/v1/predict", 
    summary="Predict stock price",
    response_description="Stock price prediction with quantiles and regime information"
)
@limiter.limit("30/minute")
async def predict_stock(request: Request, pred_request: PredictionRequest):
    """
    Predict stock price for the specified ticker and horizon
    
    - **ticker**: Stock ticker symbol
    - **horizon**: Prediction horizon in days (1-30)
    - **include_quantiles**: Include prediction quantiles for uncertainty
    - **include_regime**: Include market regime information
    """
    if not model_pool:
        raise HTTPException(status_code=503, detail="Model pool not initialized")
    
    # Make prediction
    result = await model_pool.predict(
        pred_request.ticker,
        pred_request.horizon,
        pred_request.include_quantiles,
        pred_request.include_regime
    )
    
    # Handle error
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@app.get("/api/v1/historical/{ticker}", 
    summary="Get historical data",
    response_description="Historical price data for a ticker"
)
@limiter.limit("30/minute")
async def get_historical_data(
    request: Request,
    ticker: str,
    days: int = Query(30, ge=1, le=365),
    include_features: bool = Query(False)
):
    """
    Get historical price data for a specific ticker
    
    - **ticker**: Stock ticker symbol
    - **days**: Number of days of historical data (1-365)
    - **include_features**: Include engineered features
    """
    if not model_pool:
        raise HTTPException(status_code=503, detail="Model pool not initialized")
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    try:
        # Fetch data
        data = model_pool.data_pipeline.fetch_data(start_date, end_date)
        merged_data = model_pool.data_pipeline.merge_data(data)
        
        # Process ticker-specific data
        ticker_data = merged_data[merged_data["symbol"] == ticker].copy()
        
        if ticker_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")
        
        # Extract required fields
        ohlcv_data = ticker_data[['open', 'high', 'low', 'close', 'volume']].reset_index()
        ohlcv_data = ohlcv_data.rename(columns={'index': 'date'})
        
        # Format data for response
        formatted_data = []
        for _, row in ohlcv_data.iterrows():
            record = {
                "date": row['date'].strftime('%Y-%m-%d'),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row['volume'])
            }
            formatted_data.append(record)
        
        # Prepare response
        response = {
            "ticker": ticker,
            "data": formatted_data
        }
        
        # Add features if requested
        if include_features:
            # Generate features
            feature_data = model_pool.data_pipeline.process_ticker(merged_data, ticker)
            
            if not feature_data.empty:
                # Include only numeric features
                numeric_features = feature_data.select_dtypes(include=['number'])
                
                # Format for response (sample of top features)
                top_features = numeric_features.columns[:20]  # Limit to 20 features
                feature_list = []
                
                for date, row in numeric_features[top_features].iterrows():
                    feature_row = {
                        "date": date.strftime('%Y-%m-%d'),
                        "features": {
                            col: float(row[col]) for col in top_features
                        }
                    }
                    feature_list.append(feature_row)
                
                response["features"] = feature_list
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data: {str(e)}")

@app.get("/api/v1/health", 
    summary="API health check",
    response_description="Health status of the API"
)
async def health_check():
    """Check if the API is healthy and ready to serve requests"""
    if not model_pool:
        return {"status": "initializing"}
    
    return {
        "status": "healthy",
        "models_available": len(model_pool.available_models),
        "models_loaded": len(model_pool.loaded_models),
        "timestamp": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

# Run app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    uvicorn.run(
        "src.api.app:app", 
        host=host, 
        port=port, 
        reload=debug,
        log_level="info"
    )
