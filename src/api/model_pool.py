import torch
import os
import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
import numpy as np
import pickle

from src.models.transformer import HierarchicalStockTransformer
from src.data.pipeline import DataPipeline

logger = logging.getLogger(__name__)

class ModelPool:
    """
    Manages loading and serving of multiple models
    """
    def __init__(
        self, 
        model_dir: str = 'models/trained',
        max_models_in_memory: int = 10,
        cache_ttl: int = 300  # 5 minutes
    ):
        self.model_dir = model_dir
        self.max_models = max_models_in_memory
        self.cache_ttl = cache_ttl
        
        # Model metadata
        self.available_models = {}
        
        # Loaded models in memory
        self.loaded_models = {}
        
        # Model usage tracking for LRU eviction
        self.model_last_used = {}
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_timestamps = {}
        
        # Data pipeline for fetching latest data
        self.data_pipeline = DataPipeline()
        
        # Lock for model loading
        self.model_locks = {}
    
    async def initialize(self):
        """Initialize model pool by scanning available models"""
        await self.scan_available_models()
    
    async def scan_available_models(self):
        """Scan model directory for available models"""
        if not os.path.exists(self.model_dir):
            logger.warning(f"Model directory {self.model_dir} does not exist")
            return
            
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pt')]
        
        for model_file in model_files:
            # Get ticker from filename
            ticker = model_file.split('_')[0] if '_' in model_file else model_file.replace('.pt', '')
            
            # Load model metadata
            try:
                metadata = self._load_model_metadata(os.path.join(self.model_dir, model_file))
                self.available_models[ticker] = {
                    'file_path': os.path.join(self.model_dir, model_file),
                    'last_updated': metadata.get('last_updated', 'unknown'),
                    'metrics': metadata.get('metrics', {}),
                    'config': metadata.get('model_config', {})
                }
                logger.info(f"Found model for {ticker}")
            except Exception as e:
                logger.error(f"Error loading metadata for {model_file}: {e}")
        
        logger.info(f"Discovered {len(self.available_models)} available models")
    
    def _load_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """Load model metadata without loading the full model"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            return {
                'last_updated': checkpoint.get('timestamp', 'unknown'),
                'metrics': checkpoint.get('metrics', {}),
                'model_config': checkpoint.get('model_config', {})
            }
        except Exception as e:
            logger.error(f"Error loading model metadata from {model_path}: {e}")
            return {}
    
    async def get_model(self, ticker: str) -> Optional[HierarchicalStockTransformer]:
        """
        Get model for a specific ticker, loading if necessary
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Model instance or None if model not available
        """
        if ticker not in self.available_models:
            logger.warning(f"No model available for ticker {ticker}")
            return None
        
        # If model already loaded, update usage time and return
        if ticker in self.loaded_models:
            self.model_last_used[ticker] = datetime.now()
            return self.loaded_models[ticker]
        
        # If we already have a lock for this model, wait for it
        if ticker in self.model_locks:
            async with self.model_locks[ticker]:
                # Check again after acquiring lock in case another task loaded it
                if ticker in self.loaded_models:
                    self.model_last_used[ticker] = datetime.now()
                    return self.loaded_models[ticker]
        else:
            # Create a lock for this model
            self.model_locks[ticker] = asyncio.Lock()
        
        # Load the model
        try:
            async with self.model_locks[ticker]:
                # Make space if needed
                if len(self.loaded_models) >= self.max_models:
                    self._evict_least_recently_used()
                
                # Load model
                model_path = self.available_models[ticker]['file_path']
                model = await self._load_model(model_path)
                
                # Store model
                self.loaded_models[ticker] = model
                self.model_last_used[ticker] = datetime.now()
                
                return model
        except Exception as e:
            logger.error(f"Error loading model for {ticker}: {e}")
            return None
    
    async def _load_model(self, model_path: str) -> HierarchicalStockTransformer:
        """Load model from file"""
        # Run this in a separate thread to not block the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._load_model_sync, model_path)
    
    def _load_model_sync(self, model_path: str) -> HierarchicalStockTransformer:
        """Synchronous version of model loading for executor"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            model_config = checkpoint.get('model_config', {})
            model = HierarchicalStockTransformer(**model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            logger.info(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def _evict_least_recently_used(self):
        """Evict least recently used model from memory"""
        if not self.model_last_used:
            return
        
        # Find least recently used
        lru_ticker = min(self.model_last_used.items(), key=lambda x: x[1])[0]
        
        # Remove from memory
        logger.info(f"Evicting model for {lru_ticker} from memory")
        del self.loaded_models[lru_ticker]
        del self.model_last_used[lru_ticker]
    
    async def predict(
        self, 
        ticker: str, 
        horizon: int = 5,
        include_quantiles: bool = True,
        include_regime: bool = True,
        use_cached: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction for a ticker
        
        Args:
            ticker: Stock symbol
            horizon: Prediction horizon in days
            include_quantiles: Whether to include uncertainty quantiles
            include_regime: Whether to include regime information
            use_cached: Whether to use cached predictions
            
        Returns:
            Dictionary with prediction results
        """
        cache_key = f"{ticker}_{horizon}_{include_quantiles}_{include_regime}"
        
        # Check cache if enabled
        if use_cached and cache_key in self.prediction_cache:
            cache_time = self.cache_timestamps.get(cache_key)
            if cache_time and (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                logger.debug(f"Using cached prediction for {ticker} horizon={horizon}")
                return self.prediction_cache[cache_key]
        
        # Get latest data for prediction
        try:
            # Fetch most recent 100 days of data to ensure sufficient history
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - pd.Timedelta(days=100)).strftime('%Y-%m-%d')
            
            data = self.data_pipeline.fetch_data(start_date, end_date)
            merged_data = self.data_pipeline.merge_data(data)
            
            # Process ticker-specific data
            ticker_data = self.data_pipeline.process_ticker(merged_data, ticker)
            
            if ticker_data.empty:
                raise ValueError(f"No valid data available for ticker {ticker}")
            
        except Exception as e:
            logger.error(f"Error fetching data for prediction on {ticker}: {e}")
            return {
                "error": f"Failed to fetch data: {str(e)}",
                "ticker": ticker,
                "timestamp": datetime.now().isoformat()
            }
        
        # Get model
        model = await self.get_model(ticker)
        if model is None:
            return {
                "error": f"No model available for ticker {ticker}",
                "ticker": ticker,
                "timestamp": datetime.now().isoformat()
            }
        
        # Prepare data for prediction
        try:
            # Get feature columns from model
            checkpoint = torch.load(self.available_models[ticker]['file_path'], map_location='cpu')
            feature_cols = checkpoint.get('feature_cols', ticker_data.columns.tolist())
            
            # Use only available features
            avail_features = [f for f in feature_cols if f in ticker_data.columns]
            features = ticker_data[avail_features].values
            
            # Standardize if scaler is available
            scaler = checkpoint.get('scaler')
            if scaler:
                features = scaler.transform(features)
            
            # Get window size
            window_size = checkpoint.get('window_size', 60)
            
            # Use the most recent window for prediction
            if len(features) < window_size:
                raise ValueError(f"Insufficient data for {ticker}: need {window_size} rows, got {len(features)}")
            
            predict_window = features[-window_size:]
            predict_tensor = torch.FloatTensor(predict_window).unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            logger.error(f"Error preparing data for prediction on {ticker}: {e}")
            return {
                "error": f"Failed to prepare data: {str(e)}",
                "ticker": ticker,
                "timestamp": datetime.now().isoformat()
            }
        
        # Make prediction
        try:
            with torch.no_grad():
                output = model(predict_tensor)
            
            # Get point prediction
            point_pred = output["prediction"].cpu().numpy()[0, 0]
            
            # Get most recent price
            last_price = ticker_data['close'].iloc[-1]
            
            # Prepare result
            result = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "latest_price": float(last_price),
                "horizon": horizon,
                "prediction": float(point_pred),
                "change_pct": float((point_pred / last_price - 1) * 100)
            }
            
            # Add quantiles if requested
            if include_quantiles and "quantiles" in output:
                quantiles = output["quantiles"].cpu().numpy()[0]
                result["quantiles"] = {
                    "p10": float(quantiles[0]),
                    "p50": float(quantiles[1]),
                    "p90": float(quantiles[2])
                }
                
                # Add confidence interval
                result["confidence_interval"] = {
                    "lower": float(quantiles[0]),
                    "upper": float(quantiles[2]),
                    "width_pct": float((quantiles[2] - quantiles[0]) / last_price * 100)
                }
            
            # Add regime information if requested
            if include_regime and "regime_weights" in output:
                regime_weights = output["regime_weights"].cpu().numpy()[0]
                regime_labels = ["Bull", "Bear", "Consolidation"]
                
                # Get dominant regime
                dominant_idx = np.argmax(regime_weights)
                result["regime"] = {
                    "dominant": regime_labels[dominant_idx],
                    "confidence": float(regime_weights[dominant_idx]),
                    "distribution": {
                        regime_labels[i]: float(regime_weights[i]) 
                        for i in range(len(regime_labels))
                    }
                }
            
            # Cache the result
            self.prediction_cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction for {ticker}: {e}")
            return {
                "error": f"Prediction failed: {str(e)}",
                "ticker": ticker,
                "timestamp": datetime.now().isoformat()
            }
