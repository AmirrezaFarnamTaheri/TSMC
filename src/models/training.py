import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import logging
import time
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

from src.models.transformer import HierarchicalStockTransformer

logger = logging.getLogger(__name__)

def prepare_data_for_training(
    df: pd.DataFrame, 
    target_col: str = 'close',
    window_size: int = 60,
    horizon: int = 5,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    scale_features: bool = True
) -> Dict[str, Any]:
    """
    Prepare time series data for training
    
    Args:
        df: DataFrame with features and target
        target_col: Column to predict
        window_size: Size of the sliding window (sequence length)
        horizon: Prediction horizon (days ahead)
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        scale_features: Whether to standardize features
        
    Returns:
        Dictionary with training data components
    """
    # Make sure the target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Create target values (future prices)
    df = df.copy()
    df['target'] = df[target_col].shift(-horizon)
    
    # Remove rows with NaN in target
    df = df.dropna(subset=['target'])
    
    # Separate features and target
    exclude_cols = ['target']
    if 'symbol' in df.columns:
        exclude_cols.append('symbol')
    
    features = df.drop(columns=exclude_cols).select_dtypes(['number']).fillna(0)
    feature_cols = features.columns.tolist()
    target = df['target'].values
    
    logger.info(f"Prepared {len(feature_cols)} features for training")
    
    # Scale features
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = features.values
    
    # Create sequences
    X, y = [], []
    for i in range(len(features_scaled) - window_size):
        X.append(features_scaled[i:i+window_size])
        y.append(target[i+window_size])
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Created {len(X)} sequences with window size {window_size}")
    
    # Train/val/test split
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'window_size': window_size,
        'horizon': horizon,
        'n_features': X.shape[2]
    }

def quantile_loss(preds: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    """Calculate quantile loss"""
    error = target - preds
    return torch.max(quantile * error, (quantile - 1) * error).mean()

def train_model(
    train_data: Dict[str, Any],
    model_config: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    model_dir: str = 'models/trained',
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train the hierarchical stock transformer model
    
    Args:
        train_data: Dictionary with training data components
        model_config: Model configuration
        training_config: Training configuration
        model_dir: Directory to save model
        device: Device to use for training
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Training on device: {device}")
    
    # Default configurations
    default_model_config = {
        'feature_dim': train_data['n_features'],
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'dropout': 0.1,
        'max_seq_len': train_data['window_size'],
        'num_regimes': 3,
        'num_quantiles': 3
    }
    
    default_training_config = {
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 0.0005,
        'weight_decay': 1e-5,
        'lr_patience': 5,
        'early_stop_patience': 10,
        'grad_clip': 1.0,
        'quantile_levels': [0.1, 0.5, 0.9]  # 10%, 50%, 90%
    }
    
    # Merge with provided configs
    model_config = {**default_model_config, **(model_config or {})}
    training_config = {**default_training_config, **(training_config or {})}
    
    # Create model
    model = HierarchicalStockTransformer(**model_config).to(device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_data['train_dataset'], 
        batch_size=training_config['batch_size'], 
        shuffle=True
    )
    
    val_loader = DataLoader(
        train_data['val_dataset'], 
        batch_size=training_config['batch_size']
    )
    
    # Define loss functions
    mse_loss = nn.MSELoss()
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        'min', 
        patience=training_config['lr_patience'],
        factor=0.5,
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_quantile_loss': [],
        'learning_rates': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    best_model_state = None
    early_stop_counter = 0
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(training_config['epochs']):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Main prediction loss
            point_pred = output["prediction"]
            pred_loss = mse_loss(point_pred, target)
            
            # Quantile loss
            quantile_preds = output["quantiles"]
            q_loss = 0
            for i, q in enumerate(training_config['quantile_levels']):
                q_loss += quantile_loss(quantile_preds[:, i].unsqueeze(1), target, q)
            
            # Combine losses
            loss = pred_loss + 0.2 * q_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if training_config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    training_config['grad_clip']
                )
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Progress logging
            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch+1}/{training_config['epochs']} "
                            f"[{batch_idx+1}/{len(train_loader)}] "
                            f"Loss: {loss.item():.6f}")
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_mae = 0
        val_q_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Calculate losses
                point_pred = output["prediction"]
                val_loss += mse_loss(point_pred, target).item()
                val_mae += torch.abs(point_pred - target).mean().item()
                
                # Quantile loss
                quantile_preds = output["quantiles"]
                batch_q_loss = 0
                for i, q in enumerate(training_config['quantile_levels']):
                    batch_q_loss += quantile_loss(
                        quantile_preds[:, i].unsqueeze(1), target, q
                    ).item()
                
                val_q_loss += batch_q_loss
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_q_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log epoch results
        elapsed_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{training_config['epochs']} completed in {elapsed_time:.2f}s")
        logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_quantile_loss'].append(val_q_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Check for best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
            
            # Save best model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_mae': val_mae,
                'history': history,
                'model_config': model_config,
                'training_config': training_config,
                'feature_cols': train_data['feature_cols']
            }, os.path.join(model_dir, 'best_model.pt'))
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= training_config['early_stop_patience']:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'model_config': model_config,
        'training_config': training_config,
        'feature_cols': train_data['feature_cols'],
        'scaler': train_data['scaler']
    }, os.path.join(model_dir, 'final_model.pt'))
    
    return model, history

def evaluate_model(
    model: nn.Module,
    test_data: Dict[str, Any],
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Evaluate model performance on test data
    
    Args:
        model: Trained model
        test_data: Dictionary with test data components
        device: Device to use for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    test_loader = DataLoader(
        test_data['test_dataset'],
        batch_size=64,
        shuffle=False
    )
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            preds = output["prediction"]
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_targets)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((all_targets - all_preds) / all_targets)) * 100
    
    # Direction accuracy
    actual_direction = np.sign(np.diff(all_targets.flatten()))
    pred_direction = np.sign(np.diff(all_preds.flatten()))
    direction_accuracy = np.mean(actual_direction == pred_direction)
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'direction_accuracy': float(direction_accuracy)
    }
