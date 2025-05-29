{
  "model": {
    "feature_dim": 0,
    "d_model": 256,
    "nhead": 8,
    "num_layers": 6,
    "d_ff": 1024,
    "dropout": 0.1,
    "num_regimes": 3,
    "num_quantiles": 3
  },
  "training": {
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 0.0005,
    "weight_decay": 1e-5,
    "window_size": 60,
    "horizon": 5,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "early_stop_patience": 10,
    "lr_patience": 5,
    "grad_clip": 1.0,
    "quantile_levels": [0.1, 0.5, 0.9],
    "scale_features": true
  }
}
