import argparse
import json
import logging
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mlops.training_pipeline import TrainingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train stock prediction models')
    
    parser.add_argument('--tickers', type=str, required=True,
                        help='Comma-separated list of tickers or path to ticker file')
    
    parser.add_argument('--start-date', type=str,
                        help='Start date (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date (YYYY-MM-DD), defaults to today')
    
    parser.add_argument('--days', type=int, default=730,
                        help='Number of days of historical data (used if start-date not provided)')
    
    parser.add_argument('--config', type=str, default='config/training_config.json',
                        help='Path to training configuration file')
    
    parser.add_argument('--output', type=str, default='training_results.json',
                        help='Path to output results file')
    
    parser.add_argument('--model-dir', type=str, default='models/trained',
                        help='Directory to save trained models')
    
    parser.add_argument('--mlflow-uri', type=str, default='sqlite:///mlflow.db',
                        help='MLflow tracking URI')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Determine tickers list
    if "," in args.tickers:
        # Comma-separated list
        tickers = [ticker.strip() for ticker in args.tickers.split(",")]
    elif args.tickers.endswith(('.txt', '.csv')):
        # File with tickers
        if os.path.exists(args.tickers):
            with open(args.tickers, 'r') as f:
                tickers = [line.strip() for line in f.readlines() if line.strip()]
        else:
            logger.error(f"Ticker file {args.tickers} not found")
            sys.exit(1)
    else:
        # Single ticker
        tickers = [args.tickers]
    
    logger.info(f"Processing {len(tickers)} tickers: {tickers}")
    
    # Determine date range
    end_date = args.end_date
    
    if args.start_date:
        start_date = args.start_date
    else:
        # Calculate start date based on days
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - 
                     timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Load configuration
    training_config = {}
    model_config = {}
    
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                
            training_config = config.get('training', {})
            model_config = config.get('model', {})
            logger.info(f"Loaded configuration from {args.config}")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading config file: {e}")
            logger.warning("Using default configuration")
    else:
        logger.warning(f"Config file {args.config} not found, using default configuration")
    
    # Initialize and run training pipeline
    pipeline = TrainingPipeline(
        mlflow_tracking_uri=args.mlflow_uri,
        model_dir=args.model_dir,
        config_path=args.config if os.path.exists(args.config) else None
    )
    
    try:
        results = pipeline.run(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            training_config=training_config,
            model_config=model_config
        )
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Training results saved to {args.output}")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Total tickers: {len(tickers)}")
        print(f"Successful: {results['success_count']}")
        print(f"Skipped: {results['skipped_count']}")
        print(f"Errors: {results['error_count']}")
        print("="*50)
        
        # Print detailed results for each ticker
        print("\nRESULTS BY TICKER:")
        for ticker, result in results['ticker_results'].items():
            status = result['status']
            if status == 'success':
                print(f"{ticker}: SUCCESS (Val Loss: {result['val_loss']:.4f}, Val MAE: {result['val_mae']:.4f})")
            elif status == 'skipped':
                print(f"{ticker}: SKIPPED ({result.get('reason', 'unknown reason')})")
            else:
                print(f"{ticker}: ERROR ({result.get('message', 'unknown error')})")
        
        # Exit with appropriate code
        if results['error_count'] > 0:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
