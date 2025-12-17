"""Main execution script for the World Cup prediction pipeline."""

import sys
import logging
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import setup_logger
from src.utils.constants import LOG_FORMAT, LOG_LEVEL
from src.data.download_data import main as download_main
from src.data.data_processor import main as process_main
from src.data.feature_engineering import main as feature_main
from src.models.train_model import main as train_main
from src.models.predict_2022 import main as predict_2022_main
from src.models.predict_2026 import main as predict_2026_main

# Configure logging
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = setup_logger(__name__)


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")


def run_pipeline(steps: list = None):
    """Run the complete pipeline or selected steps.
    
    Args:
        steps: List of steps to run. If None, runs all steps.
                Options: 'download', 'process', 'features', 'train', 'predict_2022', 'predict_2026'
    """
    if steps is None:
        steps = ['download', 'process', 'features', 'train', 'predict_2022', 'predict_2026']
    
    print_header("⚽ FIFA WORLD CUP 2026 PREDICTION")
    logger.info("Starting pipeline execution...")
    logger.info(f"Steps to execute: {', '.join(steps)}")
    
    try:
        # Step 1: Download Data
        if 'download' in steps:
            print_header("STEP 1: DOWNLOADING DATA")
            if not download_main():
                logger.error("Data download failed. Continuing with existing data...")
        
        # Step 2: Process Data
        if 'process' in steps:
            print_header("STEP 2: PROCESSING DATA")
            if not process_main():
                logger.error("Data processing failed!")
                return False
        
        # Step 3: Feature Engineering
        if 'features' in steps:
            print_header("STEP 3: FEATURE ENGINEERING")
            if not feature_main():
                logger.error("Feature engineering failed!")
                return False
        
        # Step 4: Train Models
        if 'train' in steps:
            print_header("STEP 4: TRAINING MODELS")
            if not train_main():
                logger.error("Model training failed!")
                return False
        
        # Step 5: Predict 2022 (Validation)
        if 'predict_2022' in steps:
            print_header("STEP 5: PREDICTING 2022 WORLD CUP (VALIDATION)")
            if not predict_2022_main():
                logger.error("2022 prediction failed!")
                return False
        
        # Step 6: Predict 2026
        if 'predict_2026' in steps:
            print_header("STEP 6: PREDICTING 2026 WORLD CUP")
            if not predict_2026_main():
                logger.error("2026 prediction failed!")
                return False
        
        print_header("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("All steps completed successfully!")
        print("\nNext steps:")
        print("1. View predictions in results/ folder")
        print("2. Run: streamlit run app/streamlit_app.py")
        print("3. Check visualizations and interactive simulations")
        print("\n")
        
        return True
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FIFA World Cup 2026 Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run full pipeline
  python main.py --steps download process # Run only download and process
  python main.py --steps predict_2026     # Run only 2026 prediction
        """
    )
    
    parser.add_argument(
        '--steps',
        nargs='+',
        help='Steps to execute',
        choices=['download', 'process', 'features', 'train', 'predict_2022', 'predict_2026'],
        default=None
    )
    
    parser.add_argument(
        '--web',
        action='store_true',
        help='Launch Streamlit web app after pipeline'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    success = run_pipeline(args.steps)
    
    if success and args.web:
        import subprocess
        logger.info("Launching Streamlit web app...")
        subprocess.run(["streamlit", "run", "app/streamlit_app.py"])
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
