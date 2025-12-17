"""Download World Cup prediction datasets from Kaggle."""

import os
import sys
import logging
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.constants import RAW_DATA_DIR, KAGGLE_DATASET
from src.utils.helpers import setup_logger, ensure_directory_exists

logger = setup_logger(__name__)


def check_kaggle_api() -> bool:
    """Check if Kaggle API is installed and credentials exist."""
    try:
        import kaggle
        kaggle_dir = Path.home() / '.kaggle'
        credentials = kaggle_dir / 'kaggle.json'
        
        if not credentials.exists():
            logger.error(f"Kaggle credentials not found at {credentials}")
            logger.error("Please download kaggle.json from https://www.kaggle.com/settings/account")
            logger.error(f"and place it in {kaggle_dir}")
            return False
        
        # Check permissions (should be 600)
        if os.name != 'nt':  # Not Windows
            mode = oct(credentials.stat().st_mode)[-3:]
            if mode != '600':
                logger.warning(f"Kaggle credentials have incorrect permissions: {mode}")
                logger.warning(f"Run: chmod 600 {credentials}")
        
        return True
    except ImportError:
        logger.error("Kaggle API not installed. Run: pip install kaggle")
        return False


def download_dataset() -> bool:
    """Download dataset from Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        logger.info(f"Downloading {KAGGLE_DATASET}...")
        ensure_directory_exists(RAW_DATA_DIR)
        
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=RAW_DATA_DIR, unzip=True)
        
        logger.info(f"Dataset downloaded to {RAW_DATA_DIR}")
        return True
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        return False


def verify_downloaded_files() -> bool:
    """Verify that all expected files are present."""
    expected_files = [
        'teams_form.csv',
        'matches.csv',
        'world_cup_matches.csv',
        'team_ratings.csv'
    ]
    
    missing_files = []
    for filename in expected_files:
        filepath = os.path.join(RAW_DATA_DIR, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
        else:
            logger.info(f"✓ Found {filename}")
    
    if missing_files:
        logger.warning(f"Missing files: {missing_files}")
        return False
    
    logger.info("All expected files are present!")
    return True


def main():
    """Main function to download data."""
    logger.info("Starting data download...")
    
    # Check Kaggle API
    if not check_kaggle_api():
        logger.error("Kaggle API setup failed")
        return False
    
    # Download dataset
    if not download_dataset():
        logger.error("Dataset download failed")
        return False
    
    # Verify files
    if not verify_downloaded_files():
        logger.warning("Some files are missing")
        return False
    
    logger.info("Data download completed successfully!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
