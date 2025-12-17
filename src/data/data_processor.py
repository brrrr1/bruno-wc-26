"""Data processing and cleaning module."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.helpers import setup_logger, normalize_team_names, load_data, save_data
from src.utils.constants import (
    TEAMS_FORM_CSV, MATCHES_CSV, WORLD_CUP_MATCHES_CSV,
    PROCESSED_MATCHES_CSV, PROCESSED_WORLD_CUP_CSV, TRAINING_DATA_CSV
)

logger = setup_logger(__name__)


class DataProcessor:
    """Class to handle data processing and cleaning."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.matches_df = None
        self.teams_form_df = None
        self.world_cup_df = None
        self.processed_matches_df = None
        
    def load_raw_data(self) -> bool:
        """Load raw data from CSV files."""
        try:
            logger.info("Loading raw data...")
            
            self.matches_df = load_data(MATCHES_CSV)
            self.teams_form_df = load_data(TEAMS_FORM_CSV)
            self.world_cup_df = load_data(WORLD_CUP_MATCHES_CSV)
            
            if any(df is None for df in [self.matches_df, self.teams_form_df, self.world_cup_df]):
                logger.error("Failed to load some data files")
                return False
            
            logger.info("Raw data loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            return False
    
    def clean_matches_data(self) -> pd.DataFrame:
        """Clean matches data."""
        logger.info("Cleaning matches data...")
        df = self.matches_df.copy()
        
        # Remove rows with missing critical values
        df = df.dropna(subset=['date', 'home_team', 'away_team', 'home_score', 'away_score'])
        
        # Normalize team names
        df = normalize_team_names(df, ['home_team', 'away_team'])
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Remove duplicate rows
        df = df.drop_duplicates(subset=['date', 'home_team', 'away_team'])
        
        # Ensure score columns are integers
        df['home_score'] = df['home_score'].astype(int)
        df['away_score'] = df['away_score'].astype(int)
        
        logger.info(f"Matches data cleaned: {len(df)} rows remaining")
        return df
    
    def clean_world_cup_data(self) -> pd.DataFrame:
        """Clean World Cup data."""
        logger.info("Cleaning World Cup data...")
        df = self.world_cup_df.copy()
        
        # Remove rows with missing critical values
        df = df.dropna(subset=['date', 'home_team', 'away_team', 'home_score', 'away_score'])
        
        # Normalize team names
        df = normalize_team_names(df, ['home_team', 'away_team'])
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Ensure score columns are integers
        df['home_score'] = df['home_score'].astype(int)
        df['away_score'] = df['away_score'].astype(int)
        
        logger.info(f"World Cup data cleaned: {len(df)} rows remaining")
        return df
    
    def process_all(self) -> bool:
        """Process all data."""
        try:
            # Load raw data
            if not self.load_raw_data():
                return False
            
            # Clean data
            self.processed_matches_df = self.clean_matches_data()
            processed_wc = self.clean_world_cup_data()
            
            # Save processed data
            save_data(self.processed_matches_df, PROCESSED_MATCHES_CSV)
            save_data(processed_wc, PROCESSED_WORLD_CUP_CSV)
            
            logger.info("Data processing completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during data processing: {str(e)}")
            return False
    
    def get_matches_by_year(self, year: int) -> pd.DataFrame:
        """Get matches from a specific year."""
        if self.processed_matches_df is None:
            return pd.DataFrame()
        
        df = self.processed_matches_df.copy()
        df['year'] = df['date'].dt.year
        return df[df['year'] == year]
    
    def get_team_matches(self, team: str) -> pd.DataFrame:
        """Get all matches for a specific team."""
        if self.processed_matches_df is None:
            return pd.DataFrame()
        
        df = self.processed_matches_df.copy()
        df = df[(df['home_team'] == team) | (df['away_team'] == team)]
        return df.sort_values('date')
    
    def get_recent_form(self, team: str, matches: int = 10) -> pd.DataFrame:
        """Get recent form for a team."""
        team_matches = self.get_team_matches(team)
        return team_matches.tail(matches)
    
    def get_head_to_head(self, team1: str, team2: str) -> pd.DataFrame:
        """Get head-to-head records between two teams."""
        if self.processed_matches_df is None:
            return pd.DataFrame()
        
        df = self.processed_matches_df.copy()
        h2h = df[((df['home_team'] == team1) & (df['away_team'] == team2)) |
                 ((df['home_team'] == team2) & (df['away_team'] == team1))]
        return h2h.sort_values('date')


def main():
    """Main function to process data."""
    logger.info("Starting data processing...")
    
    processor = DataProcessor()
    if processor.process_all():
        logger.info("Data processing completed successfully!")
        return True
    else:
        logger.error("Data processing failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
