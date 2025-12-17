"""Utility helper functions for data processing and analysis."""

import logging
import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from src.utils.constants import LOG_FORMAT, LOG_LEVEL

# Configure logging
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def setup_logger(name: str) -> logging.Logger:
    """Setup logger for a module."""
    return logging.getLogger(name)


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data with error handling."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {filepath}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        return None


def save_data(df: pd.DataFrame, filepath: str, index: bool = False) -> bool:
    """Save DataFrame to CSV with error handling."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=index)
        logger.info(f"Saved data to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving to {filepath}: {str(e)}")
        return False


def print_data_summary(df: pd.DataFrame, name: str = "Dataset") -> None:
    """Print summary statistics of a DataFrame."""
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nBasic statistics:")
    print(df.describe())
    print(f"{'='*60}\n")


def normalize_team_names(df: pd.DataFrame, team_columns: List[str]) -> pd.DataFrame:
    """Normalize team names for consistency."""
    # Mapping dictionary for team name variations
    team_name_mapping = {
        'United States': 'USA',
        'USA': 'USA',
        'South Korea': 'South Korea',
        'Korea': 'South Korea',
        'Korea Republic': 'South Korea',
        'Czech Republic': 'Czechia',
        'England': 'England',
        'Scotland': 'Scotland',
        'Wales': 'Wales',
        'Northern Ireland': 'Northern Ireland',
        'Republic of Ireland': 'Ireland',
        'Iran': 'Iran',
        'Vietnam': 'Vietnam',
        'Hong Kong': 'Hong Kong',
        'Trinidad and Tobago': 'Trinidad and Tobago',
        'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
        'Bosnia-Herzegovina': 'Bosnia-Herzegovina',
    }
    
    for col in team_columns:
        if col in df.columns:
            df[col] = df[col].str.strip()
            df[col] = df[col].replace(team_name_mapping)
    
    return df


def calculate_team_stats(df: pd.DataFrame, team_col: str, goals_for_col: str, 
                        goals_against_col: str) -> Dict[str, Dict]:
    """Calculate team statistics."""
    stats = {}
    
    for team in df[team_col].unique():
        team_data = df[df[team_col] == team]
        
        stats[team] = {
            'matches_played': len(team_data),
            'goals_for': team_data[goals_for_col].sum(),
            'goals_against': team_data[goals_against_col].sum(),
            'goal_difference': team_data[goals_for_col].sum() - team_data[goals_against_col].sum(),
            'avg_goals_for': team_data[goals_for_col].mean(),
            'avg_goals_against': team_data[goals_against_col].mean(),
        }
    
    return stats


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def ensure_directory_exists(directory: str) -> None:
    """Ensure directory exists."""
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Directory ensured: {directory}")


def get_season_from_date(date_str: str) -> int:
    """Extract season/year from date string."""
    try:
        if isinstance(date_str, str):
            year = int(date_str.split('-')[0])
        else:
            year = int(date_str.year)
        return year
    except:
        return None


def calculate_elo_rating(rating1: float, rating2: float, score1: float, 
                        k_factor: float = 32) -> Tuple[float, float]:
    """Calculate Elo rating change for two teams.
    
    Args:
        rating1: Current rating of team 1
        rating2: Current rating of team 2
        score1: Result for team 1 (1.0=win, 0.5=draw, 0.0=loss)
        k_factor: K-factor (32 for most cases)
    
    Returns:
        Tuple of new ratings for both teams
    """
    expected1 = 1 / (1 + 10**((rating2 - rating1) / 400))
    expected2 = 1 - expected1
    
    score2 = 1 - score1 if score1 != 0.5 else 0.5
    
    new_rating1 = rating1 + k_factor * (score1 - expected1)
    new_rating2 = rating2 + k_factor * (score2 - expected2)
    
    return new_rating1, new_rating2


def get_match_result(goals_for: int, goals_against: int) -> str:
    """Convert goal difference to match result."""
    if goals_for > goals_against:
        return 'W'
    elif goals_for < goals_against:
        return 'L'
    else:
        return 'D'


def calculate_points(result: str) -> int:
    """Calculate points from match result."""
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0


def format_probability(prob: float) -> str:
    """Format probability as percentage string."""
    return f"{prob*100:.2f}%"


if __name__ == "__main__":
    logger.info("Helpers module loaded successfully")
