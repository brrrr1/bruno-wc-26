"""Feature engineering for machine learning models."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.helpers import setup_logger, load_data, save_data
from src.utils.constants import PROCESSED_MATCHES_CSV, TRAINING_DATA_CSV, RECENT_FORM_MATCHES
from src.data.data_processor import DataProcessor

logger = setup_logger(__name__)


class FeatureEngineer:
    """Class to handle feature engineering."""
    
    def __init__(self, matches_df: pd.DataFrame = None):
        """Initialize feature engineer."""
        self.matches_df = matches_df
        self.features_df = None
        
    def load_matches_data(self) -> bool:
        """Load processed matches data."""
        try:
            self.matches_df = load_data(PROCESSED_MATCHES_CSV)
            return self.matches_df is not None
        except Exception as e:
            logger.error(f"Error loading matches data: {str(e)}")
            return False
    
    def calculate_team_stats(self, team: str, up_to_date: pd.Timestamp) -> dict:
        """Calculate team statistics up to a specific date."""
        df = self.matches_df[self.matches_df['date'] < up_to_date].copy()
        
        home_matches = df[df['home_team'] == team]
        away_matches = df[df['away_team'] == team]
        
        # Home stats
        home_goals_for = home_matches['home_score'].sum()
        home_goals_against = home_matches['away_score'].sum()
        home_wins = (home_matches['home_score'] > home_matches['away_score']).sum()
        home_draws = (home_matches['home_score'] == home_matches['away_score']).sum()
        home_losses = (home_matches['home_score'] < home_matches['away_score']).sum()
        home_games = len(home_matches)
        
        # Away stats
        away_goals_for = away_matches['away_score'].sum()
        away_goals_against = away_matches['home_score'].sum()
        away_wins = (away_matches['away_score'] > away_matches['home_score']).sum()
        away_draws = (away_matches['away_score'] == away_matches['home_score']).sum()
        away_losses = (away_matches['away_score'] < away_matches['home_score']).sum()
        away_games = len(away_matches)
        
        # Combined stats
        total_games = home_games + away_games
        total_wins = home_wins + away_wins
        total_draws = home_draws + away_draws
        total_losses = home_losses + away_losses
        total_goals_for = home_goals_for + away_goals_for
        total_goals_against = home_goals_against + away_goals_against
        
        if total_games == 0:
            return None
        
        return {
            'team': team,
            'total_games': total_games,
            'total_wins': total_wins,
            'total_draws': total_draws,
            'total_losses': total_losses,
            'win_rate': total_wins / total_games if total_games > 0 else 0,
            'draw_rate': total_draws / total_games if total_games > 0 else 0,
            'loss_rate': total_losses / total_games if total_games > 0 else 0,
            'goals_for': total_goals_for,
            'goals_against': total_goals_against,
            'goal_difference': total_goals_for - total_goals_against,
            'avg_goals_for': total_goals_for / total_games if total_games > 0 else 0,
            'avg_goals_against': total_goals_against / total_games if total_games > 0 else 0,
            'home_wins': home_wins,
            'home_draws': home_draws,
            'home_losses': home_losses,
            'away_wins': away_wins,
            'away_draws': away_draws,
            'away_losses': away_losses,
            'home_goals_for': home_goals_for,
            'home_goals_against': home_goals_against,
            'away_goals_for': away_goals_for,
            'away_goals_against': away_goals_against,
        }
    
    def get_recent_form(self, team: str, up_to_date: pd.Timestamp, num_matches: int = RECENT_FORM_MATCHES) -> dict:
        """Get recent form for a team."""
        df = self.matches_df[self.matches_df['date'] < up_to_date].copy()
        
        home_matches = df[df['home_team'] == team].tail(num_matches)
        away_matches = df[df['away_team'] == team].tail(num_matches)
        recent_matches = pd.concat([home_matches, away_matches]).sort_values('date').tail(num_matches)
        
        if len(recent_matches) == 0:
            return None
        
        recent_wins = 0
        recent_draws = 0
        recent_losses = 0
        recent_goals_for = 0
        recent_goals_against = 0
        
        for _, row in recent_matches.iterrows():
            if row['home_team'] == team:
                if row['home_score'] > row['away_score']:
                    recent_wins += 1
                elif row['home_score'] == row['away_score']:
                    recent_draws += 1
                else:
                    recent_losses += 1
                recent_goals_for += row['home_score']
                recent_goals_against += row['away_score']
            else:
                if row['away_score'] > row['home_score']:
                    recent_wins += 1
                elif row['away_score'] == row['home_score']:
                    recent_draws += 1
                else:
                    recent_losses += 1
                recent_goals_for += row['away_score']
                recent_goals_against += row['home_score']
        
        recent_games = len(recent_matches)
        
        return {
            'recent_games': recent_games,
            'recent_wins': recent_wins,
            'recent_draws': recent_draws,
            'recent_losses': recent_losses,
            'recent_win_rate': recent_wins / recent_games if recent_games > 0 else 0,
            'recent_goals_for': recent_goals_for,
            'recent_goals_against': recent_goals_against,
            'recent_goal_difference': recent_goals_for - recent_goals_against,
            'recent_avg_goals_for': recent_goals_for / recent_games if recent_games > 0 else 0,
            'recent_avg_goals_against': recent_goals_against / recent_games if recent_games > 0 else 0,
        }
    
    def get_head_to_head(self, team1: str, team2: str, up_to_date: pd.Timestamp) -> dict:
        """Get head-to-head stats between two teams."""
        df = self.matches_df[self.matches_df['date'] < up_to_date].copy()
        
        h2h = df[((df['home_team'] == team1) & (df['away_team'] == team2)) |
                 ((df['home_team'] == team2) & (df['away_team'] == team1))]
        
        if len(h2h) == 0:
            return {
                'h2h_games': 0,
                'h2h_team1_wins': 0,
                'h2h_draws': 0,
                'h2h_team2_wins': 0,
                'h2h_team1_goals_for': 0,
                'h2h_team1_goals_against': 0,
            }
        
        team1_wins = 0
        team2_wins = 0
        draws = 0
        team1_goals_for = 0
        team1_goals_against = 0
        
        for _, row in h2h.iterrows():
            if row['home_team'] == team1:
                team1_goals_for += row['home_score']
                team1_goals_against += row['away_score']
                if row['home_score'] > row['away_score']:
                    team1_wins += 1
                elif row['home_score'] == row['away_score']:
                    draws += 1
                else:
                    team2_wins += 1
            else:
                team1_goals_for += row['away_score']
                team1_goals_against += row['home_score']
                if row['away_score'] > row['home_score']:
                    team1_wins += 1
                elif row['away_score'] == row['home_score']:
                    draws += 1
                else:
                    team2_wins += 1
        
        return {
            'h2h_games': len(h2h),
            'h2h_team1_wins': team1_wins,
            'h2h_draws': draws,
            'h2h_team2_wins': team2_wins,
            'h2h_team1_goals_for': team1_goals_for,
            'h2h_team1_goals_against': team1_goals_against,
        }
    
    def create_match_features(self, match_row: pd.Series, up_to_date: pd.Timestamp) -> dict:
        """Create features for a match."""
        home_team = match_row['home_team']
        away_team = match_row['away_team']
        home_score = match_row['home_score']
        away_score = match_row['away_score']
        
        # Determine result (1=home win, 0.5=draw, 0=away win)
        if home_score > away_score:
            result = 1
        elif home_score == away_score:
            result = 0.5
        else:
            result = 0
        
        # Calculate stats for both teams
        home_stats = self.calculate_team_stats(home_team, up_to_date)
        away_stats = self.calculate_team_stats(away_team, up_to_date)
        
        home_form = self.get_recent_form(home_team, up_to_date)
        away_form = self.get_recent_form(away_team, up_to_date)
        
        h2h = self.get_head_to_head(home_team, away_team, up_to_date)
        
        if home_stats is None or away_stats is None:
            return None
        
        features = {
            'home_team': home_team,
            'away_team': away_team,
            'date': match_row['date'],
            'result': result,  # Target variable
        }
        
        # Add home team stats
        for key, value in home_stats.items():
            features[f'home_{key}'] = value
        
        # Add away team stats
        for key, value in away_stats.items():
            features[f'away_{key}'] = value
        
        # Add home team recent form
        if home_form:
            for key, value in home_form.items():
                features[f'home_{key}'] = value
        
        # Add away team recent form
        if away_form:
            for key, value in away_form.items():
                features[f'away_{key}'] = value
        
        # Add difference features
        features['win_rate_diff'] = features.get('home_total_wins', 0) - features.get('away_total_wins', 0)
        features['goal_diff_diff'] = features.get('home_goal_difference', 0) - features.get('away_goal_difference', 0)
        features['avg_goals_diff'] = features.get('home_avg_goals_for', 0) - features.get('away_avg_goals_for', 0)
        
        # Add h2h stats
        for key, value in h2h.items():
            features[key] = value
        
        return features
    
    def create_training_data(self) -> bool:
        """Create training dataset with all features."""
        try:
            logger.info("Creating training data with features...")
            
            if self.matches_df is None:
                if not self.load_matches_data():
                    return False
            
            df = self.matches_df.copy()
            df = df.sort_values('date').reset_index(drop=True)
            
            features_list = []
            for idx, row in df.iterrows():
                if idx % 100 == 0:
                    logger.info(f"Processing match {idx} / {len(df)}")
                
                features = self.create_match_features(row, row['date'])
                if features:
                    features_list.append(features)
            
            self.features_df = pd.DataFrame(features_list)
            
            # Save training data
            save_data(self.features_df, TRAINING_DATA_CSV)
            logger.info(f"Training data created: {len(self.features_df)} matches, {len(self.features_df.columns)} features")
            
            return True
        except Exception as e:
            logger.error(f"Error creating training data: {str(e)}")
            return False


def main():
    """Main function to create features."""
    logger.info("Starting feature engineering...")
    
    engineer = FeatureEngineer()
    if engineer.create_training_data():
        logger.info("Feature engineering completed successfully!")
        return True
    else:
        logger.error("Feature engineering failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
