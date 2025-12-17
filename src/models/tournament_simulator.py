"""Tournament simulator using Monte Carlo approach."""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.helpers import setup_logger, load_data, save_data
from src.utils.constants import (
    LOGISTIC_REGRESSION_MODEL, RANDOM_FOREST_MODEL, SCALER_FILE,
    FEATURE_NAMES_FILE, SIMULATION_RUNS, PROCESSED_MATCHES_CSV
)

logger = setup_logger(__name__)


class TournamentSimulator:
    """Simulates World Cup tournaments using trained models."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """Initialize tournament simulator.
        
        Args:
            model_type: 'logistic_regression' or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.matches_df = None
        self.team_stats = {}
        
    def load_model(self) -> bool:
        """Load trained model."""
        try:
            logger.info(f"Loading {self.model_type} model...")
            
            if self.model_type == 'random_forest':
                model_path = RANDOM_FOREST_MODEL
            else:
                model_path = LOGISTIC_REGRESSION_MODEL
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(SCALER_FILE, 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(FEATURE_NAMES_FILE, 'rb') as f:
                self.feature_names = pickle.load(f)
            
            logger.info(f"Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def load_matches_data(self) -> bool:
        """Load matches data for stats calculation."""
        try:
            self.matches_df = load_data(PROCESSED_MATCHES_CSV)
            return self.matches_df is not None
        except Exception as e:
            logger.error(f"Error loading matches data: {str(e)}")
            return False
    
    def calculate_match_probability(self, home_team: str, away_team: str, 
                                   neutral_ground: bool = True) -> float:
        """Calculate probability of home team winning.
        
        Args:
            home_team: Name of home team
            away_team: Name of away team
            neutral_ground: Whether match is on neutral ground
        
        Returns:
            Probability of home team winning (0-1)
        """
        try:
            # Get team stats
            home_stats = self._get_team_stats(home_team)
            away_stats = self._get_team_stats(away_team)
            
            # Create feature vector (simplified)
            features = self._create_feature_vector(home_team, away_team, home_stats, away_stats)
            
            if features is None:
                # Return default probability based on recent form
                home_win_rate = home_stats.get('win_rate', 0.4)
                return home_win_rate
            
            # Scale features
            features_scaled = self.scaler.transform([features])[0]
            
            # Get probability
            proba = self.model.predict_proba([features_scaled])[0]
            
            # Return probability of home win (index 1 or index 2 depending on classes)
            if len(proba) == 3:
                return proba[2]  # Home win
            elif len(proba) == 2:
                return proba[1]  # Home win
            else:
                return 0.5
        except Exception as e:
            logger.debug(f"Error calculating probability for {home_team} vs {away_team}: {str(e)}")
            return 0.5
    
    def simulate_match(self, home_team: str, away_team: str, 
                      neutral_ground: bool = True) -> Tuple[int, int]:
        """Simulate a single match and return goals.
        
        Args:
            home_team: Name of home team
            away_team: Name of away team
            neutral_ground: Whether match is on neutral ground
        
        Returns:
            Tuple of (home_goals, away_goals)
        """
        # Get match probability
        home_win_prob = self.calculate_match_probability(home_team, away_team, neutral_ground)
        
        # Simulate result
        rand = np.random.random()
        
        if rand < home_win_prob * 0.7:  # Home win
            home_goals = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
            away_goals = max(0, home_goals - np.random.choice([1, 2], p=[0.7, 0.3]))
        elif rand < home_win_prob * 0.7 + (1 - home_win_prob) * 0.7:  # Away win
            away_goals = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
            home_goals = max(0, away_goals - np.random.choice([1, 2], p=[0.7, 0.3]))
        else:  # Draw
            goals = np.random.choice([0, 1, 2, 3], p=[0.1, 0.4, 0.4, 0.1])
            home_goals = goals
            away_goals = goals
        
        return int(home_goals), int(away_goals)
    
    def simulate_tournament(self, teams: List[str], 
                           num_simulations: int = SIMULATION_RUNS) -> Dict[str, float]:
        """Simulate tournament multiple times and return win probabilities.
        
        Args:
            teams: List of team names
            num_simulations: Number of simulations to run
        
        Returns:
            Dictionary with team names as keys and win probability as values
        """
        logger.info(f"Simulating tournament with {len(teams)} teams, {num_simulations} iterations...")
        
        win_count = defaultdict(int)
        
        for sim in range(num_simulations):
            if sim % 100 == 0:
                logger.info(f"Simulation {sim} / {num_simulations}")
            
            # Simple group stage simulation (simplified)
            # In real implementation, this would simulate the actual bracket
            
            # For now, use matchday win probability
            winner = self._simulate_tournament_simple(teams)
            win_count[winner] += 1
        
        # Convert counts to probabilities
        probabilities = {
            team: count / num_simulations
            for team, count in win_count.items()
        }
        
        return probabilities
    
    def _simulate_tournament_simple(self, teams: List[str]) -> str:
        """Simplified tournament simulation (winner by aggregate performance)."""
        # Assign each team a strength score
        team_scores = {}
        for team in teams:
            stats = self._get_team_stats(team)
            strength = (stats.get('win_rate', 0.4) * 3 + 
                       stats.get('goal_difference', 0) / 10 +
                       np.random.normal(0, 0.5))
            team_scores[team] = strength
        
        # Winner is team with highest strength
        return max(team_scores, key=team_scores.get)
    
    def _get_team_stats(self, team: str) -> Dict:
        """Get team statistics."""
        if team in self.team_stats:
            return self.team_stats[team]
        
        try:
            df = self.matches_df
            
            home_matches = df[df['home_team'] == team]
            away_matches = df[df['away_team'] == team]
            
            # Calculate stats
            home_wins = (home_matches['home_score'] > home_matches['away_score']).sum()
            away_wins = (away_matches['away_score'] > away_matches['home_score']).sum()
            total_wins = home_wins + away_wins
            
            home_games = len(home_matches)
            away_games = len(away_matches)
            total_games = home_games + away_games
            
            home_goals_for = home_matches['home_score'].sum()
            away_goals_for = away_matches['away_score'].sum()
            total_goals_for = home_goals_for + away_goals_for
            
            home_goals_against = home_matches['away_score'].sum()
            away_goals_against = away_matches['home_score'].sum()
            total_goals_against = home_goals_against + away_goals_against
            
            stats = {
                'team': team,
                'total_games': total_games,
                'total_wins': total_wins,
                'win_rate': total_wins / total_games if total_games > 0 else 0.4,
                'goals_for': total_goals_for,
                'goals_against': total_goals_against,
                'goal_difference': total_goals_for - total_goals_against,
                'avg_goals_for': total_goals_for / total_games if total_games > 0 else 1.5,
            }
            
            self.team_stats[team] = stats
            return stats
        except:
            return {
                'team': team,
                'total_games': 0,
                'win_rate': 0.4,
                'goal_difference': 0,
                'avg_goals_for': 1.5,
            }
    
    def _create_feature_vector(self, home_team: str, away_team: str,
                              home_stats: Dict, away_stats: Dict) -> List[float]:
        """Create feature vector for match prediction."""
        try:
            features = []
            for fname in self.feature_names:
                if fname.startswith('home_'):
                    key = fname.replace('home_', '')
                    if key in home_stats:
                        features.append(home_stats[key])
                    else:
                        features.append(0.0)
                elif fname.startswith('away_'):
                    key = fname.replace('away_', '')
                    if key in away_stats:
                        features.append(away_stats[key])
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)
            
            return features if len(features) == len(self.feature_names) else None
        except:
            return None


def main():
    """Main function for tournament simulation."""
    logger.info("Tournament simulator initialized")
    simulator = TournamentSimulator(model_type='random_forest')
    
    if simulator.load_model() and simulator.load_matches_data():
        logger.info("Simulator ready to simulate tournaments")
        return True
    else:
        logger.error("Failed to initialize simulator")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
