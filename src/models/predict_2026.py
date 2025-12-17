"""Predict 2026 World Cup winners."""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.helpers import setup_logger, load_data, save_data
from src.utils.constants import (
    WC_2026_LIKELY_TEAMS, PREDICTIONS_2026_CSV, RANDOM_FOREST_MODEL,
    SCALER_FILE, FEATURE_NAMES_FILE
)
from src.models.tournament_simulator import TournamentSimulator

logger = setup_logger(__name__)


class Predictor2026:
    """Predict 2026 World Cup."""
    
    def __init__(self):
        """Initialize predictor."""
        self.simulator = TournamentSimulator(model_type='random_forest')
        self.predictions = {}
        
    def predict(self, num_simulations: int = 10000) -> bool:
        """Run predictions."""
        try:
            # Load model and data
            if not self.simulator.load_model():
                logger.error("Failed to load model")
                return False
            
            if not self.simulator.load_matches_data():
                logger.error("Failed to load match data")
                return False
            
            logger.info(f"Predicting 2026 World Cup with {num_simulations} simulations...")
            logger.info(f"Number of teams: {len(WC_2026_LIKELY_TEAMS)}")
            
            # Simulate tournament
            probabilities = self.simulator.simulate_tournament(WC_2026_LIKELY_TEAMS, num_simulations)
            
            # Sort by probability
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            
            logger.info("\n" + "="*70)
            logger.info("2026 WORLD CUP PREDICTIONS (48-TEAM FORMAT)")
            logger.info("="*70)
            logger.info(f"{'Rank':>4} {'Team':25s} {'Probability':>15s} {'Odds':>12s}")
            logger.info("-"*70)
            
            for rank, (team, prob) in enumerate(sorted_probs, 1):
                if prob > 0:
                    odds = 1 / prob
                    logger.info(f"{rank:4d} {team:25s} {prob*100:14.2f}% {'1 in ' + str(int(odds)):>12s}")
            
            logger.info("\n" + "="*70)
            logger.info("TOP 10 FAVORITES")
            logger.info("="*70)
            for rank, (team, prob) in enumerate(sorted_probs[:10], 1):
                logger.info(f"{rank:2d}. {team:25s} - {prob*100:6.2f}%")
            
            logger.info("\n" + "="*70)
            logger.info("KEY STATISTICS")
            logger.info("="*70)
            
            cumulative_prob = 0
            for i, (team, prob) in enumerate(sorted_probs[:5]):
                cumulative_prob += prob
            logger.info(f"Top 5 teams win probability: {cumulative_prob*100:.2f}%")
            
            cumulative_prob = 0
            for i, (team, prob) in enumerate(sorted_probs[:10]):
                cumulative_prob += prob
            logger.info(f"Top 10 teams win probability: {cumulative_prob*100:.2f}%")
            
            # Save predictions
            df = pd.DataFrame(sorted_probs, columns=['Team', 'Win_Probability'])
            df['Rank'] = range(1, len(df) + 1)
            df = df[['Rank', 'Team', 'Win_Probability']]
            save_data(df, PREDICTIONS_2026_CSV)
            
            logger.info(f"\nPredictions saved to {PREDICTIONS_2026_CSV}")
            
            self.predictions = dict(sorted_probs)
            return True
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return False


def main():
    """Main function."""
    logger.info("Starting 2026 World Cup prediction...")
    predictor = Predictor2026()
    return predictor.predict(num_simulations=10000)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
