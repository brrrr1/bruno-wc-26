"""Predict 2022 World Cup results for model validation."""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.helpers import setup_logger, load_data, save_data
from src.utils.constants import (
    WC_2022_TEAMS, PREDICTIONS_2022_CSV, RANDOM_FOREST_MODEL,
    SCALER_FILE, FEATURE_NAMES_FILE, PROCESSED_MATCHES_CSV
)
from src.models.tournament_simulator import TournamentSimulator

logger = setup_logger(__name__)

# Actual 2022 World Cup winner and results
WC_2022_RESULTS = {
    'winner': 'Argentina',
    'runner_up': 'France',
    'third': 'Morocco',
    'fourth': 'Croatia'
}


class Predictor2022:
    """Predict 2022 World Cup for validation."""
    
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
            
            logger.info(f"Predicting 2022 World Cup with {num_simulations} simulations...")
            
            # Simulate tournament
            probabilities = self.simulator.simulate_tournament(WC_2022_TEAMS, num_simulations)
            
            # Sort by probability
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            
            logger.info("\n" + "="*60)
            logger.info("2022 WORLD CUP PREDICTIONS")
            logger.info("="*60)
            
            for rank, (team, prob) in enumerate(sorted_probs, 1):
                logger.info(f"{rank:2d}. {team:25s} - {prob*100:6.2f}%")
            
            logger.info("\n" + "="*60)
            logger.info("ACTUAL 2022 RESULTS")
            logger.info("="*60)
            for key, team in WC_2022_RESULTS.items():
                logger.info(f"{key:15s}: {team}")
            
            # Check predictions
            predicted_winner = sorted_probs[0][0]
            actual_winner = WC_2022_RESULTS['winner']
            
            logger.info("\n" + "="*60)
            logger.info("VALIDATION")
            logger.info("="*60)
            logger.info(f"Predicted Winner: {predicted_winner}")
            logger.info(f"Actual Winner: {actual_winner}")
            logger.info(f"Correct: {'YES' if predicted_winner == actual_winner else 'NO'}")
            
            # Save predictions
            df = pd.DataFrame(sorted_probs, columns=['Team', 'Win_Probability'])
            save_data(df, PREDICTIONS_2022_CSV)
            
            self.predictions = dict(sorted_probs)
            return True
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return False


def main():
    """Main function."""
    logger.info("Starting 2022 World Cup prediction...")
    predictor = Predictor2022()
    return predictor.predict(num_simulations=10000)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
