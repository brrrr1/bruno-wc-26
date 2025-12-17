"""Project-wide constants and configuration."""

import os
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')
APP_DIR = os.path.join(BASE_DIR, 'app')

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Kaggle Dataset
KAGGLE_DATASET = "lchikry/international-football-match-features-and-statistics"

# Data Files
TEAMS_FORM_CSV = os.path.join(RAW_DATA_DIR, 'teams_form.csv')
MATCHES_CSV = os.path.join(RAW_DATA_DIR, 'matches.csv')
WORLD_CUP_MATCHES_CSV = os.path.join(RAW_DATA_DIR, 'world_cup_matches.csv')
TEAM_RATINGS_CSV = os.path.join(RAW_DATA_DIR, 'team_ratings.csv')

# Processed Data
PROCESSED_MATCHES_CSV = os.path.join(PROCESSED_DATA_DIR, 'processed_matches.csv')
PROCESSED_WORLD_CUP_CSV = os.path.join(PROCESSED_DATA_DIR, 'processed_world_cup.csv')
TRAINING_DATA_CSV = os.path.join(PROCESSED_DATA_DIR, 'training_data.csv')

# Model Files
LOGISTIC_REGRESSION_MODEL = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
RANDOM_FOREST_MODEL = os.path.join(MODELS_DIR, 'random_forest.pkl')
SCALER_FILE = os.path.join(MODELS_DIR, 'scaler.pkl')
FEATURE_NAMES_FILE = os.path.join(MODELS_DIR, 'feature_names.pkl')

# Results Files
PREDICTIONS_2022_CSV = os.path.join(RESULTS_DIR, '2022_predictions.csv')
PREDICTIONS_2026_CSV = os.path.join(RESULTS_DIR, '2026_predictions.csv')
BRACKET_2022_CSV = os.path.join(RESULTS_DIR, '2022_bracket.csv')
BRACKET_2026_CSV = os.path.join(RESULTS_DIR, '2026_bracket.csv')

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Random Forest Parameters
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2
RF_MAX_FEATURES = 'sqrt'
RF_JOBS = -1

# Logistic Regression Parameters
LR_C = 1.0
LR_MAX_ITER = 1000
LR_SOLVER = 'lbfgs'
LR_RANDOM_STATE = RANDOM_STATE

# Tournament Simulation
SIMULATION_RUNS = 10000
MONTE_CARLO_ITERATIONS = 10000

# Feature Engineering
RECENT_FORM_MATCHES = 10  # Number of recent matches for form calculation
MIN_MATCHES_FOR_RANKING = 5  # Minimum matches to be considered for ranking

# Teams to Exclude (new teams in 2026 or non-FIFA members)
EXCLUDE_TEAMS = []

# 2022 World Cup Participants
WC_2022_TEAMS = [
    'Netherlands', 'Senegal', 'Ecuador', 'Qatar',
    'England', 'Iran', 'USA', 'Wales',
    'Argentina', 'Saudi Arabia', 'Mexico', 'Poland',
    'France', 'Denmark', 'Tunisia', 'Australia',
    'Spain', 'Costa Rica', 'Germany', 'Japan',
    'Belgium', 'Canada', 'Morocco', 'Croatia'
]

# 2026 World Cup Likely Participants (estimated)
WC_2026_LIKELY_TEAMS = [
    'Argentina', 'Australia', 'Belgium', 'Brazil', 'Canada', 'Croatia',
    'Denmark', 'Ecuador', 'England', 'France', 'Germany', 'Hungary',
    'Iran', 'Japan', 'Mexico', 'Morocco', 'Netherlands', 'Poland',
    'Portugal', 'Senegal', 'South Korea', 'Spain', 'USA', 'Wales',
    'Switzerland', 'Italy', 'Uruguay', 'Colombia', 'Chile', 'Paraguay',
    'Venezuela', 'Peru', 'Jamaica', 'Panama', 'Costa Rica', 'Honduras',
    'Saudi Arabia', 'UAE', 'Uzbekistan', 'China', 'Japan', 'South Korea'
]

# Logging
LOG_FORMAT = '[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
LOG_LEVEL = 'INFO'

# Timestamps
CURRENT_DATE = datetime.now().strftime('%Y-%m-%d')
CURRENT_YEAR = datetime.now().year
