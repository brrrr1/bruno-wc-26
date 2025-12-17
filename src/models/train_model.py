"""Model training and validation."""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, classification_report, roc_auc_score
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.helpers import setup_logger, load_data, ensure_directory_exists
from src.utils.constants import (
    TRAINING_DATA_CSV, LOGISTIC_REGRESSION_MODEL, RANDOM_FOREST_MODEL,
    SCALER_FILE, FEATURE_NAMES_FILE, MODELS_DIR,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT, RF_MIN_SAMPLES_LEAF,
    RF_MAX_FEATURES, RF_JOBS, LR_C, LR_MAX_ITER, LR_SOLVER, RANDOM_STATE, TEST_SIZE
)

logger = setup_logger(__name__)


class ModelTrainer:
    """Class to train and evaluate models."""
    
    def __init__(self):
        """Initialize model trainer."""
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.lr_model = None
        self.rf_model = None
        self.feature_names = None
        
    def load_training_data(self) -> bool:
        """Load training data."""
        try:
            logger.info("Loading training data...")
            df = load_data(TRAINING_DATA_CSV)
            
            if df is None or len(df) == 0:
                logger.error("No training data found")
                return False
            
            # Separate features and target
            self.y = df['result'].values
            
            # Drop non-feature columns
            drop_cols = ['result', 'home_team', 'away_team', 'date']
            self.X = df.drop(columns=[col for col in drop_cols if col in df.columns]).values
            self.feature_names = df.drop(columns=[col for col in drop_cols if col in df.columns]).columns.tolist()
            
            # Remove any NaN or infinite values
            mask = ~(np.isnan(self.X).any(axis=1) | np.isinf(self.X).any(axis=1))
            self.X = self.X[mask]
            self.y = self.y[mask]
            
            logger.info(f"Loaded {len(self.X)} samples with {len(self.feature_names)} features")
            return True
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            return False
    
    def split_data(self) -> bool:
        """Split data into train and test sets."""
        try:
            logger.info("Splitting data...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=self.y
            )
            logger.info(f"Train set: {len(self.X_train)} samples")
            logger.info(f"Test set: {len(self.X_test)} samples")
            return True
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            return False
    
    def scale_data(self) -> bool:
        """Scale features."""
        try:
            logger.info("Scaling features...")
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            logger.info("Data scaled successfully")
            return True
        except Exception as e:
            logger.error(f"Error scaling data: {str(e)}")
            return False
    
    def train_logistic_regression(self) -> bool:
        """Train logistic regression model."""
        try:
            logger.info("Training Logistic Regression model...")
            self.lr_model = LogisticRegression(
                C=LR_C,
                max_iter=LR_MAX_ITER,
                solver=LR_SOLVER,
                random_state=RANDOM_STATE,
                verbose=1
            )
            self.lr_model.fit(self.X_train, self.y_train)
            logger.info("Logistic Regression training completed")
            return True
        except Exception as e:
            logger.error(f"Error training Logistic Regression: {str(e)}")
            return False
    
    def train_random_forest(self) -> bool:
        """Train random forest model."""
        try:
            logger.info("Training Random Forest model...")
            self.rf_model = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                min_samples_split=RF_MIN_SAMPLES_SPLIT,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                max_features=RF_MAX_FEATURES,
                random_state=RANDOM_STATE,
                n_jobs=RF_JOBS,
                verbose=1
            )
            self.rf_model.fit(self.X_train, self.y_train)
            logger.info("Random Forest training completed")
            return True
        except Exception as e:
            logger.error(f"Error training Random Forest: {str(e)}")
            return False
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> dict:
        """Evaluate model performance."""
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # For multiclass (result: 0, 0.5, 1)
            if len(np.unique(y_test)) > 2:
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            else:
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            metrics = {
                'model': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'log_loss': log_loss(y_test, y_pred_proba),
                'auc': auc_score,
            }
            
            logger.info(f"{model_name} Results:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            return None
    
    def save_models(self) -> bool:
        """Save trained models to disk."""
        try:
            ensure_directory_exists(MODELS_DIR)
            
            # Save models
            with open(LOGISTIC_REGRESSION_MODEL, 'wb') as f:
                pickle.dump(self.lr_model, f)
            logger.info(f"Saved Logistic Regression to {LOGISTIC_REGRESSION_MODEL}")
            
            with open(RANDOM_FOREST_MODEL, 'wb') as f:
                pickle.dump(self.rf_model, f)
            logger.info(f"Saved Random Forest to {RANDOM_FOREST_MODEL}")
            
            # Save scaler
            with open(SCALER_FILE, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Saved scaler to {SCALER_FILE}")
            
            # Save feature names
            with open(FEATURE_NAMES_FILE, 'wb') as f:
                pickle.dump(self.feature_names, f)
            logger.info(f"Saved feature names to {FEATURE_NAMES_FILE}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False
    
    def train_all(self) -> bool:
        """Train all models and evaluate."""
        try:
            # Load data
            if not self.load_training_data():
                return False
            
            # Split and scale data
            if not self.split_data() or not self.scale_data():
                return False
            
            # Train models
            if not self.train_logistic_regression():
                return False
            
            if not self.train_random_forest():
                return False
            
            # Evaluate models
            logger.info("\n" + "="*60)
            logger.info("MODEL EVALUATION")
            logger.info("="*60)
            
            lr_metrics = self.evaluate_model(self.lr_model, self.X_test, self.y_test, "Logistic Regression")
            rf_metrics = self.evaluate_model(self.rf_model, self.X_test, self.y_test, "Random Forest")
            
            # Save models
            if not self.save_models():
                return False
            
            logger.info("\n" + "="*60)
            logger.info("Training completed successfully!")
            logger.info("="*60)
            
            return True
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False


def main():
    """Main function."""
    logger.info("Starting model training...")
    trainer = ModelTrainer()
    return trainer.train_all()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
