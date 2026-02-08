"""
Configuration file for pregnancy risk prediction system.
"""

# ============================================================================
# RANDOM STATE (for reproducibility)
# ============================================================================
RANDOM_STATE = 42

# ============================================================================
# DATA SPLITTING
# ============================================================================
TEST_SIZE = 0.15
VALIDATION_SPLIT = 0.2

# ============================================================================
# FEATURE NAMES - PRIMARY DATASET
# ============================================================================
# ⚠️ UPDATED TO MATCH THE CSV FILE EXACTLY

NUMERICAL_FEATURES = [
    'Age',
    'Systolic BP',      # Note: has space!
    'Diastolic',
    'BS',
    'Body Temp',        # Note: has space!
    'BMI',
    'Heart Rate'        # Note: has space!
]

BOOLEAN_FEATURES = [
    'Previous Complications',
    'Preexisting Diabetes',
    'Gestational Diabetes',
    'Mental Health'
]

ALL_FEATURES = NUMERICAL_FEATURES + BOOLEAN_FEATURES

TARGET_COLUMN = 'Risk Level'  # Note: has space!

# ============================================================================
# SECONDARY DATASET (different columns - we'll use this later)
# ============================================================================
SECONDARY_NUMERICAL_FEATURES = [
    'Age',
    'SystolicBP',      # No space
    'DiastolicBP',     # No space
    'BS',
    'BodyTemp',        # No space
    'HeartRate'        # No space
]

SECONDARY_TARGET = 'RiskLevel'  # No space

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss'
}

# ============================================================================
# FILE PATHS
# ============================================================================
DATA_RAW_PATH = 'data/raw/'
DATA_PROCESSED_PATH = 'data/processed/'
MODELS_PATH = 'models/'
RESULTS_FIGURES_PATH = 'results/figures/'
RESULTS_METRICS_PATH = 'results/metrics/'

# ============================================================================
# RISK MAPPING
# ============================================================================
RISK_MAPPING = {
    'High': 1,
    'Low': 0,
    'high': 1,
    'low': 0,
    'high risk': 1,
    'low risk': 0,
    'mid risk': 1,  # We'll treat mid risk as high risk for binary classification
    1: 'High Risk',
    0: 'Low Risk'
}

# ============================================================================
# EVALUATION
# ============================================================================
CLASSIFICATION_THRESHOLD = 0.5
CV_FOLDS = 5