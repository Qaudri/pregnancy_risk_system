"""
Model Training Pipeline
Chapter 3.6: Model Training

This script:
1. Loads preprocessed training data
2. Trains Random Forest classifier
3. Trains XGBoost classifier
4. Performs cross-validation
5. Saves trained models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import joblib
import os
import time

# Configuration
DATA_PROCESSED_PATH = 'data/processed/'
MODELS_PATH = 'models/'
RANDOM_STATE = 42
CV_FOLDS = 5

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'class_weight': 'balanced',  # Handles class imbalance
    'random_state': RANDOM_STATE,
    'n_jobs': -1  # Use all CPU cores
}

# XGBoost parameters
XGB_BASE_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss'
}

def load_training_data():
    """
    Load preprocessed training data
    """
    print(f"\n{'='*60}")
    print("LOADING TRAINING DATA")
    print(f"{'='*60}")
    
    X_train = pd.read_csv(f'{DATA_PROCESSED_PATH}X_train.csv')
    y_train = pd.read_csv(f'{DATA_PROCESSED_PATH}y_train.csv').values.ravel()
    
    print(f" Training features loaded: {X_train.shape}")
    print(f" Training labels loaded: {y_train.shape}")
    
    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))
    
    print(f"\nClass distribution:")
    print(f"  Low Risk (0): {class_dist[0]} ({class_dist[0]/len(y_train)*100:.1f}%)")
    print(f"  High Risk (1): {class_dist[1]} ({class_dist[1]/len(y_train)*100:.1f}%)")
    
    imbalance_ratio = max(counts) / min(counts)
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    return X_train, y_train

def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier
    
    Random Forest uses ensemble of decision trees with:
    - Bootstrap aggregating (bagging) for robustness
    - Random feature selection at each split
    - Majority voting for final prediction
    """
    print(f"\n{'='*60}")
    print("TRAINING RANDOM FOREST")
    print(f"{'='*60}")
    
    print("\nModel hyperparameters:")
    for param, value in RF_PARAMS.items():
        print(f"  {param}: {value}")
    
    # Initialize model
    rf_model = RandomForestClassifier(**RF_PARAMS)
    
    # Train
    print("\nTraining Random Forest...")
    start_time = time.time()
    
    rf_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f" Training completed in {training_time:.2f} seconds")
    
    # Cross-validation
    print(f"\nPerforming {CV_FOLDS}-fold cross-validation...")
    print("(This evaluates model performance on different data splits)")
    
    # Evaluate multiple metrics
    cv_accuracy = cross_val_score(rf_model, X_train, y_train, 
                                   cv=CV_FOLDS, scoring='accuracy')
    cv_precision = cross_val_score(rf_model, X_train, y_train, 
                                    cv=CV_FOLDS, scoring='precision')
    cv_recall = cross_val_score(rf_model, X_train, y_train, 
                                 cv=CV_FOLDS, scoring='recall')
    cv_f1 = cross_val_score(rf_model, X_train, y_train, 
                            cv=CV_FOLDS, scoring='f1')
    
    print(f"\nCross-validation results (mean ± std):")
    print(f"  Accuracy:  {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
    print(f"  Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
    print(f"  Recall:    {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
    print(f"  F1-Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    
    # Feature importance
    print(f"\nTop 5 most important features:")
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    return rf_model

def train_xgboost(X_train, y_train):
    """
    Train XGBoost classifier
    
    XGBoost uses gradient boosting with:
    - Sequential tree building (each corrects previous errors)
    - Regularization to prevent overfitting
    - Efficient handling of class imbalance
    """
    print(f"\n{'='*60}")
    print("TRAINING XGBOOST")
    print(f"{'='*60}")
    
    # Calculate scale_pos_weight for class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    print(f"\nClass imbalance handling:")
    print(f"  Negative class (0): {neg_count}")
    print(f"  Positive class (1): {pos_count}")
    print(f"  Scale pos weight: {scale_pos_weight:.2f}")
    print(f"  (This weights positive class to balance training)")
    
    # Add scale_pos_weight to parameters
    xgb_params = XGB_BASE_PARAMS.copy()
    xgb_params['scale_pos_weight'] = scale_pos_weight
    
    print("\nModel hyperparameters:")
    for param, value in xgb_params.items():
        print(f"  {param}: {value}")
    
    # Initialize model
    xgb_model = XGBClassifier(**xgb_params)
    
    # Train
    print("\nTraining XGBoost...")
    start_time = time.time()
    
    xgb_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f" Training completed in {training_time:.2f} seconds")
    
    # Cross-validation
    print(f"\nPerforming {CV_FOLDS}-fold cross-validation...")
    
    cv_accuracy = cross_val_score(xgb_model, X_train, y_train, 
                                   cv=CV_FOLDS, scoring='accuracy')
    cv_precision = cross_val_score(xgb_model, X_train, y_train, 
                                    cv=CV_FOLDS, scoring='precision')
    cv_recall = cross_val_score(xgb_model, X_train, y_train, 
                                 cv=CV_FOLDS, scoring='recall')
    cv_f1 = cross_val_score(xgb_model, X_train, y_train, 
                            cv=CV_FOLDS, scoring='f1')
    
    print(f"\nCross-validation results (mean ± std):")
    print(f"  Accuracy:  {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
    print(f"  Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
    print(f"  Recall:    {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
    print(f"  F1-Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    
    # Feature importance
    print(f"\nTop 5 most important features:")
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    return xgb_model

def save_models(rf_model, xgb_model):
    """
    Save trained models to disk
    """
    print(f"\n{'='*60}")
    print("SAVING MODELS")
    print(f"{'='*60}")
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    # Save Random Forest
    rf_path = f'{MODELS_PATH}rf_model.pkl'
    joblib.dump(rf_model, rf_path)
    print(f" Random Forest saved: {rf_path}")
    
    # Save XGBoost
    xgb_path = f'{MODELS_PATH}xgb_model.pkl'
    joblib.dump(xgb_model, xgb_path)
    print(f" XGBoost saved: {xgb_path}")
    
    # Check file sizes
    rf_size = os.path.getsize(rf_path) / 1024  # KB
    xgb_size = os.path.getsize(xgb_path) / 1024  # KB
    
    print(f"\nModel file sizes:")
    print(f"  Random Forest: {rf_size:.1f} KB")
    print(f"  XGBoost: {xgb_size:.1f} KB")

def training_pipeline():
    """
    Complete training pipeline
    """
    print(f"\n{'#'*60}")
    print("#" + " "*58 + "#")
    print("#" + " "*18 + "MODEL TRAINING" + " "*25 + "#")
    print("#" + " "*58 + "#")
    print(f"{'#'*60}")
    
    # Load data
    X_train, y_train = load_training_data()
    
    # Train Random Forest first (easier to debug)
    rf_model = train_random_forest(X_train, y_train)
    
    # Train XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    
    # Save both models
    save_models(rf_model, xgb_model)
    
    print(f"\n{'#'*60}")
    print("#" + " "*58 + "#")
    print("#" + " "*15 + "TRAINING COMPLETE!" + " "*24 + "#")
    print("#" + " "*58 + "#")
    print(f"{'#'*60}")
    
    print(f"\nSUMMARY:")
    print(f"   Random Forest trained and saved")
    print(f"   XGBoost trained and saved")
    print(f"   Both models validated with {CV_FOLDS}-fold cross-validation")
    print(f"   Models ready for evaluation on test set")
    
    print(f"\n{'='*60}")
    print("Next step: Model evaluation (src/evaluate_model.py)")
    print(f"{'='*60}")
    
    return rf_model, xgb_model

# ============================================================================
# RUN TRAINING
# ============================================================================
if __name__ == "__main__":
    rf_model, xgb_model = training_pipeline()
    
    print("\n Training script completed successfully!")