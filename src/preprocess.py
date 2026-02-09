"""
Data Preprocessing Pipeline
Chapter 3.4: Data Preprocessing

This script:
1. Loads raw data
2. Handles missing values
3. Encodes target variable (High=1, Low=0)
4. Splits into train/test sets with stratification
5. Scales numerical features using StandardScaler
6. Saves processed data and scaler
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Manual config (same as notebook)
DATA_RAW_PATH = 'data/raw/'
DATA_PROCESSED_PATH = 'data/processed/'
MODELS_PATH = 'models/'

NUMERICAL_FEATURES = [
    'Age',
    'Systolic BP',
    'Diastolic',
    'BS',
    'Body Temp',
    'BMI',
    'Heart Rate'
]

BOOLEAN_FEATURES = [
    'Previous Complications',
    'Preexisting Diabetes',
    'Gestational Diabetes',
    'Mental Health'
]

ALL_FEATURES = NUMERICAL_FEATURES + BOOLEAN_FEATURES
TARGET_COLUMN = 'Risk Level'

RISK_MAPPING = {
    'High': 1,
    'Low': 0,
    'high': 1,
    'low': 0
}

RANDOM_STATE = 42
TEST_SIZE = 0.15

def load_data(filepath):
    """
    Load raw dataset from CSV
    """
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {df.shape[0]} samples with {df.shape[1]} features")
    
    return df

def handle_missing_values(df):
    """
    Handle missing values
    
    Strategy:
    1. Drop rows where Risk Level is missing (target variable)
    2. Drop rows where any feature has missing values
       (Since we only have 53 missing values out of 1205, this is acceptable)
    """
    print(f"\n{'='*60}")
    print("HANDLING MISSING VALUES")
    print(f"{'='*60}")
    
    initial_count = len(df)
    
    # Check missing values before
    total_missing = df.isnull().sum().sum()
    print(f"\nInitial missing values: {total_missing}")
    print(f"Missing by column:")
    missing_cols = df.isnull().sum()
    for col in missing_cols[missing_cols > 0].index:
        print(f"  {col}: {missing_cols[col]}")
    
    # Drop rows with missing Risk Level first
    df_clean = df.dropna(subset=[TARGET_COLUMN])
    print(f"\n✓ Dropped {initial_count - len(df_clean)} rows with missing Risk Level")
    
    # Drop rows with any missing features
    df_clean = df_clean.dropna()
    
    final_count = len(df_clean)
    dropped = initial_count - final_count
    
    print(f"✓ Dropped {dropped} total rows with missing values ({dropped/initial_count*100:.2f}%)")
    print(f"✓ Remaining samples: {final_count}")
    
    # Verify no missing values remain
    remaining_missing = df_clean.isnull().sum().sum()
    if remaining_missing == 0:
        print(f"✓ All missing values handled successfully")
    else:
        print(f"⚠ Warning: {remaining_missing} missing values still remain")
    
    return df_clean

def encode_target(df):
    """
    Encode target variable to binary (0/1)
    High Risk = 1
    Low Risk = 0
    """
    print(f"\n{'='*60}")
    print("ENCODING TARGET VARIABLE")
    print(f"{'='*60}")
    
    print(f"\nOriginal Risk Level values:")
    print(df[TARGET_COLUMN].value_counts())
    
    # Map to binary
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(RISK_MAPPING)
    
    print(f"\nEncoded Risk Level values:")
    print(df[TARGET_COLUMN].value_counts())
    print(f"\nMapping:")
    print(f"  High Risk → 1")
    print(f"  Low Risk → 0")
    
    # Verify encoding worked
    if df[TARGET_COLUMN].isnull().sum() > 0:
        print(f"⚠ Warning: Encoding failed for some values")
        print(f"  Unknown values: {df[df[TARGET_COLUMN].isnull()][TARGET_COLUMN]}")
    else:
        print(f"✓ Target encoding successful")
    
    return df

def split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split data into train and test sets with stratification
    
    Stratification ensures both sets have the same proportion of High/Low risk
    """
    print(f"\n{'='*60}")
    print("SPLITTING DATA")
    print(f"{'='*60}")
    
    # Separate features and target
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # CRITICAL: maintains class balance
    )
    
    print(f"\n✓ Split complete:")
    print(f"  Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    print(f"  Test set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    
    print(f"\nClass distribution in training set:")
    train_dist = y_train.value_counts()
    print(f"  Low Risk (0): {train_dist[0]} ({train_dist[0]/len(y_train)*100:.1f}%)")
    print(f"  High Risk (1): {train_dist[1]} ({train_dist[1]/len(y_train)*100:.1f}%)")
    
    print(f"\nClass distribution in test set:")
    test_dist = y_test.value_counts()
    print(f"  Low Risk (0): {test_dist[0]} ({test_dist[0]/len(y_test)*100:.1f}%)")
    print(f"  High Risk (1): {test_dist[1]} ({test_dist[1]/len(y_test)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Standardize numerical features using Z-score normalization
    
    Formula: X_scaled = (X - μ) / σ
    where μ = mean, σ = standard deviation
    
    CRITICAL: Scaler is fit ONLY on training data to prevent data leakage
    """
    print(f"\n{'='*60}")
    print("SCALING FEATURES")
    print(f"{'='*60}")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    print(f"\nScaling {len(NUMERICAL_FEATURES)} numerical features:")
    for feature in NUMERICAL_FEATURES:
        print(f"  - {feature}")
    
    # Fit scaler on TRAINING data only
    scaler.fit(X_train[NUMERICAL_FEATURES])
    
    print(f"\n✓ Scaler fitted on training data")
    print(f"  Training set means:")
    for i, feature in enumerate(NUMERICAL_FEATURES):
        print(f"    {feature}: {scaler.mean_[i]:.2f}")
    
    # Transform both sets
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[NUMERICAL_FEATURES] = scaler.transform(X_train[NUMERICAL_FEATURES])
    X_test_scaled[NUMERICAL_FEATURES] = scaler.transform(X_test[NUMERICAL_FEATURES])
    
    # Verify scaling worked
    scaled_means = X_train_scaled[NUMERICAL_FEATURES].mean()
    scaled_stds = X_train_scaled[NUMERICAL_FEATURES].std()
    
    print(f"\n✓ Features scaled successfully")
    print(f"  Training set after scaling:")
    print(f"    Mean (should be ~0): {scaled_means.mean():.6f}")
    print(f"    Std (should be ~1): {scaled_stds.mean():.6f}")
    
    return X_train_scaled, X_test_scaled, scaler

def save_processed_data(X_train, X_test, y_train, y_test, scaler):
    """
    Save all processed data and scaler for later use
    """
    print(f"\n{'='*60}")
    print("SAVING PROCESSED DATA")
    print(f"{'='*60}")
    
    # Create directories if they don't exist
    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    # Save datasets
    X_train.to_csv(f'{DATA_PROCESSED_PATH}X_train.csv', index=False)
    X_test.to_csv(f'{DATA_PROCESSED_PATH}X_test.csv', index=False)
    y_train.to_csv(f'{DATA_PROCESSED_PATH}y_train.csv', index=False, header=True)
    y_test.to_csv(f'{DATA_PROCESSED_PATH}y_test.csv', index=False, header=True)
    
    print(f"✓ Saved training features: {DATA_PROCESSED_PATH}X_train.csv")
    print(f"✓ Saved test features: {DATA_PROCESSED_PATH}X_test.csv")
    print(f"✓ Saved training labels: {DATA_PROCESSED_PATH}y_train.csv")
    print(f"✓ Saved test labels: {DATA_PROCESSED_PATH}y_test.csv")
    
    # Save scaler
    joblib.dump(scaler, f'{MODELS_PATH}scaler.pkl')
    print(f"✓ Saved scaler: {MODELS_PATH}scaler.pkl")
    
    print(f"\n✓ All files saved successfully")

def preprocessing_pipeline(input_filepath):
    """
    Complete preprocessing pipeline
    
    Args:
        input_filepath: Path to raw CSV file
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    print(f"\n{'#'*60}")
    print("#" + " "*58 + "#")
    print("#" + " "*15 + "PREPROCESSING PIPELINE" + " "*23 + "#")
    print("#" + " "*58 + "#")
    print(f"{'#'*60}")
    
    # Step 1: Load data
    df = load_data(input_filepath)
    
    # Step 2: Handle missing values
    df_clean = handle_missing_values(df)
    
    # Step 3: Encode target variable
    df_encoded = encode_target(df_clean)
    
    # Step 4: Split into train/test
    X_train, X_test, y_train, y_test = split_data(df_encoded)
    
    # Step 5: Scale numerical features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 6: Save everything
    save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    
    print(f"\n{'#'*60}")
    print("#" + " "*58 + "#")
    print("#" + " "*12 + "PREPROCESSING COMPLETE!" + " "*25 + "#")
    print("#" + " "*58 + "#")
    print(f"{'#'*60}")
    
    print(f"\nSUMMARY:")
    print(f"  Initial samples: {len(df)}")
    print(f"  After cleaning: {len(df_clean)}")
    print(f"  Training samples: {len(X_train_scaled)}")
    print(f"  Test samples: {len(X_test_scaled)}")
    print(f"  Features: {len(ALL_FEATURES)}")
    print(f"\n✓ Ready for model training!")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ============================================================================
# RUN PREPROCESSING
# ============================================================================
if __name__ == "__main__":
    # Run the pipeline
    X_train, X_test, y_train, y_test, scaler = preprocessing_pipeline(
        f'{DATA_RAW_PATH}primary_dataset.csv'
    )
    
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE FINISHED SUCCESSFULLY")
    print("="*60)
    print("\nNext step: Model training (src/train_model.py)")