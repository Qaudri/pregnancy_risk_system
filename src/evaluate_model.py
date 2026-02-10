"""
Model Evaluation Pipeline
Chapter 3.7 & Chapter 4: Results

This script:
1. Loads trained models and test data
2. Evaluates both models on unseen test set
3. Generates confusion matrices
4. Creates ROC curves
5. Compares model performance
6. Saves all figures for thesis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import joblib
import os

# Configuration
DATA_PROCESSED_PATH = 'data/processed/'
MODELS_PATH = 'models/'
RESULTS_FIGURES_PATH = 'results/figures/'
RESULTS_METRICS_PATH = 'results/metrics/'

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

def load_models():
    """Load trained models"""
    print(f"\n{'='*60}")
    print("LOADING MODELS")
    print(f"{'='*60}")
    
    rf_model = joblib.load(f'{MODELS_PATH}rf_model.pkl')
    xgb_model = joblib.load(f'{MODELS_PATH}xgb_model.pkl')
    
    print(" Random Forest loaded")
    print(" XGBoost loaded")
    
    return rf_model, xgb_model

def load_test_data():
    """Load test data"""
    print(f"\n{'='*60}")
    print("LOADING TEST DATA")
    print(f"{'='*60}")
    
    X_test = pd.read_csv(f'{DATA_PROCESSED_PATH}X_test.csv')
    y_test = pd.read_csv(f'{DATA_PROCESSED_PATH}y_test.csv').values.ravel()
    
    print(f" Test features: {X_test.shape}")
    print(f" Test labels: {y_test.shape}")
    
    # Check class distribution
    unique, counts = np.unique(y_test, return_counts=True)
    class_dist = dict(zip(unique, counts))
    
    print(f"\nTest set class distribution:")
    print(f"  Low Risk (0): {class_dist[0]} ({class_dist[0]/len(y_test)*100:.1f}%)")
    print(f"  High Risk (1): {class_dist[1]} ({class_dist[1]/len(y_test)*100:.1f}%)")
    
    return X_test, y_test

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                cbar_kws={'label': 'Count'},
                xticklabels=['Low Risk', 'High Risk'],
                yticklabels=['Low Risk', 'High Risk'])
    
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Risk Level', fontsize=12)
    plt.xlabel('Predicted Risk Level', fontsize=12)
    
    # Add accuracy text
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.4f}',
             ha='center', transform=plt.gca().transAxes, fontsize=10)
    
    # Add cell labels
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                    ha='center', va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    filename = f'{RESULTS_FIGURES_PATH}{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()

def evaluate_single_model(model, X_test, y_test, model_name):
    """Comprehensive evaluation of a single model"""
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name.upper()}")
    print(f"{'='*60}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    metrics['Model'] = model_name
    
    # Print metrics
    print(f"\nPerformance Metrics:")
    for metric, value in metrics.items():
        if metric != 'Model':
            print(f"  {metric}: {value:.4f}")
    
    # Confusion Matrix
    print(f"\nGenerating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, model_name)
    
    # Classification Report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred,
                               target_names=['Low Risk', 'High Risk'],
                               digits=4))
    
    return metrics, y_pred_proba

def plot_roc_comparison(models_data, y_test):
    """Plot ROC curves for model comparison"""
    print(f"\n{'='*60}")
    print("GENERATING ROC CURVES")
    print(f"{'='*60}")
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#3498db', '#e74c3c']
    
    for idx, (model_name, y_pred_proba) in enumerate(models_data.items()):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[idx], lw=2,
                label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # Random classifier line
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--',
            label='Random Classifier (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves',
             fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{RESULTS_FIGURES_PATH}roc_curves_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()

def create_comparison_table(metrics_list):
    """Create and save model comparison table"""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    df_comparison = pd.DataFrame(metrics_list)
    
    # Reorder columns
    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    df_comparison = df_comparison[cols]
    
    print(f"\n{df_comparison.to_string(index=False)}")
    
    # Save to CSV
    os.makedirs(RESULTS_METRICS_PATH, exist_ok=True)
    filename = f'{RESULTS_METRICS_PATH}model_comparison.csv'
    df_comparison.to_csv(filename, index=False)
    print(f"\n Saved comparison table: {filename}")
    
    return df_comparison

def plot_metrics_comparison(df_comparison):
    """Visualize model comparison"""
    print(f"\n{'='*60}")
    print("GENERATING COMPARISON CHART")
    print(f"{'='*60}")
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    df_plot = df_comparison.set_index('Model')[metrics_to_plot]
    
    ax = df_plot.plot(kind='bar', figsize=(12, 6), width=0.8,
                     color=['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c'])
    
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.ylim(0.9, 1.0)  # Focus on high-performance range
    plt.xticks(rotation=0)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3, fontsize=8)
    
    plt.tight_layout()
    filename = f'{RESULTS_FIGURES_PATH}model_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()

def save_detailed_results(metrics_list, y_test, rf_proba, xgb_proba):
    """Save detailed results to text file"""
    print(f"\n{'='*60}")
    print("SAVING DETAILED RESULTS")
    print(f"{'='*60}")
    
    os.makedirs(RESULTS_METRICS_PATH, exist_ok=True)
    filename = f'{RESULTS_METRICS_PATH}evaluation_report.txt'
    
    with open(filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("PREGNANCY RISK PREDICTION SYSTEM\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("TEST SET EVALUATION\n")
        f.write("-"*60 + "\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write(f"Low Risk (0): {(y_test==0).sum()} ({(y_test==0).sum()/len(y_test)*100:.1f}%)\n")
        f.write(f"High Risk (1): {(y_test==1).sum()} ({(y_test==1).sum()/len(y_test)*100:.1f}%)\n\n")
        
        f.write("MODEL PERFORMANCE COMPARISON\n")
        f.write("-"*60 + "\n")
        df_comparison = pd.DataFrame(metrics_list)
        f.write(df_comparison.to_string(index=False))
        f.write("\n\n")
        
        f.write("INTERPRETATION\n")
        f.write("-"*60 + "\n")
        f.write("Both models achieved excellent performance on the test set.\n")
        f.write("High recall is critical for healthcare - we want to catch all high-risk cases.\n")
        f.write("Precision ensures we don't create too many false alarms.\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-"*60 + "\n")
        
        # Compare models
        rf_metrics = metrics_list[0]
        xgb_metrics = metrics_list[1]
        
        if xgb_metrics['F1-Score'] > rf_metrics['F1-Score']:
            f.write("XGBoost shows slightly better overall performance.\n")
            f.write("Recommended for deployment: XGBoost\n")
        else:
            f.write("Random Forest shows slightly better overall performance.\n")
            f.write("Recommended for deployment: Random Forest\n")
        
        f.write("\nBoth models are suitable for clinical decision support.\n")
        f.write("Consider using ensemble predictions for maximum reliability.\n")
    
    print(f"   Saved: {filename}")

def evaluation_pipeline():
    """Complete evaluation pipeline"""
    print(f"\n{'#'*60}")
    print("#" + " "*58 + "#")
    print("#" + " "*17 + "MODEL EVALUATION" + " "*23 + "#")
    print("#" + " "*58 + "#")
    print(f"{'#'*60}")
    
    # Load models and test data
    rf_model, xgb_model = load_models()
    X_test, y_test = load_test_data()
    
    # Evaluate Random Forest
    metrics_rf, proba_rf = evaluate_single_model(
        rf_model, X_test, y_test, 'Random Forest'
    )
    
    # Evaluate XGBoost
    metrics_xgb, proba_xgb = evaluate_single_model(
        xgb_model, X_test, y_test, 'XGBoost'
    )
    
    # Create comparison table
    df_comparison = create_comparison_table([metrics_rf, metrics_xgb])
    
    # Plot comparison
    plot_metrics_comparison(df_comparison)
    
    # ROC curves
    plot_roc_comparison({
        'Random Forest': proba_rf,
        'XGBoost': proba_xgb
    }, y_test)
    
    # Save detailed results
    save_detailed_results([metrics_rf, metrics_xgb], y_test, proba_rf, proba_xgb)
    
    print(f"\n{'#'*60}")
    print("#" + " "*58 + "#")
    print("#" + " "*15 + "EVALUATION COMPLETE!" + " "*22 + "#")
    print("#" + " "*58 + "#")
    print(f"{'#'*60}")
    
    print(f"\nGENERATED FILES:")
    print(f"  Figures:")
    print(f"    - random_forest_confusion_matrix.png")
    print(f"    - xgboost_confusion_matrix.png")
    print(f"    - roc_curves_comparison.png")
    print(f"    - model_comparison.png")
    print(f"  Metrics:")
    print(f"    - model_comparison.csv")
    print(f"    - evaluation_report.txt")
    
    print(f"\n All results saved to:")
    print(f"  {RESULTS_FIGURES_PATH}")
    print(f"  {RESULTS_METRICS_PATH}")
    
    print(f"\n{'='*60}")
    print("Next steps:")
    print("1. Review confusion matrices and ROC curves")
    print("2. Check evaluation_report.txt for detailed analysis")
    print("3. Use these figures in Chapter 4 of thesis")
    print("4. Proceed to SHAP analysis for interpretability")
    print(f"{'='*60}")

# ============================================================================
# RUN EVALUATION
# ============================================================================
if __name__ == "__main__":
    evaluation_pipeline()
    
    print("\n Evaluation script completed successfully!")