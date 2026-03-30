"""Standalone enhanced figure generation for the pregnancy risk project.

This script implements the publication-style figure guidance directly against the
current repository layout so you do not need to edit notebooks manually.

Usage:
    python notebooks/enhanced_figures.py
    python notebooks/enhanced_figures.py --skip-shap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "primary_dataset.csv"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "figures"

TARGET_COLUMN = "Risk Level"
NUMERICAL_FEATURES = [
    "Age",
    "Systolic BP",
    "Diastolic",
    "BS",
    "Body Temp",
    "BMI",
    "Heart Rate",
]
BOOLEAN_FEATURES = [
    "Previous Complications",
    "Preexisting Diabetes",
    "Gestational Diabetes",
    "Mental Health",
]
ALL_FEATURES = NUMERICAL_FEATURES + BOOLEAN_FEATURES
CLASS_ORDER = ["Low", "High"]

# Distinct Tableau-10-inspired palette — one colour per bar
BAR_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 2.0,
            "lines.linewidth": 2.5,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    sns.set_theme(style="ticks")


def ensure_output_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    png_path = RESULTS_DIR / f"{stem}.png"
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved {png_path.name}")
    plt.close(fig)


def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH)


def load_processed_test_data() -> tuple[pd.DataFrame, np.ndarray]:
    x_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").values.ravel()
    return x_test, y_test


def load_models():
    rf_model = joblib.load(MODELS_DIR / "rf_model.pkl")
    xgb_model = joblib.load(MODELS_DIR / "xgb_model.pkl")
    return rf_model, xgb_model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_pred_proba),
    }


def create_enhanced_class_distribution(df: pd.DataFrame) -> None:
    df_clean = df[df[TARGET_COLUMN].notna()].copy()
    counts = df_clean[TARGET_COLUMN].value_counts().reindex(CLASS_ORDER)
    percentages = counts.div(counts.sum()).mul(100)
    colors = [BAR_PALETTE[0], BAR_PALETTE[1]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    bars = axes[0].bar(CLASS_ORDER, counts.values, color=colors, edgecolor="black", linewidth=2.0)
    axes[0].grid(axis="y")
    axes[0].bar_label(bars, labels=[f"{int(v)}" for v in counts.values],
                      label_type="center", color="white", fontweight="bold", fontsize=12)

    bars = axes[1].bar(CLASS_ORDER, percentages.values, color=colors, edgecolor="black", linewidth=2.0)
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y")
    axes[1].bar_label(bars, labels=[f"{v:.1f}%" for v in percentages.values],
                      label_type="center", color="white", fontweight="bold", fontsize=12)

    fig.tight_layout()
    save_figure(fig, "class_distribution_enhanced")


def create_enhanced_correlation_matrix(df: pd.DataFrame) -> None:
    df_clean = df.dropna(subset=ALL_FEATURES + [TARGET_COLUMN]).copy()
    df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].map({"Low": 0, "High": 1})
    corr = df_clean[ALL_FEATURES + [TARGET_COLUMN]].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        square=True,
        linewidths=2.0,
        vmin=-1,
        vmax=1,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    save_figure(fig, "correlation_matrix_enhanced")


def create_enhanced_feature_distributions(df: pd.DataFrame) -> None:
    df_viz = df[df[TARGET_COLUMN].notna()].copy()
    n_cols = 3
    n_rows = int(np.ceil(len(NUMERICAL_FEATURES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, feature in enumerate(NUMERICAL_FEATURES):
        ax = axes[idx]
        sns.violinplot(
            data=df_viz,
            x=TARGET_COLUMN,
            hue=TARGET_COLUMN,
            y=feature,
            order=CLASS_ORDER,
            hue_order=CLASS_ORDER,
            palette={"Low": BAR_PALETTE[0], "High": BAR_PALETTE[1]},
            inner="box",
            linewidth=2.0,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="y")

    for idx in range(len(NUMERICAL_FEATURES), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    save_figure(fig, "features_by_risk_enhanced")


def create_enhanced_rf_importance(rf_model, feature_names: list[str]) -> pd.DataFrame:
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": rf_model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    bar_colors = [BAR_PALETTE[i % len(BAR_PALETTE)] for i in range(len(importance_df))]
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        importance_df["Feature"],
        importance_df["Importance"],
        color=bar_colors,
        edgecolor="black",
        linewidth=2.0,
    )
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in importance_df["Importance"]],
                 label_type="center", color="white", fontweight="bold", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "rf_feature_importance_enhanced")
    return importance_df


def create_enhanced_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Count"},
        xticklabels=["Low Risk", "High Risk"],
        yticklabels=["Low Risk", "High Risk"],
        linewidths=2.0,
        linecolor="black",
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax,
    )
    fig.tight_layout()
    save_figure(fig, f"{model_name.lower().replace(' ', '_')}_confusion_matrix_enhanced")


def create_enhanced_roc_comparison(y_test: np.ndarray, probability_map: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    roc_colors = [BAR_PALETTE[0], BAR_PALETTE[1]]
    for (model_name, probabilities), color in zip(probability_map.items(), roc_colors):
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {score:.4f})", color=color, linewidth=2.5)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="black", alpha=0.5, label="Random Classifier")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "roc_curves_comparison_enhanced")


def create_enhanced_metrics_comparison(metrics_rows: list[dict[str, float | str]]) -> None:
    df_comparison = pd.DataFrame(metrics_rows)
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    df_plot = df_comparison.set_index("Model")[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot.plot(kind="bar", ax=ax, width=0.78, color=[BAR_PALETTE[i] for i in range(len(metrics_to_plot))])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y")
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", label_type="center",
                     color="white", fontweight="bold", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "model_comparison_enhanced")


def get_optional_shap():
    try:
        import shap

        return shap
    except ModuleNotFoundError:
        return None


def normalise_shap_output(shap_values, expected_value):
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.asarray(expected_value).ravel()[-1]
    return shap_values, float(expected_value)


def create_enhanced_shap_figures(xgb_model, x_test: pd.DataFrame) -> None:
    shap = get_optional_shap()
    if shap is None:
        print("! SHAP is not installed; skipping SHAP enhanced figures.")
        print("  Install with: venv\\Scripts\\python.exe -m pip install shap")
        return

    print("Generating SHAP figures...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(x_test)
    shap_values, base_value = normalise_shap_output(shap_values, explainer.expected_value)

    shap.summary_plot(shap_values, x_test, show=False, plot_size=(10, 8), color_bar_label="Feature Value")
    fig = plt.gcf()
    plt.tight_layout()
    save_figure(fig, "shap_summary_enhanced")

    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False, plot_size=(10, 7))
    fig = plt.gcf()
    plt.tight_layout()
    save_figure(fig, "shap_bar_enhanced")

    sample_index = int(np.argmax(xgb_model.predict_proba(x_test)[:, 1]))
    explanation = shap.Explanation(
        values=shap_values[sample_index],
        base_values=base_value,
        data=x_test.iloc[sample_index].values,
        feature_names=x_test.columns.tolist(),
    )
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    save_figure(fig, "shap_waterfall_enhanced")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = x_test.columns[np.argsort(mean_abs_shap)[-3:][::-1]].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, feature in enumerate(top_features):
        plt.sca(axes[idx])
        shap.dependence_plot(feature, shap_values, x_test, interaction_index=None, show=False, ax=axes[idx], dot_size=20, alpha=0.6)
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel("")
        axes[idx].grid(True)
    fig.tight_layout()
    save_figure(fig, "shap_dependence_top3_enhanced")


def generate_all_figures(skip_shap: bool = False) -> None:
    configure_publication_style()
    ensure_output_dir()

    print(f"Loading data from {RAW_DATA_PATH}")
    raw_df = load_raw_data()
    x_test, y_test = load_processed_test_data()
    rf_model, xgb_model = load_models()

    create_enhanced_class_distribution(raw_df)
    create_enhanced_correlation_matrix(raw_df)
    create_enhanced_feature_distributions(raw_df)
    create_enhanced_rf_importance(rf_model, x_test.columns.tolist())

    rf_pred = rf_model.predict(x_test)
    rf_prob = rf_model.predict_proba(x_test)[:, 1]
    xgb_pred = xgb_model.predict(x_test)
    xgb_prob = xgb_model.predict_proba(x_test)[:, 1]

    create_enhanced_confusion_matrix(y_test, rf_pred, "Random Forest")
    create_enhanced_confusion_matrix(y_test, xgb_pred, "XGBoost")
    create_enhanced_roc_comparison(y_test, {"Random Forest": rf_prob, "XGBoost": xgb_prob})
    create_enhanced_metrics_comparison(
        [
            {"Model": "Random Forest", **calculate_metrics(y_test, rf_pred, rf_prob)},
            {"Model": "XGBoost", **calculate_metrics(y_test, xgb_pred, xgb_prob)},
        ]
    )

    if skip_shap:
        print("Skipping SHAP figures by request.")
    else:
        create_enhanced_shap_figures(xgb_model, x_test)

    print("\nEnhanced figure generation complete.")
    print(f"Output directory: {RESULTS_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate enhanced thesis figures.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP figures even if shap is installed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all_figures(skip_shap=args.skip_shap)
