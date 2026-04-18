#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bank Marketing Experiment Tracking with Optuna and MLflow

Complete ML pipeline orchestration for bank marketing classification:
- Data loading and preprocessing
- Feature engineering
- Train/test split
- Preprocessing pipelines
- Hyperparameter optimization (HPO) with Optuna for:
  * Logistic Regression
  * Random Forest
  * XGBoost
- Experiment tracking with MLflow
- Report generation

Usage:
    python flow.py --n-trials 10
    or
    prefect deploy flow.py
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import optuna

# ============================================================================
# Imports: Scikit-learn & Preprocessing
# ============================================================================
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score,
    accuracy_score, precision_score, recall_score
)
from xgboost import XGBClassifier
from category_encoders import TargetEncoder

# ============================================================================
# Imports: Imbalanced Learning & Hyperparameter Optimization
# ============================================================================
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

# ============================================================================
# Imports: Prefect
# ============================================================================
from prefect import task, flow, get_run_logger
from prefect.artifacts import create_markdown_artifact

# ============================================================================
# Setup Logging
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: CONFIGURATION & CONSTANTS
# ============================================================================

class Config:
    """Configuration for bank marketing experiment tracking."""
    
    # MLflow
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    MLFLOW_BACKEND_STORE = "sqlite:///mlflow.db"
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_PATHS = [
        PROJECT_ROOT / "notebooks" / "data" / "processed" / "dataset.parquet",
        PROJECT_ROOT / "data" / "processed" / "dataset.parquet",
        PROJECT_ROOT / "notebooks" / "02-Experiment-Tracking" / "data" / "processed" / "dataset.parquet",
        Path.cwd() / "data" / "processed" / "dataset.parquet",
    ]
    
    # Feature engineering
    AGE_BINS = [18, 30, 45, 60, 100]
    AGE_LABELS = ['young', 'adult', 'mid', 'senior']
    
    # Features lists
    NUMERIC_FEATURES = ['age', 'balance', 'campaign_log', 'pdays_clean', 'previous']
    CATEGORICAL_FEATURES = ['job', 'marital', 'education', 'contact', 'season', 'poutcome']
    BINARY_FEATURES = ['housing', 'loan', 'prev_success']
    ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    
    # Data split
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Preprocessing
    NUMERIC_IMPUTER_STRATEGY = "median"
    CATEGORICAL_IMPUTER_FILL_VALUE = "Unknown"
    TARGET_ENCODER_SMOOTHING = 10
    
    # HPO: Logistic Regression
    LR_C_RANGE = (1e-3, 10.0)
    LR_MAX_ITER_OPTIONS = [500, 1000]
    
    # HPO: Random Forest
    RF_N_ESTIMATORS_OPTIONS = [200, 300, 500]
    RF_MAX_DEPTH_OPTIONS = [5, 10, 15]
    
    # HPO: XGBoost
    XGB_N_ESTIMATORS_OPTIONS = [200, 300, 500]
    XGB_LEARNING_RATE_OPTIONS = [0.01, 0.05, 0.1]
    XGB_MAX_DEPTH_OPTIONS = [3, 5, 7]
    XGB_SUBSAMPLE_OPTIONS = [0.8, 1.0]
    
    # Threshold optimization
    THRESHOLD_STEPS = 60
    MIN_PRECISION_FBETA = 0.18
    FBETA_VALUE = 1.5
    
    # Experiments
    EXPERIMENT_NAMES = {
        "lr": "bank-marketing-lr-optuna",
        "rf": "bank-marketing-rf-optuna",
        "xgb": "bank-marketing-xgb-optuna",
        "main": "bank-marketing-experiments"
    }
    
    DEFAULT_N_TRIALS = 3


# ============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ============================================================================

def setup_mlflow(experiment_name: str = "bank-marketing-experiments") -> None:
    """Setup MLflow with proper error handling."""
    try:
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        mlflow.search_experiments()
        logger.info(f"Connected to MLflow at: {Config.MLFLOW_TRACKING_URI}")
    except Exception as e:
        logger.warning(f"Failed to connect to {Config.MLFLOW_TRACKING_URI}: {e}")
        mlflow.set_tracking_uri(Config.MLFLOW_BACKEND_STORE)

    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.error(f"Failed to set MLflow experiment: {e}")
        raise


def find_dataset_path() -> str:
    """Find dataset in predefined locations."""
    for path in Config.DATA_PATHS:
        if path.exists():
            logger.info(f"Found dataset at: {path}")
            return str(path)
    
    raise FileNotFoundError(
        f"Dataset not found. Searched in:\n" + 
        "\n".join(str(p) for p in Config.DATA_PATHS)
    )


def season_mapper(month: str) -> str:
    """Map month to season."""
    if month in ['dec', 'jan', 'feb']:
        return 'winter'
    elif month in ['mar', 'apr', 'may']:
        return 'spring'
    elif month in ['jun', 'jul', 'aug']:
        return 'summer'
    else:
        return 'fall'


def find_best_threshold_fbeta(
    y_true: pd.Series,
    y_proba: np.ndarray,
    beta: float = Config.FBETA_VALUE,
    min_precision: float = Config.MIN_PRECISION_FBETA,
    pos_label: str = "yes"
) -> Tuple[float, float, dict]:
    """Find best threshold optimizing F-beta metric."""
    thresholds = np.linspace(0.1, 0.9, Config.THRESHOLD_STEPS)
    best_threshold = 0.5
    best_score = -1
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    for t in thresholds:
        y_pred = np.where(y_proba >= t, pos_label, "no" if pos_label == "yes" else 0)
        precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

        denominator = (beta**2 * precision) + recall
        f_beta = ((1 + beta**2) * precision * recall / denominator) if denominator > 0 else 0.0

        if precision >= min_precision and f_beta > best_score:
            best_score = f_beta
            best_threshold = t
            best_metrics = {"precision": precision, "recall": recall, "f1": f1}

    # Fallback if min_precision constraint not met
    if best_score < 0:
        for t in thresholds:
            y_pred = np.where(y_proba >= t, pos_label, "no" if pos_label == "yes" else 0)
            precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

            denominator = (beta**2 * precision) + recall
            f_beta = ((1 + beta**2) * precision * recall / denominator) if denominator > 0 else 0.0

            if f_beta > best_score:
                best_score = f_beta
                best_threshold = t
                best_metrics = {"precision": precision, "recall": recall, "f1": f1}

    return best_threshold, best_score, best_metrics


# ============================================================================
# SECTION 3: PREFECT TASKS - DATA PIPELINE
# ============================================================================

@task(name="load_and_preprocess_data", description="Load and preprocess bank marketing data")
def load_and_preprocess_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load raw data and perform initial preprocessing."""
    log = get_run_logger()
    
    if data_path is None:
        data_path = find_dataset_path()
    
    log.info(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    log.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    return df


@task(name="feature_engineering", description="Create feature engineering transformations")
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Perform comprehensive feature engineering."""
    log = get_run_logger()
    log.info("Performing feature engineering...")
    
    # Age grouping
    df['age_group'] = pd.cut(df['age'], bins=Config.AGE_BINS, labels=Config.AGE_LABELS)
    
    # Campaign log transformation
    df['campaign_log'] = np.log1p(df['campaign'])
    
    # Contact history
    df['contacted_before'] = (df['previous'] > 0).astype(int)
    df['pdays_clean'] = df['pdays'].replace(999, np.nan)
    df['recent_contact'] = (df['pdays_clean'] < 30).astype(int)
    
    # Previous success
    df['prev_success'] = ((df['previous'] > 0) & (df['poutcome'] == 'success')).astype(int)
    
    # Debt indicator
    df['has_debt'] = ((df['housing'] == 'yes') | (df['loan'] == 'yes')).astype(int)
    
    # Season encoding
    df['season'] = df['month'].apply(season_mapper)
    
    log.info(f"Created new features. Total columns: {len(df.columns)}")
    
    return df


@task(name="prepare_data", description="Prepare train/test split and feature lists")
def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list, list, list]:
    """Prepare train/test split and classify features."""
    log = get_run_logger()
    
    # Data type conversions
    df["pdays_clean"] = df["pdays_clean"].astype(float)
    df["previous"] = df["previous"].astype(float)
    df["campaign_log"] = df["campaign_log"].astype(float)
    
    # Train/test split
    X = df[Config.ALL_FEATURES]
    y = df["y"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=y
    )
    
    # Convert binary features to numeric
    for col in Config.BINARY_FEATURES:
        if X_train[col].dtype == "object":
            X_train[col] = X_train[col].map({"yes": 1, "no": 0})
            X_test[col] = X_test[col].map({"yes": 1, "no": 0})
    
    log.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, Config.NUMERIC_FEATURES, Config.CATEGORICAL_FEATURES, Config.BINARY_FEATURES


@task(name="create_preprocessing_pipelines", description="Create sklearn preprocessing pipelines")
def create_preprocessing_pipelines(numeric_features: list, categorical_features: list) -> Tuple:
    """Create preprocessing pipelines for different model types."""
    log = get_run_logger()
    
    # Numeric transformer
    numeric_transformer = SklearnPipeline(steps=[
        ("imputer", SimpleImputer(strategy=Config.NUMERIC_IMPUTER_STRATEGY)),
        ("scaler", StandardScaler()),
    ])
    
    # Categorical transformer
    categorical_transformer = SklearnPipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=Config.CATEGORICAL_IMPUTER_FILL_VALUE)),
        ("encoder", TargetEncoder(smoothing=Config.TARGET_ENCODER_SMOOTHING)),
    ])
    
    # LR Preprocessor (with scaling)
    lr_preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ], remainder="passthrough")
    
    # Tree Preprocessor (no scaling)
    tree_preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ], remainder="passthrough")
    
    log.info("Preprocessing pipelines created successfully")
    
    return lr_preprocessor, tree_preprocessor


# ============================================================================
# SECTION 4: PREFECT TASKS - MODEL OPTIMIZATION
# ============================================================================

@task(name="optimize_logistic_regression", description="HPO for Logistic Regression with Optuna")
def optimize_logistic_regression(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    lr_preprocessor,
    numeric_features: list,
    categorical_features: list,
    binary_features: list,
    n_trials: int = Config.DEFAULT_N_TRIALS
) -> Dict:
    """Optimize Logistic Regression hyperparameters using F-beta threshold tuning."""
    log = get_run_logger()
    log.info("Starting Logistic Regression HPO...")
    
    mlflow.set_experiment(Config.EXPERIMENT_NAMES["lr"])
    mlflow.autolog(log_models=False)

    def objective_lr(trial):
        C = trial.suggest_float("C", Config.LR_C_RANGE[0], Config.LR_C_RANGE[1], log=True)
        max_iter = trial.suggest_categorical("max_iter", Config.LR_MAX_ITER_OPTIONS)

        pipeline_lr = Pipeline(steps=[
            ("preprocessor", lr_preprocessor),
            ("undersampler", RandomUnderSampler(random_state=Config.RANDOM_STATE)),
            ("model", LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced", random_state=Config.RANDOM_STATE))
        ])

        with mlflow.start_run(run_name=f"lr_c{C:.4f}_iter{max_iter}_trial{trial.number}", nested=True) as run:
            trial.set_user_attr("mlflow_run_id", run.info.run_id)
            mlflow.set_tag("problem_type", "classification")
            mlflow.set_tag("model_family", "logistic_regression")
            mlflow.set_tag("dataset", "bank_marketing")
            mlflow.set_tag("selection_logic", f"fbeta_{Config.FBETA_VALUE}")

            pipeline_lr.fit(X_train, y_train)
            y_proba_train = pipeline_lr.predict_proba(X_train)[:, 1]
            best_t, train_f1_5, threshold_metrics = find_best_threshold_fbeta(y_train, y_proba_train)
            
            y_proba_test = pipeline_lr.predict_proba(X_test)[:, 1]
            y_pred_lr = np.where(y_proba_test >= best_t, "yes", "no")

            accuracy = accuracy_score(y_test, y_pred_lr)
            precision = precision_score(y_test, y_pred_lr, pos_label="yes")
            recall = recall_score(y_test, y_pred_lr, pos_label="yes")
            f1 = f1_score(y_test, y_pred_lr, pos_label="yes")
            auc = roc_auc_score(y_test, y_proba_test)
            
            denominator = (Config.FBETA_VALUE**2 * precision) + recall
            f1_5_test = ((1 + Config.FBETA_VALUE**2) * precision * recall / denominator) if denominator > 0 else 0.0

            mlflow.log_metrics({
                "accuracy": accuracy, "precision": precision, "recall": recall,
                "f1": f1, "f1_5": f1_5_test, "auc": auc,
                "train_f1_5": train_f1_5,
            })
            mlflow.log_param("best_threshold", best_t)

            return f1_5_test

    study_lr = optuna.create_study(direction="maximize")
    
    with mlflow.start_run(run_name="optuna_study_lr") as parent_run:
        mlflow.set_tag("stage", "hpo")
        study_lr.optimize(objective_lr, n_trials=n_trials)
        
        best_params = study_lr.best_params
        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_5", study_lr.best_value)

    return {
        "model_type": "Logistic Regression",
        "best_params": best_params,
        "best_f1": study_lr.best_value,
        "parent_run_id": parent_run.info.run_id,
        "study_trials": len(study_lr.trials)
    }


@task(name="optimize_random_forest", description="HPO for Random Forest with Optuna")
def optimize_random_forest(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    tree_preprocessor,
    numeric_features: list,
    categorical_features: list,
    binary_features: list,
    n_trials: int = Config.DEFAULT_N_TRIALS
) -> Dict:
    """Optimize Random Forest hyperparameters."""
    log = get_run_logger()
    log.info("Starting Random Forest HPO...")
    
    mlflow.set_experiment(Config.EXPERIMENT_NAMES["rf"])
    mlflow.autolog(log_models=False)

    def objective_rf(trial):
        n_estimators = trial.suggest_categorical("n_estimators", Config.RF_N_ESTIMATORS_OPTIONS)
        max_depth = trial.suggest_categorical("max_depth", Config.RF_MAX_DEPTH_OPTIONS)

        pipeline_rf = Pipeline(steps=[
            ("preprocessor", tree_preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                class_weight="balanced", random_state=Config.RANDOM_STATE, n_jobs=-1
            ))
        ])

        with mlflow.start_run(run_name=f"rf_n{n_estimators}_depth{max_depth}_trial{trial.number}", nested=True) as run:
            trial.set_user_attr("mlflow_run_id", run.info.run_id)
            mlflow.set_tag("problem_type", "classification")
            mlflow.set_tag("model_family", "random_forest")
            mlflow.set_tag("dataset", "bank_marketing")

            pipeline_rf.fit(X_train, y_train)
            y_pred = pipeline_rf.predict(X_test)
            y_proba = pipeline_rf.predict_proba(X_test)[:, 1]

            f1 = f1_score(y_test, y_pred, pos_label="yes")
            auc = roc_auc_score(y_test, y_proba)

            mlflow.log_metrics({"f1": f1, "auc": auc})

            return f1

    study_rf = optuna.create_study(direction="maximize")
    
    with mlflow.start_run(run_name="optuna_study_rf") as parent_run:
        mlflow.set_tag("stage", "hpo")
        study_rf.optimize(objective_rf, n_trials=n_trials)

        best_params = study_rf.best_params
        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1", study_rf.best_value)

    return {
        "model_type": "Random Forest",
        "best_params": best_params,
        "best_f1": study_rf.best_value,
        "parent_run_id": parent_run.info.run_id,
        "study_trials": len(study_rf.trials)
    }


@task(name="optimize_xgboost", description="HPO for XGBoost with Optuna")
def optimize_xgboost(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    tree_preprocessor,
    numeric_features: list,
    categorical_features: list,
    binary_features: list,
    neg_pos_ratio: float,
    n_trials: int = Config.DEFAULT_N_TRIALS
) -> Dict:
    """Optimize XGBoost hyperparameters."""
    log = get_run_logger()
    log.info("Starting XGBoost HPO...")
    
    y_train_xgb = y_train.map({"no": 0, "yes": 1})
    y_test_xgb = y_test.map({"no": 0, "yes": 1})
    
    mlflow.set_experiment(Config.EXPERIMENT_NAMES["xgb"])
    mlflow.autolog(log_models=False)

    def objective_xgb(trial):
        n_estimators = trial.suggest_categorical("n_estimators", Config.XGB_N_ESTIMATORS_OPTIONS)
        learning_rate = trial.suggest_categorical("learning_rate", Config.XGB_LEARNING_RATE_OPTIONS)
        max_depth = trial.suggest_categorical("max_depth", Config.XGB_MAX_DEPTH_OPTIONS)
        subsample = trial.suggest_categorical("subsample", Config.XGB_SUBSAMPLE_OPTIONS)

        pipeline_xgb = Pipeline(steps=[
            ("preprocessor", tree_preprocessor),
            ("model", XGBClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate,
                max_depth=max_depth, subsample=subsample,
                eval_metric="logloss", scale_pos_weight=round(neg_pos_ratio, 2),
                random_state=Config.RANDOM_STATE
            ))
        ])

        with mlflow.start_run(run_name=f"xgb_n{n_estimators}_lr{learning_rate}_trial{trial.number}", nested=True) as run:
            trial.set_user_attr("mlflow_run_id", run.info.run_id)
            mlflow.set_tag("problem_type", "classification")
            mlflow.set_tag("model_family", "xgboost")
            mlflow.set_tag("dataset", "bank_marketing")

            pipeline_xgb.fit(X_train, y_train_xgb)
            y_pred = pipeline_xgb.predict(X_test)
            y_proba = pipeline_xgb.predict_proba(X_test)[:, 1]

            f1 = f1_score(y_test_xgb, y_pred)
            auc = roc_auc_score(y_test_xgb, y_proba)

            mlflow.log_metrics({"f1": f1, "auc": auc})

            return f1

    study_xgb = optuna.create_study(direction="maximize")
    
    with mlflow.start_run(run_name="optuna_study_xgb") as parent_run:
        mlflow.set_tag("stage", "hpo")
        study_xgb.optimize(objective_xgb, n_trials=n_trials)

        best_params = study_xgb.best_params
        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1", study_xgb.best_value)

    return {
        "model_type": "XGBoost",
        "best_params": best_params,
        "best_f1": study_xgb.best_value,
        "parent_run_id": parent_run.info.run_id,
        "study_trials": len(study_xgb.trials)
    }


# ============================================================================
# SECTION 5: PREFECT TASKS - REPORTING
# ============================================================================

@task(name="generate_report", description="Generate experiment summary report")
def generate_report(results: list) -> str:
    """Generate final experiment report."""
    log = get_run_logger()
    
    results_sorted = sorted(results, key=lambda x: x["best_f1"], reverse=True)
    
    markdown_content = """
# Bank Marketing Experiment Tracking Report

## Experiment Overview
Hyperparameter optimization for bank marketing classification using Optuna + MLflow.

## Models Evaluated

"""
    
    for i, result in enumerate(results_sorted, 1):
        params_str = json.dumps(result['best_params'], indent=4)
        markdown_content += f"""
### {i}. {result['model_type']}

- **Best F1 Score**: {result['best_f1']:.4f}
- **Number of Trials**: {result['study_trials']}
- **MLflow Run ID**: {result['parent_run_id']}
- **Best Parameters**: 
  ```
  {params_str}
  ```

"""
    
    best_model = results_sorted[0]
    markdown_content += f"""

## Best Model
**Winner: {best_model['model_type']}** with F1 Score = {best_model['best_f1']:.4f}

## Recommendations
1. Review detailed metrics in MLflow UI: {Config.MLFLOW_TRACKING_URI}
2. Consider ensemble methods or stacking
3. Deploy best model to production
4. Monitor prediction drift

---
Generated by Prefect Bank Marketing Flow
"""
    
    create_markdown_artifact(
        key="experiment-report",
        markdown=markdown_content,
        description="Complete experiment tracking report"
    )
    
    log.info("Report generated successfully")
    
    return markdown_content


# ============================================================================
# SECTION 6: PREFECT MAIN FLOW
# ============================================================================

@flow(
    name="Bank Marketing Experiment Tracking",
    description="End-to-end ML pipeline with Optuna HPO and MLflow tracking"
)
def bank_marketing_experiment_flow(
    data_path: Optional[str] = None,
    n_trials_per_model: int = Config.DEFAULT_N_TRIALS
) -> Dict:
    """
    Main Prefect flow for bank marketing experiment tracking.
    
    Orchestrates:
    1. Data loading and preprocessing
    2. Feature engineering
    3. Train/test split
    4. Hyperparameter optimization for 3 models
    5. Experiment tracking with MLflow
    6. Report generation
    
    Args:
        data_path: Path to processed dataset
        n_trials_per_model: Number of Optuna trials per model
        
    Returns:
        Dictionary with experiment results
    """
    log = get_run_logger()
    
    log.info("=" * 70)
    log.info("Starting Bank Marketing Experiment Tracking Flow")
    log.info("=" * 70)
    
    # Setup MLflow
    setup_mlflow(Config.EXPERIMENT_NAMES["main"])
    
    # ====================================================================
    # Data Pipeline
    # ====================================================================
    log.info("\n[STEP 1/6] Loading and preprocessing data...")
    df = load_and_preprocess_data(data_path)
    
    log.info("[STEP 2/6] Feature engineering...")
    df = feature_engineering(df)
    
    log.info("[STEP 3/6] Preparing train/test split...")
    X_train, X_test, y_train, y_test, numeric_feat, categorical_feat, binary_feat = prepare_data(df)
    
    log.info("[STEP 4/6] Creating preprocessing pipelines...")
    lr_preprocessor, tree_preprocessor = create_preprocessing_pipelines(numeric_feat, categorical_feat)
    
    # ====================================================================
    # Model Optimization
    # ====================================================================
    log.info("\n[STEP 5/6] Hyperparameter Optimization...")
    
    neg_pos_ratio = (y_train == "no").sum() / (y_train == "yes").sum()
    log.info(f"  Class ratio: {neg_pos_ratio:.2f}")
    
    lr_results = optimize_logistic_regression(
        X_train, X_test, y_train, y_test, lr_preprocessor,
        numeric_feat, categorical_feat, binary_feat, n_trials=n_trials_per_model
    )
    
    rf_results = optimize_random_forest(
        X_train, X_test, y_train, y_test, tree_preprocessor,
        numeric_feat, categorical_feat, binary_feat, n_trials=n_trials_per_model
    )
    
    xgb_results = optimize_xgboost(
        X_train, X_test, y_train, y_test, tree_preprocessor,
        numeric_feat, categorical_feat, binary_feat, neg_pos_ratio,
        n_trials=n_trials_per_model
    )
    
    # ====================================================================
    # Results & Reporting
    # ====================================================================
    log.info("\n[STEP 6/6] Generating report...")
    
    all_results = [lr_results, rf_results, xgb_results]
    best_model = max(all_results, key=lambda x: x["best_f1"])
    
    generate_report(all_results)
    
    log.info("\n" + "=" * 70)
    log.info("✅ Flow Completed!")
    log.info("=" * 70)
    log.info(f"\n🏆 BEST MODEL: {best_model['model_type']} (F1 = {best_model['best_f1']:.4f})")
    log.info(f"\n📊 MLflow UI: {Config.MLFLOW_TRACKING_URI}")
    log.info("\n📋 Results:")
    for result in sorted(all_results, key=lambda x: x["best_f1"], reverse=True):
        log.info(f"  - {result['model_type']}: F1 = {result['best_f1']:.4f}")
    
    return {
        "all_results": all_results,
        "best_model": best_model
    }


# ============================================================================
# SECTION 7: MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """Run the flow locally."""
    result = bank_marketing_experiment_flow(n_trials_per_model=3)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Best Model: {result['best_model']['model_type']}")
    print(f"Best F1 Score: {result['best_model']['best_f1']:.4f}")
    print(f"\nView experiments at: {Config.MLFLOW_TRACKING_URI}")
