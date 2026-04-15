#!/usr/bin/env python
# coding: utf-8

"""
Bank Marketing Experiment Tracking with Optuna and MLflow

This Prefect flow orchestrates the complete ML pipeline for bank marketing classification:
1. Load and preprocess data
2. Feature engineering
3. Train/test split
4. Configure preprocessing pipelines
5. Hyperparameter optimization (HPO) with Optuna for multiple models:
   - Logistic Regression
   - Random Forest
   - XGBoost
6. Track experiments with MLflow
7. Generate artifacts and reports
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

# Preprocesamiento y Modelado
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

# Imbalanced learning
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Hyperparameter optimization
import optuna

# Prefect
from prefect import task, flow, get_run_logger
from prefect.artifacts import create_table_artifact, create_markdown_artifact

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_mlflow(experiment_name: str = "bank-marketing-experiments"):
    """Setup MLflow with proper error handling."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.search_experiments()
        logger.info(f"Connected to MLflow at: {mlflow_uri}")
    except Exception as e:
        logger.warning(f"Failed to connect to {mlflow_uri}: {e}")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.error(f"Failed to set MLflow experiment: {e}")
        raise


@task(name="load_and_preprocess_data", description="Load and preprocess bank marketing data")
def load_and_preprocess_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load raw data and perform initial preprocessing."""
    logger = get_run_logger()
    
    # Si no se proporciona ruta, buscar en ubicaciones comunes
    if data_path is None:
        script_dir = Path(__file__).parent.parent  # Proyecto_final_MLops
        possible_paths = [
            script_dir / "notebooks" / "data" / "processed" / "dataset.parquet",
            script_dir / "data" / "processed" / "dataset.parquet",
            script_dir / "notebooks" / "02-Experiment-Tracking" / "data" / "processed" / "dataset.parquet",
            Path.cwd() / "data" / "processed" / "dataset.parquet",
        ]
        
        for path in possible_paths:
            if path.exists():
                data_path = str(path)
                logger.info(f"Found dataset at: {path}")
                break
        
        if data_path is None:
            raise FileNotFoundError(
                f"Dataset not found. Searched in:\n" + 
                "\n".join(str(p) for p in possible_paths)
            )
    
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    return df


@task(name="feature_engineering", description="Create feature engineering transformations")
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering on the dataset."""
    logger = get_run_logger()
    logger.info("Performing feature engineering...")
    
    # Age grouping
    df['age_group'] = pd.cut(df['age'], bins=[18, 30, 45, 60, 100], 
                              labels=['young', 'adult', 'mid', 'senior'])
    
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
    def season(month):
        if month in ['dec', 'jan', 'feb']:
            return 'winter'
        elif month in ['mar', 'apr', 'may']:
            return 'spring'
        elif month in ['jun', 'jul', 'aug']:
            return 'summer'
        else:
            return 'fall'
    
    df['season'] = df['month'].apply(season)
    
    logger.info(f"Created 7 new features. Total features: {len(df.columns)}")
    
    return df


@task(name="prepare_data", description="Prepare train/test split and feature lists")
def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, 
                                             list, list, list]:
    """Prepare train/test split and classify features."""
    logger = get_run_logger()
    
    # Data type conversions
    df["pdays_clean"] = df["pdays_clean"].astype(float)
    df["previous"] = df["previous"].astype(float)
    df["campaign_log"] = df["campaign_log"].astype(float)
    
    # Feature selection
    features = [
        'age', 'job', 'marital', 'education', 'balance', 'housing', 'loan',
        'contact', 'season', 'campaign_log', 'pdays_clean', 'previous',
        'poutcome', 'prev_success',
    ]
    
    numeric_features = ['age', 'balance', 'campaign_log', 'pdays_clean', 'previous']
    categorical_features = ['job', 'marital', 'education', 'contact', 'season', 'poutcome']
    binary_features = ['housing', 'loan', 'prev_success']
    
    # Train/test split
    X = df[features]
    y = df["y"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert binary features to numeric
    for col in binary_features:
        if X_train[col].dtype == "object":
            X_train[col] = X_train[col].map({"yes": 1, "no": 0})
            X_test[col] = X_test[col].map({"yes": 1, "no": 0})
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Numeric: {len(numeric_features)}, Categorical: {len(categorical_features)}, Binary: {len(binary_features)}")
    
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features, binary_features


@task(name="create_preprocessing_pipelines", description="Create sklearn preprocessing pipelines")
def create_preprocessing_pipelines(numeric_features: list, categorical_features: list):
    """Create preprocessing pipelines for different model types."""
    logger = get_run_logger()
    
    numeric_transformer = SklearnPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    categorical_transformer = SklearnPipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", TargetEncoder(smoothing=10)),
    ])
    
    # LR Preprocessor
    lr_preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ], remainder="passthrough")
    
    # Tree Preprocessor (no scaling)
    tree_preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ], remainder="passthrough")
    
    logger.info("Preprocessing pipelines created successfully")
    
    return lr_preprocessor, tree_preprocessor


@task(name="find_best_threshold_cv", description="Find optimal classification threshold using CV")
def find_best_threshold_cv(X_train: pd.DataFrame, y_train: pd.Series, model_pipeline, 
                          n_splits: int = 3) -> float:
    """Find best threshold using cross-validation on training data (no leakage)."""
    logger = get_run_logger()
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    thresholds = np.linspace(0.2, 0.8, 20)
    best_threshold = 0.5
    best_f1 = 0

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model_pipeline.fit(X_fold_train, y_fold_train)
        y_proba_val = model_pipeline.predict_proba(X_fold_val)[:, 1]

        for t in thresholds:
            y_pred_val = np.where(y_proba_val >= t, "yes", "no")
            f1 = f1_score(y_fold_val, y_pred_val, pos_label="yes")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

    logger.info(f"Best threshold found: {best_threshold:.4f} (F1: {best_f1:.4f})")
    
    return best_threshold


@task(name="optimize_logistic_regression", description="HPO for Logistic Regression with Optuna")
def optimize_logistic_regression(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                y_train: pd.Series, y_test: pd.Series,
                                lr_preprocessor, numeric_features: list,
                                categorical_features: list, binary_features: list,
                                n_trials: int = 10) -> Dict:
    """Optimize Logistic Regression hyperparameters."""
    logger = get_run_logger()
    logger.info("Starting Logistic Regression HPO...")
    
    mlflow.set_experiment("bank-marketing-lr-optuna")
    mlflow.autolog(log_models=False)

    def objective_lr(trial):
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        max_iter = trial.suggest_categorical("max_iter", [500, 1000])

        logger.info(f"LR trial {trial.number}: C={C:.4f}, max_iter={max_iter}")

        pipeline_lr = Pipeline(steps=[
            ("preprocessor", lr_preprocessor),
            ("undersampler", RandomUnderSampler(random_state=42)),
            ("model", LogisticRegression(C=C, max_iter=max_iter, random_state=42))
        ])

        with mlflow.start_run(run_name=f"lr_c{C:.4f}_iter{max_iter}_trial{trial.number}", nested=True) as run:
            trial.set_user_attr("mlflow_run_id", run.info.run_id)
            mlflow.set_tag("problem_type", "classification")
            mlflow.set_tag("model_family", "logistic_regression")
            mlflow.set_tag("dataset", "bank_marketing")

            best_t = find_best_threshold_cv(X_train, y_train, pipeline_lr)
            pipeline_lr.fit(X_train, y_train)
            y_proba = pipeline_lr.predict_proba(X_test)[:, 1]
            y_pred_lr = np.where(y_proba >= best_t, "yes", "no")

            accuracy = accuracy_score(y_test, y_pred_lr)
            precision = precision_score(y_test, y_pred_lr, pos_label="yes")
            recall = recall_score(y_test, y_pred_lr, pos_label="yes")
            f1_yes = f1_score(y_test, y_pred_lr, pos_label="yes")
            auc = roc_auc_score(y_test, y_proba)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_yes", f1_yes)
            mlflow.log_metric("auc", auc)
            mlflow.log_param("best_threshold", best_t)

            logger.info(f"LR trial {trial.number} finished: f1_yes={f1_yes:.4f}, best_threshold={best_t:.4f}")
            return f1_yes

    study_lr = optuna.create_study(direction="maximize")
    
    with mlflow.start_run(run_name="optuna_study_lr") as parent_run:
        mlflow.set_tag("stage", "hpo")
        study_lr.optimize(objective_lr, n_trials=n_trials)
        
        top_trials = sorted(
            [t for t in study_lr.trials if t.value is not None],
            key=lambda t: t.value, reverse=True
        )[:5]

        best_params = study_lr.best_params
        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_yes", study_lr.best_value)
        
        logger.info(f"Completed LR HPO with {len(study_lr.trials)} trials")
        logger.info(f"Best F1 (LR): {study_lr.best_value:.4f}")
        logger.info(f"Best Params: {best_params}")

    return {
        "model_type": "Logistic Regression",
        "best_params": best_params,
        "best_f1": study_lr.best_value,
        "parent_run_id": parent_run.info.run_id,
        "study_trials": len(study_lr.trials)
    }


@task(name="optimize_random_forest", description="HPO for Random Forest with Optuna")
def optimize_random_forest(X_train: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: pd.Series, y_test: pd.Series,
                          tree_preprocessor, numeric_features: list,
                          categorical_features: list, binary_features: list,
                          n_trials: int = 10) -> Dict:
    """Optimize Random Forest hyperparameters."""
    logger = get_run_logger()
    logger.info("Starting Random Forest HPO...")
    
    mlflow.set_experiment("bank-marketing-rf-optuna")
    mlflow.autolog(log_models=False)

    def objective_rf(trial):
        n_estimators = trial.suggest_categorical("n_estimators", [200, 300, 500])
        max_depth = trial.suggest_categorical("max_depth", [5, 10, 15])

        pipeline_rf = Pipeline(steps=[
            ("preprocessor", tree_preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ])

        logger.info(f"RF trial {trial.number}: n_estimators={n_estimators}, max_depth={max_depth}")

        with mlflow.start_run(run_name=f"rf_n{n_estimators}_depth{max_depth}_trial{trial.number}", nested=True) as run:
            trial.set_user_attr("mlflow_run_id", run.info.run_id)
            mlflow.set_tag("problem_type", "classification")
            mlflow.set_tag("model_family", "random_forest")

            pipeline_rf.fit(X_train, y_train)
            y_pred = pipeline_rf.predict(X_test)
            y_proba = pipeline_rf.predict_proba(X_test)[:, 1]

            f1 = f1_score(y_test, y_pred, pos_label="yes")
            auc = roc_auc_score(y_test, y_proba)

            mlflow.log_metric("f1_yes", f1)
            mlflow.log_metric("auc", auc)

            logger.info(f"RF trial {trial.number} finished: f1_yes={f1:.4f}")
            return f1

    study_rf = optuna.create_study(direction="maximize")
    
    with mlflow.start_run(run_name="optuna_study_rf") as parent_run:
        mlflow.set_tag("stage", "hpo")
        study_rf.optimize(objective_rf, n_trials=n_trials)

        best_params = study_rf.best_params
        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_yes", study_rf.best_value)
        
        logger.info(f"Completed RF HPO with {len(study_rf.trials)} trials")
        logger.info(f"Best F1 (RF): {study_rf.best_value:.4f}")
        logger.info(f"Best Params: {best_params}")

    return {
        "model_type": "Random Forest",
        "best_params": best_params,
        "best_f1": study_rf.best_value,
        "parent_run_id": parent_run.info.run_id,
        "study_trials": len(study_rf.trials)
    }


@task(name="optimize_xgboost", description="HPO for XGBoost with Optuna")
def optimize_xgboost(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: pd.Series, y_test: pd.Series,
                    tree_preprocessor, numeric_features: list,
                    categorical_features: list, binary_features: list,
                    neg_pos_ratio: float, n_trials: int = 10) -> Dict:
    """Optimize XGBoost hyperparameters."""
    logger = get_run_logger()
    logger.info("Starting XGBoost HPO...")
    
    # Encode target for XGBoost
    y_train_xgb = y_train.map({"no": 0, "yes": 1})
    y_test_xgb = y_test.map({"no": 0, "yes": 1})
    
    mlflow.set_experiment("bank-marketing-xgb-optuna")
    mlflow.autolog(log_models=False)

    def objective_xgb(trial):
        n_estimators = trial.suggest_categorical("n_estimators", [200, 300, 500])
        learning_rate = trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1])
        max_depth = trial.suggest_categorical("max_depth", [3, 5, 7])
        subsample = trial.suggest_categorical("subsample", [0.8, 1.0])

        pipeline_xgb = Pipeline(steps=[
            ("preprocessor", tree_preprocessor),
            ("model", XGBClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate,
                max_depth=max_depth, subsample=subsample,
                eval_metric="logloss", scale_pos_weight=round(neg_pos_ratio, 2),
                random_state=42
            ))
        ])

        logger.info(f"XGB trial {trial.number}: n_estimators={n_estimators}, lr={learning_rate}, max_depth={max_depth}, subsample={subsample}")

        with mlflow.start_run(run_name=f"xgb_n{n_estimators}_lr{learning_rate}_trial{trial.number}", nested=True) as run:
            trial.set_user_attr("mlflow_run_id", run.info.run_id)
            mlflow.set_tag("problem_type", "classification")
            mlflow.set_tag("model_family", "xgboost")

            pipeline_xgb.fit(X_train, y_train_xgb)
            y_pred = pipeline_xgb.predict(X_test)
            y_proba = pipeline_xgb.predict_proba(X_test)[:, 1]

            f1 = f1_score(y_test_xgb, y_pred)
            auc = roc_auc_score(y_test_xgb, y_proba)

            mlflow.log_metric("f1_yes", f1)
            mlflow.log_metric("auc", auc)

            logger.info(f"XGB trial {trial.number} finished: f1_yes={f1:.4f}")
            return f1

    study_xgb = optuna.create_study(direction="maximize")
    
    with mlflow.start_run(run_name="optuna_study_xgb") as parent_run:
        mlflow.set_tag("stage", "hpo")
        study_xgb.optimize(objective_xgb, n_trials=n_trials)

        best_params = study_xgb.best_params
        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_yes", study_xgb.best_value)
        
        logger.info(f"Completed XGB HPO with {len(study_xgb.trials)} trials")
        logger.info(f"Best F1 (XGB): {study_xgb.best_value:.4f}")
        logger.info(f"Best Params: {best_params}")

    return {
        "model_type": "XGBoost",
        "best_params": best_params,
        "best_f1": study_xgb.best_value,
        "parent_run_id": parent_run.info.run_id,
        "study_trials": len(study_xgb.trials)
    }


@task(name="generate_report", description="Generate experiment summary report")
def generate_report(results: list) -> str:
    """Generate final experiment report."""
    logger = get_run_logger()
    
    # Sort by F1 score
    results_sorted = sorted(results, key=lambda x: x["best_f1"], reverse=True)
    
    markdown_content = """
# Bank Marketing Experiment Tracking Report

## Experiment Overview
This report summarizes the hyperparameter optimization for bank marketing classification using Optuna + MLflow.

## Models Evaluated

"""
    
    for i, result in enumerate(results_sorted, 1):
        markdown_content += f"""
### {i}. {result['model_type']}

- **Best F1 Score**: {result['best_f1']:.4f}
- **Number of Trials**: {result['study_trials']}
- **MLflow Run ID**: {result['parent_run_id']}
- **Best Parameters**: 
  - {json.dumps(result['best_params'], indent=4)}

"""
    
    best_model = results_sorted[0]
    markdown_content += f"""

## Best Model
**Winner: {best_model['model_type']}** with F1 Score = {best_model['best_f1']:.4f}

## Next Steps
1. Review detailed metrics in MLflow UI: http://127.0.0.1:5001
2. Consider ensemble methods or stacking
3. Deploy best model to production
4. Monitor prediction drift over time

---
Generated by Prefect Bank Marketing Flow
"""
    
    # Create Prefect artifact
    create_markdown_artifact(
        key="experiment-report",
        markdown=markdown_content,
        description="Complete experiment tracking report"
    )
    
    logger.info("Report generated successfully")
    
    return markdown_content


@flow(name="Bank Marketing Experiment Tracking", description="End-to-end ML pipeline with Optuna HPO and MLflow tracking")
def bank_marketing_experiment_flow(
    data_path: Optional[str] = None,
    n_trials_per_model: int = 3
) -> Dict:
    """
    Main Prefect flow for bank marketing experiment tracking.
    
    Args:
        data_path: Path to processed dataset. If None, searches in standard locations.
        n_trials_per_model: Number of Optuna trials per model
        
    Returns:
        Dictionary with experiment results
    """
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("Starting Bank Marketing Experiment Tracking Flow")
    logger.info("=" * 60)
    
    # Setup MLflow
    setup_mlflow("bank-marketing-experiments")
    
    # 1. Load and preprocess data
    df = load_and_preprocess_data(data_path)
    
    # 2. Feature engineering
    df = feature_engineering(df)
    
    # 3. Prepare data splits and features
    X_train, X_test, y_train, y_test, numeric_features, categorical_features, binary_features = prepare_data(df)
    
    # 4. Create preprocessing pipelines
    lr_preprocessor, tree_preprocessor = create_preprocessing_pipelines(numeric_features, categorical_features)
    
    # 5. Calculate class ratio for XGBoost
    neg_pos_ratio = (y_train == "no").sum() / (y_train == "yes").sum()
    
    # 6. Optimize models
    lr_results = optimize_logistic_regression(
        X_train, X_test, y_train, y_test, lr_preprocessor,
        numeric_features, categorical_features, binary_features,
        n_trials=n_trials_per_model
    )
    
    rf_results = optimize_random_forest(
        X_train, X_test, y_train, y_test, tree_preprocessor,
        numeric_features, categorical_features, binary_features,
        n_trials=n_trials_per_model
    )
    
    xgb_results = optimize_xgboost(
        X_train, X_test, y_train, y_test, tree_preprocessor,
        numeric_features, categorical_features, binary_features,
        neg_pos_ratio, n_trials=n_trials_per_model
    )
    
    # 7. Generate report
    all_results = [lr_results, rf_results, xgb_results]
    report = generate_report(all_results)
    
    # Create summary artifact
    summary_data = [
        ["Model", "Best F1", "Trials", "Run ID"],
        [lr_results["model_type"], f"{lr_results['best_f1']:.4f}", 
         str(lr_results["study_trials"]), lr_results["parent_run_id"][:8]],
        [rf_results["model_type"], f"{rf_results['best_f1']:.4f}", 
         str(rf_results["study_trials"]), rf_results["parent_run_id"][:8]],
        [xgb_results["model_type"], f"{xgb_results['best_f1']:.4f}", 
         str(xgb_results["study_trials"]), xgb_results["parent_run_id"][:8]],
    ]
    
    create_table_artifact(
        key="experiment-summary",
        table=summary_data,
        description="Experiment summary with all model results"
    )
    
    logger.info("=" * 60)
    logger.info("Flow completed successfully!")
    logger.info("=" * 60)
    
    return {
        "logistic_regression": lr_results,
        "random_forest": rf_results,
        "xgboost": xgb_results,
        "best_model": sorted(all_results, key=lambda x: x["best_f1"], reverse=True)[0]
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Bank Marketing Experiment Tracking with Optuna"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to processed dataset (if None, searches in standard locations)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="Number of Optuna trials per model"
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        help="MLflow tracking URI"
    )
    args = parser.parse_args()
    
    # Override MLflow URI if provided
    if args.mlflow_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
    
    try:
        results = bank_marketing_experiment_flow(
            data_path=args.data_path,
            n_trials_per_model=args.n_trials
        )
        
        print("\n✅ Experiment completed successfully!")
        print(f"🏆 Best Model: {results['best_model']['model_type']}")
        print(f"📊 Best F1 Score: {results['best_model']['best_f1']:.4f}")
        print(f"🔗 MLflow Run ID: {results['best_model']['parent_run_id']}")
        
    except Exception as e:
        logger.error(f"Flow failed: {e}")
        raise
