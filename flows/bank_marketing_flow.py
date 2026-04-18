from prefect import task, flow
import pandas as pd
import numpy as np
import mlflow
import optuna

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


# -----------------------------
# TASK 1: LOAD DATA
# -----------------------------
@task
def load_data():

    df = pd.read_parquet("data/processed/bank.parquet")  # ajusta ruta

    X = df.drop(columns=["y"])
    y = df["y"]

    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )


# -----------------------------
# THRESHOLD
# -----------------------------
def find_best_threshold(y_true, y_proba):

    thresholds = np.linspace(0.2, 0.7, 30)
    best_t, best_f1 = 0.5, 0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t


# -----------------------------
# TASK 2: TRAIN (OPTUNA)
# -----------------------------
@task
def train_model(X_train, y_train, X_test, y_test):

    def objective(trial):

        C = trial.suggest_float("C", 1e-3, 10.0, log=True)

        model = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        # threshold en train
        y_proba_train = model.predict_proba(X_train)[:, 1]
        best_t = find_best_threshold(y_train, y_proba_train)

        # test
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= best_t).astype(int)

        return f1_score(y_test, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    return study.best_params


# -----------------------------
# TASK 3: EVALUATE
# -----------------------------
@task
def evaluate_model(best_params, X_train, y_train, X_test, y_test):

    model = LogisticRegression(
        **best_params,
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # threshold en train
    y_proba_train = model.predict_proba(X_train)[:, 1]
    best_t = find_best_threshold(y_train, y_proba_train)

    # test
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= best_t).astype(int)

    metrics = {
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "threshold": best_t
    }

    return metrics


# -----------------------------
# FLOW
# -----------------------------
@flow(name="Bank Marketing Flow")
def main():

    mlflow.set_experiment("bank-marketing-prefect")

    with mlflow.start_run():

        X_train, X_test, y_train, y_test = load_data()

        best_params = train_model(X_train, y_train, X_test, y_test)

        metrics = evaluate_model(
            best_params,
            X_train, y_train,
            X_test, y_test
        )

        mlflow.log_metrics(metrics)

        print(metrics)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()