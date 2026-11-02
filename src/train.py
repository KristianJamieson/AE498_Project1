from pathlib import Path
import argparse
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"

RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


def load_data():
    print("Loading processed dataset...")

    df = pd.read_parquet(RESULTS / "processed_data.parquet")

    state_files = sorted((ROOT / "initial_states").glob("*.csv"))
    states = pd.concat(
        [
            pd.read_csv(path, usecols=["File ID", "Timestamp"])
            for path in state_files
        ],
        ignore_index=True
    )

    states = states.rename(
        columns={"File ID": "case_id", "Timestamp": "start_time"}
    )

    states["start_time"] = pd.to_datetime(states["start_time"])

    df = df.merge(
        states, on="case_id", how="left", validate="many_to_one"
    )

    if df["start_time"].isna().any():
        raise ValueError("Missing initial timestamps.")

    return df


def split_data(df):
    cases = (
        df[["case_id", "start_time"]]
        .drop_duplicates()
        .sort_values("start_time")
        .reset_index(drop=True)
    )

    n = len(cases)

    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_boundary = cases.loc[train_end, "start_time"]
    val_boundary = cases.loc[val_end, "start_time"]

    gap = pd.Timedelta(hours=72)

    train_ids = cases.loc[
        cases["start_time"] < train_boundary - gap,
        "case_id"
    ]

    val_ids = cases.loc[
        (cases["start_time"] >= train_boundary)
        & (cases["start_time"] < val_boundary - gap),
        "case_id"
    ]

    test_ids = cases.loc[
        cases["start_time"] >= val_boundary,
        "case_id"
    ]

    assignments = cases.copy()
    assignments["split"] = "excluded_gap"
    assignments.loc[
        assignments["case_id"].isin(train_ids), "split"
    ] = "train"
    assignments.loc[
        assignments["case_id"].isin(val_ids), "split"
    ] = "validation"
    assignments.loc[
        assignments["case_id"].isin(test_ids), "split"
    ] = "test"

    assignments.to_csv(RESULTS / "split_assignments.csv", index=False)

    train = df[df["case_id"].isin(train_ids)]
    val = df[df["case_id"].isin(val_ids)]
    test = df[df["case_id"].isin(test_ids)]

    print(f"Training:   {len(train):,} observations")
    print(f"Validation: {len(val):,} observations")
    print(f"Testing:    {len(test):,} observations")

    return train, val, test


def get_model(name):
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=5,
            max_features=0.8,
            n_jobs=-1,
            random_state=42
        )

    if name == "lightgbm":
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

    if name == "ridge":
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=10.0)
        )

    raise ValueError(f"Unknown model: {name}")


def evaluate(model, X, y_log, name, split):
    predictions_log = model.predict(X)

    actual = 10 ** y_log
    predicted = 10 ** predictions_log

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    print(
        f"{name} | {split} | "
        f"RMSE={rmse:.6e} | MAE={mae:.6e}"
    )

    return {
        "model": name,
        "split": split,
        "rmse": rmse,
        "mae": mae,
        "observations": len(actual)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train atmospheric density forecasting models."
    )

    parser.add_argument(
        "--model",
        choices=["rf", "lightgbm", "ridge", "both"],
        default="both",
        help="Model to train (default: both)"
    )

    args = parser.parse_args()

    df = load_data()
    train, val, test = split_data(df)

    features = [
        c for c in df.columns
        if c not in ["case_id", "density", "start_time"]
    ]

    X_train = train[features]
    X_val = val[features]
    X_test = test[features]

    y_train = np.log10(train["density"].to_numpy())
    y_val = np.log10(val["density"].to_numpy())
    y_test = np.log10(test["density"].to_numpy())

    selected = (
        ["rf", "lightgbm", "ridge"]
        if args.model == "both"
        else [args.model]
    )

    all_metrics = []

    for name in selected:
        print(f"\nTraining {name}...")

        model = get_model(name)

        start = time.perf_counter()
        model.fit(X_train, y_train)
        training_seconds = time.perf_counter() - start

        print(f"Training time: {training_seconds:.2f} seconds")

        val_metrics = evaluate(
            model, X_val, y_val, name, "validation"
        )
        test_metrics = evaluate(
            model, X_test, y_test, name, "test"
        )

        for metrics in [val_metrics, test_metrics]:
            metrics["training_seconds"] = training_seconds
            all_metrics.append(metrics)

        joblib.dump(model, MODELS / f"{name}.joblib")
        print(f"Saved {name} model.")

    metrics_path = RESULTS / "metrics.csv"
    new_metrics = pd.DataFrame(all_metrics)

    if metrics_path.exists():
        previous = pd.read_csv(metrics_path)
        previous = previous[
            ~previous["model"].isin(selected)
        ]
        new_metrics = pd.concat(
            [previous, new_metrics], ignore_index=True
        )

    new_metrics.to_csv(metrics_path, index=False)

    print("\nFinal metrics:")
    print(new_metrics.to_string(index=False))
    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    main()