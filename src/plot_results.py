from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(RESULTS / "processed_data.parquet")
splits = pd.read_csv(RESULTS / "split_assignments.csv")

test_ids = splits.loc[splits["split"] == "test", "case_id"]
test = df[df["case_id"].isin(test_ids)].copy()

features = [
    c for c in df.columns
    if c not in ["case_id", "density", "start_time"]
]

X = test[features]
actual = test["density"].to_numpy()

models = {
    "Random Forest": joblib.load(RESULTS / "models" / "rf.joblib"),
    "LightGBM": joblib.load(RESULTS / "models" / "lightgbm.joblib"),
    "Ridge Regression": joblib.load(RESULTS / "models" / "ridge.joblib"),
}

# Group predictions into six-hour forecast intervals.
bins = np.arange(0, 78, 6)
test["horizon_bin"] = pd.cut(
    test["lead_hours"],
    bins=bins,
    include_lowest=True,
    right=True
)

plt.figure(figsize=(8, 5))

for name, model in models.items():
    predictions = 10 ** model.predict(X)
    test["squared_error"] = (actual - predictions) ** 2

    rmse_by_horizon = (
        test.groupby("horizon_bin", observed=True)["squared_error"]
        .mean()
        .pow(0.5)
    )

    centers = np.arange(3, 72, 6)
    plt.plot(
        centers[:len(rmse_by_horizon)],
        rmse_by_horizon.to_numpy(),
        marker="o",
        label=name
    )

plt.xlabel("Forecast lead time (hours)")
plt.ylabel("RMSE (kg/m³)")
plt.title("Atmospheric Density Forecast Error vs. Lead Time")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(FIGURES / "rmse_vs_horizon.png", dpi=300)
plt.close()

print("Saved results/figures/rmse_vs_horizon.png")
