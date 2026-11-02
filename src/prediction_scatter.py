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
    "Ridge Regression": joblib.load(RESULTS / "models" / "ridge.joblib")
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

rng = np.random.default_rng(42)
sample_size = min(15000, len(test))
indices = rng.choice(len(test), size=sample_size, replace=False)

for ax, (name, model) in zip(axes, models.items()):
    predicted = 10 ** model.predict(X)

    relative_error = np.abs(predicted - actual) / actual
    mape = 100 * np.mean(relative_error)

    print(f"{name} MAPE: {mape:.2f}%")

    ax.scatter(
        actual[indices] * 1e12,
        predicted[indices] * 1e12,
        s=5,
        alpha=0.2
    )

    upper = max(actual.max(), predicted.max()) * 1e12
    ax.plot([0, upper], [0, upper], "k--", linewidth=1)

    ax.set_xlabel("Actual density (10⁻¹² kg/m³)")
    ax.set_ylabel("Predicted density (10⁻¹² kg/m³)")
    ax.set_title(f"{name} (MAPE = {mape:.2f}%)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(
    FIGURES / "predicted_vs_actual.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Saved predicted_vs_actual.png")
