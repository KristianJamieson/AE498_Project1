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

features = [
    c for c in df.columns
    if c not in ["case_id", "density", "start_time"]
]

models = {
    "Random Forest": joblib.load(RESULTS / "models" / "rf.joblib"),
    "LightGBM": joblib.load(RESULTS / "models" / "lightgbm.joblib"),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, (name, model) in zip(axes, models.items()):
    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False).head(12)

    importance = importance.sort_values()

    ax.barh(importance.index, importance.values)
    ax.set_title(name)
    ax.set_xlabel("Feature importance")

fig.suptitle("Top 12 Atmospheric Density Forecasting Features")
plt.tight_layout()

plt.savefig(
    FIGURES / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved results/figures/feature_importance.png")

# Ridge regression standardized coefficients
ridge = joblib.load(RESULTS / "models" / "ridge.joblib")

coefficients = pd.Series(
    ridge.named_steps["ridge"].coef_,
    index=features
)

top = coefficients.reindex(
    coefficients.abs().sort_values(ascending=False).head(12).index
).sort_values()

plt.figure(figsize=(9, 6))

plt.barh(top.index, top.values)

plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Standardized Ridge coefficient")
plt.title("Ridge Regression: Largest Coefficients")
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()

plt.savefig(
    FIGURES / "ridge_coefficients.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Saved results/figures/ridge_coefficients.png")