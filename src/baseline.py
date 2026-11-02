from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

df = pd.read_parquet(RESULTS / "processed_data.parquet")
splits = pd.read_csv(RESULTS / "split_assignments.csv")

df = df.merge(
    splits[["case_id", "split"]],
    on="case_id",
    how="left",
    validate="many_to_one"
)

train = df[df["split"] == "train"]
validation = df[df["split"] == "validation"]
test = df[df["split"] == "test"]

# Constant baseline fitted using training data only.
baseline_density = train["density"].median()

print(f"Training median density: {baseline_density:.6e} kg/m^3")

for name, subset in [
    ("Validation", validation),
    ("Test", test)
]:
    actual = subset["density"].to_numpy()
    predicted = np.full_like(actual, baseline_density)

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    mape = 100 * np.mean(np.abs(actual - predicted) / actual)

    print(f"\n{name}")
    print(f"RMSE: {rmse:.6e} kg/m^3")
    print(f"MAE:  {mae:.6e} kg/m^3")
    print(f"MAPE: {mape:.2f}%")
