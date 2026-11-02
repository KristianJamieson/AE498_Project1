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
    how="left"
)

for name in ["train", "validation", "test"]:
    subset = df[df["split"] == name]

    print(f"\n{name.upper()}")
    print("-" * 50)

    print(f"Observations: {len(subset):,}")
    print(f"Cases: {subset['case_id'].nunique():,}")

    print("Density:")
    print(subset["density"].describe())

    print("Semi-major axis:")
    print(subset["Semi-major Axis (km)"].describe())

    print("Altitude:")
    print(subset["Altitude (km)"].describe())

    print("F10.7:")
    print(subset["f10.7_index_mean"].describe())

    print("Percentage with density < 1e-12:")
    print(100 * (subset["density"] < 1e-12).mean())
