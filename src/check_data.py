from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_parquet(ROOT / "results" / "processed_data.parquet")

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"Total observations: {len(df):,}")
print(f"Unique forecasting cases: {df['case_id'].nunique()}")
print(f"Number of columns: {len(df.columns)}")

print("\nObservations per case:")
print(df.groupby("case_id").size().describe())

print("\nDensity statistics:")
print(df["density"].describe())

print("\nForecast lead time (hours):")
print(df["lead_hours"].describe())

print("\nMissing values by feature:")
print(df.isna().sum().sort_values(ascending=False).head(20))

print("\nNon-finite values:")
numeric = df.select_dtypes(include=[np.number])
print((~np.isfinite(numeric)).sum().sort_values(ascending=False).head(20))

print("\nCases with fewer than 100 valid observations:")
counts = df.groupby("case_id").size()
print(counts[counts < 100])
