from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "results" / "processed_data.parquet")

weather_columns = [
    c for c in df.columns
    if any(term in c for term in [
        "f10.7", "Kp_", "Dst_", "ap_", "SW_Plasma"
    ])
]

summary = df[weather_columns].describe(
    percentiles=[0.01, 0.5, 0.99]
).T

print(summary[[
    "min", "1%", "50%", "99%", "max"
]].to_string())

print("\nPotentially suspicious values:")
for col in weather_columns:
    suspicious = (
        (df[col].abs() >= 1e5)
        | (df[col] == 999)
        | (df[col] == 9999)
    )

    count = suspicious.sum()

    if count:
        print(f"{col}: {count:,} suspicious observations")
