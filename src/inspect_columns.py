from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def inspect_csv(path):
    print("\n" + "=" * 80)
    print(f"FILE: {path.relative_to(ROOT)}")
    print("=" * 80)

    df = pd.read_csv(path)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print("\nLast 2 rows:")
    print(df.tail(2).to_string(index=False))

    print("\nMissing values:")
    print(df.isna().sum().to_string())

# Inspect the first initial-state file
initial_files = sorted((ROOT / "initial_states").glob("*.csv"))
inspect_csv(initial_files[0])

# Inspect the first OMNI2 file
omni_files = sorted((ROOT / "omni2").glob("*.csv"))
inspect_csv(omni_files[0])

# Inspect the first satellite-density file
density_files = sorted((ROOT / "sat_density").glob("*.csv"))
inspect_csv(density_files[0])
