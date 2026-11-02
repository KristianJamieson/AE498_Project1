from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

OMNI_FEATURES = [
    "f10.7_index",
    "Kp_index",
    "Dst_index_nT",
    "ap_index_nT",
    "SW_Plasma_Speed_km_s",
]

ORBIT_FEATURES = [
    "Semi-major Axis (km)",
    "Eccentricity",
    "Inclination (deg)",
    "RAAN (deg)",
    "Argument of Perigee (deg)",
    "True Anomaly (deg)",
    "Latitude (deg)",
    "Longitude (deg)",
    "Altitude (km)",
]

TARGET = "Orbit Mean Density (kg/m^3)"


def extract_id(path):
    name = path.stem

    if "initial_states" in name:
        raise ValueError("Initial-state files contain multiple IDs.")

    match = re.search(r"-(\d{5})-", name)
    if not match:
        raise ValueError(f"Cannot identify case ID: {name}")

    return int(match.group(1))


def load_initial_states():
    paths = sorted((ROOT / "initial_states").glob("*.csv"))
    states = pd.concat(
        [pd.read_csv(path) for path in paths],
        ignore_index=True,
    )

    states = states.set_index("File ID")
    return states


def clean_omni_column(series):
    values = pd.to_numeric(series, errors="coerce")

    # Only remove known missing-value sentinels.
    # Additional variable-specific validation can be added later.
    values = values.mask(values.isin([
        9999, 99999, 999999, 9999999,
        999.9, 9999.9, 99999.9
    ]))

    return values


def summarize_omni(path, cutoff):
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Only use measurements available before forecast initialization.
    df = df[df["Timestamp"] <= cutoff].copy()

    if df.empty:
        raise ValueError(f"No historical OMNI2 data: {path}")

    features = {}

    for col in OMNI_FEATURES:
        values = clean_omni_column(df[col])

        features[f"{col}_mean"] = values.mean()
        features[f"{col}_std"] = values.std()
        features[f"{col}_last"] = (
            values.dropna().iloc[-1]
            if values.notna().any()
            else np.nan
        )

    return features


def build_dataset(max_cases=100):
    states = load_initial_states()

    omni_paths = {
        extract_id(p): p
        for p in (ROOT / "omni2").glob("*.csv")
    }

    density_paths = {
        extract_id(p): p
        for p in (ROOT / "sat_density").glob("*.csv")
    }

    common_ids = sorted(
        set(states.index)
        & set(omni_paths)
        & set(density_paths)
    )

    if max_cases is not None:
        common_ids = common_ids[:max_cases]

    rows = []

    for i, case_id in enumerate(common_ids):

        if i % 100 == 0:
            print(
                f"Processing case {i + 1}/{len(common_ids)} "
                f"({100 * (i + 1) / len(common_ids):.1f}%)",
                flush=True
            )
        state = states.loc[case_id].copy()

        # Convert orbital features to numeric.
        for col in ORBIT_FEATURES:
            state[col] = pd.to_numeric(state[col], errors="coerce")

        # Reject non-finite values and extreme missing-data sentinels.
        orbit_values = state[ORBIT_FEATURES].to_numpy(dtype=float)

        if not np.isfinite(orbit_values).all():
            continue

        if (np.abs(orbit_values) >= 1e20).any():
            continue

        # Basic physical validity checks.
        if not (6378 < state["Semi-major Axis (km)"] < 100000):
            continue

        if not (0 <= state["Eccentricity"] < 1):
            continue

        if not (100 <= state["Altitude (km)"] <= 2000):
            continue

        start_time = pd.to_datetime(state["Timestamp"])

        try:
            weather = summarize_omni(
                omni_paths[case_id],
                start_time,
            )

            density = pd.read_csv(density_paths[case_id])
            density[TARGET] = pd.to_numeric(density[TARGET], errors="coerce")
            density["Timestamp"] = pd.to_datetime(
                density["Timestamp"]
            )

            base_features = {
                "case_id": case_id,
                **{col: state[col] for col in ORBIT_FEATURES},
                **weather,
            }

            for _, observation in density.iterrows():
                target = pd.to_numeric(observation[TARGET], errors="coerce")

                if pd.isna(target) or not np.isfinite(target):
                    continue

                if target <= 0 or target >= 9.99e32:
                    continue

                lead_hours = (
                    observation["Timestamp"] - start_time
                ).total_seconds() / 3600

                if lead_hours < 0:
                    continue

                rows.append({
                    **base_features,
                    "lead_hours": lead_hours,
                    "density": target,
                })

        except (ValueError, KeyError, IndexError) as exc:
            print(f"Skipping case {case_id}: {exc}")

    result = pd.DataFrame(rows)

    output = ROOT / "results"
    output.mkdir(exist_ok=True)

    result.to_parquet(
        output / "processed_data.parquet",
        index=False,
    )

    print(f"\nProcessed {len(common_ids)} candidate cases")
    print(f"Total observations: {len(result)}")
    print(f"Features: {result.shape[1] - 2}")
    print(result.head())

    return result


if __name__ == "__main__":
    build_dataset(max_cases=None)
