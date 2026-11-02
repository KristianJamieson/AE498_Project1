# Atmospheric Density Forecasting Using Machine Learning

**AE 498 — Project 1: Data-Driven Modeling**  
University of Illinois Urbana-Champaign

## Overview

This project investigates machine-learning approaches for forecasting orbit-averaged thermospheric density in low Earth orbit (LEO). Accurate atmospheric density predictions are important for satellite drag estimation, orbit prediction, and collision-risk assessment.

Using the publicly available STORM-AI dataset, three regression methods are evaluated:

- Random Forest
- LightGBM
- Ridge Regression

Each model predicts atmospheric density over a 72-hour forecast horizon using initial orbital conditions and historical space-weather measurements.

The objective is to compare predictive accuracy, computational cost, and generalization under a chronological train/validation/test split.

## Dataset

The dataset was released for the 2025 MIT ARCLab STORM-AI Challenge.

**Download:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/U6K6MJ

Download and extract the following archives:

- `initial_states.zip`
- `omni2.zip`
- `sat_density.zip`

Place the extracted folders in the root of this repository:

    AE498_Project1/
    ├── initial_states/
    ├── omni2/
    ├── sat_density/
    ├── src/
    ├── results/
    ├── README.md
    └── requirements.txt

The GOES dataset is not used in this implementation and does not need to be downloaded.

The input data include initial satellite orbital elements, geodetic coordinates, and 60 days of historical space-weather observations. The prediction targets are orbit-averaged atmospheric densities sampled at 10-minute intervals over 72 hours.

## Installation

Python 3.10 or newer is recommended.

Create and activate a virtual environment.

**Windows PowerShell:**

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Running the Project

Run the following commands from the repository root.

### 1. Preprocess the Dataset

    python src/preprocess.py

This loads the downloaded STORM-AI files, joins forecasting cases by identifier, removes invalid observations, and constructs orbital and space-weather features.

The processed dataset is saved as:

`results/processed_data.parquet`

### 2. Train and Evaluate Models

    python src/train.py --model both

The `both` option currently trains all three models: Random Forest, LightGBM, and Ridge Regression.

The script uses a chronological split of forecasting cases into training (70%), validation (15%), and testing (15%), with a 72-hour separation between partitions.

Targets are transformed using base-10 logarithms during training. Predictions are converted back into physical density units before RMSE and MAE are calculated.

Outputs include:

- `results/metrics.csv`
- `results/split_assignments.csv`
- Trained models in `results/models/`

### 3. Evaluate the Constant Baseline

    python src/baseline.py

This evaluates a constant predictor equal to the median atmospheric density in the training set.

### 4. Generate Figures

    python src/plot_results.py
    python src/prediction_scatter.py
    python src/feature_importance.py

Figures are saved in `results/figures/`.

## Model Results

The following results were obtained on the held-out chronological test partition containing 518,691 valid density observations.

| Model | Test RMSE (kg/m³) | Test MAE (kg/m³) | Training Time (s) |
|---|---:|---:|---:|
| Constant Baseline* | 1.0127 × 10⁻¹² | 1.0115 × 10⁻¹² | — |
| Random Forest | 2.3241 × 10⁻¹³ | 1.2445 × 10⁻¹³ | 283.54 |
| LightGBM | 1.8570 × 10⁻¹³ | 1.1421 × 10⁻¹³ | 36.56 |
| Ridge Regression | **8.7174 × 10⁻¹⁴** | **5.2454 × 10⁻¹⁴** | **2.66** |

*The constant-baseline results were calculated before the final data-cleaning pass and should be recomputed before treating them as a matched comparison.*

Ridge regression achieved the lowest test error among the three trained models. Relative to LightGBM, it reduced test RMSE by approximately 53.1%, while requiring substantially less training time.

The chronological test partition also contained considerably lower atmospheric densities than the training partition, creating a distribution shift that may help explain differences in model performance.

These results are specific to this project's preprocessing, feature selection, and evaluation split. They should not be interpreted as official STORM-AI leaderboard results.

## Evaluation

Models are evaluated using:

- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Training time
- RMSE as a function of forecast lead time

The published STORM-AI competition used an MSIS-relative, time-weighted evaluation metric and separate hidden test sets. Therefore, numerical results from this project are not directly comparable with the official competition scores.

## Limitations

This implementation uses summary statistics of historical space-weather data rather than full time-series inputs. It does not incorporate GOES X-ray observations, empirical atmospheric-model corrections, or orbital propagation.

The chronological split tests generalization to later forecasting cases, but historical 60-day input windows may overlap between partitions.

The forecast horizon is represented as an input feature; the model does not explicitly simulate the time evolution of thermospheric density.

Future improvements could incorporate empirical atmospheric-density models such as NRLMSIS, more detailed temporal features, storm-specific evaluation, and recurrent or sequence-based architectures.

## References

Sanchez-Hurtado, S., et al. (2026). *AI Challenge for Satellite Tracking and Orbit Resilience Modeling (STORM-AI): Data Set, Design, and Results*. Space Weather, 24, e2026SW004994. https://doi.org/10.1029/2026SW004994

STORM-AI Dataset: https://doi.org/10.7910/DVN/U6K6MJ

STORM-AI Development Kit: https://github.com/ARCLab-MIT/STORM-AI-devkit-2025
