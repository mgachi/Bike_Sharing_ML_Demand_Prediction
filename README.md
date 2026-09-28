# Bike Sharing Demand Forecasting

This project explores and forecasts hourly bike rental demand using the UCI Bike
Sharing Dataset. The main workflow is implemented in
`eda_and_modeling.ipynb`, with reusable preprocessing and evaluation code in
`src/`.

## Project Structure

```text
data/
    hour.csv                  Hourly rental data
    day.csv                   Daily rental data
    Readme.txt                Original dataset documentation
    mock_test_data_3weeks.csv Feature-only sample input for inference
    mock_test_data_3weeks_predictions.csv
                               Sample input with predicted rental counts
src/
    __init__.py
    pipeline.py               Feature engineering and model pipeline
    evaluate.py               Mean Absolute Deviation calculation
    predict.py                Persisted-model inference and CSV output
    visualization.py          Prediction plotting helper
best_model.joblib             Persisted fitted model and metadata
requirements.txt                Python and notebook dependencies
eda_and_modeling.ipynb       EDA, tuning, evaluation, and plots
Report.pdf                   Project report
*.png                        Generated analysis and presentation figures
README.md
```

## Workflow

### 1. Exploratory Data Analysis

- Check row integrity, missing values, duplicate rows, and timestamp uniqueness.
- Inspect rental counts, temperature, humidity, and wind speed.
- Visualize hourly demand for working days versus weekends and holidays.
- Compare demand across weather conditions and temperature ranges.
- Inspect correlations among the main numeric features.
- Review high-demand and unusually low-demand commute periods.

### 2. Feature Engineering

Feature preparation in the notebook and the pipeline in `src/pipeline.py`:

- Adds sine and cosine encodings for hour and month cycles.
- One-hot encodes `season`, `weathersit`, and `weekday`.
- Standardizes numeric and engineered cyclical features.
- Drops identifiers and leakage features before modeling:
    `instant`, `dteday`, `casual`, `registered`, `cnt`, and `atemp`.
- Keeps all preprocessing inside the sklearn pipeline so transformations are
    fitted only on the relevant training fold during cross-validation.

### 3. Model Development and Evaluation

- Uses a chronological split with 85% of the data for training and rolling
    validation, and the final 15% as an unseen holdout period.
- Uses five chronological `TimeSeriesSplit` folds.
- Runs two separate hyperparameter searches:
    - squared-error and Poisson loss
    - quantile loss with quantiles 0.5, 0.7, and 0.9
- Selects the best search automatically using cross-validated negative MAE.
- Saves the selected model and training-data fingerprint to `best_model.joblib`.
- Reuses the saved model when the current `data/hour.csv` matches that fingerprint.
- Stores model metadata including the source-data hash, training date range,
    selected search, parameters, and cross-validation score.
- Evaluates the selected model on the holdout period using Mean Absolute
    Deviation (MAD), equivalent to mean absolute error here.
- Compares the model with a historical hourly lookup baseline based only on
    the training period.

### 4. Operational Visualizations

The notebook generates:

- Model versus baseline accuracy benchmarks.
- Prediction-error distributions.
- Demand and error heatmaps by weekday and hour.
- Daily demand rhythm and a maintenance window.
- Weather-related differences in hourly rental demand.

### Inference Testing

Cell 7 creates a feature-only mock dataset at
`data/mock_test_data_3weeks.csv`. Cell 8 loads the persisted model, adds the
`cnt` predictions, saves them to
`data/mock_test_data_3weeks_predictions.csv`, and creates a prediction plot.
The reusable inference code in `src/predict.py` validates the required feature
schema, loads the saved model, and writes the enriched dataset. Plotting code
is isolated in `src/visualization.py`.

## Running the Notebook

Use Python 3.10 or newer, then create or activate a virtual environment from
the project root and install the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open `eda_and_modeling.ipynb` from the project root and run the cells in order.
The notebook expects the data files under `data/` and imports the local modules
from `src/`. Cell 4 writes the trained model cache to `best_model.joblib`; the
cache is reused only when the source data fingerprint and training split match.
The dependency file specifies minimum compatible versions; exact versions can
be pinned later if byte-for-byte reproducibility becomes necessary. The saved
model was created with scikit-learn 1.5.x, so the dependency range keeps it
within that compatible minor version.

## Outlook

The current project is suitable for analysis and a reproducible demonstration.
The following steps would make it easier for colleagues to operate and extend.

### Handoff and Reproducibility

- Move file paths, split ratios, model parameters, and output directories into
    one configuration file instead of keeping them in notebook cells.
- Add a short data dictionary and document the expected schema, units, valid
    ranges, and timestamp conventions for incoming data.
- Keep generated figures in a dedicated output directory and record the input
    data period, selected parameters, and model score alongside each run.
- Provide a small command-line entry point that runs the workflow without
    requiring colleagues to execute notebook cells manually.

### Automation

- Add an ingestion step that obtains the latest hourly rental and weather data,
    validates the schema, checks timestamp continuity, and reports missing or
    duplicated records before training or prediction.
- Schedule regular predictions, for example hourly or daily, and write the
    results to a database, dashboard, or shared operational report.
- Schedule retraining based on a fixed cadence or trigger it when recent model
    error exceeds an agreed threshold.
- Store each trained model with its training-period boundaries, feature
    configuration, hyperparameters, and validation metrics so that predictions
    can be reproduced and previous models can be restored.
- Add monitoring for prediction error, data drift, missing inputs, unusual
    demand spikes, and operational failures, with notifications when thresholds
    are exceeded.
- Separate exploratory notebook work from automated production code by moving
    data loading, training, prediction, and reporting into reusable Python
    modules.

### Operational Extensions

- Add prediction intervals or use the quantile models to provide uncertainty
    ranges alongside point forecasts.
- Compare the demand forecast with available fleet capacity to identify likely
    shortages and prioritize rebalancing actions.
- Add weather forecasts and calendar information, such as public holidays and
    major events, when producing forward-looking predictions.