from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(
    dataset: pd.DataFrame,
    prediction_column: str = 'cnt',
    date_column: str = 'dteday',
    hour_column: str = 'hr',
    output_path: str | Path | None = None,
):
    """Plot predicted demand over time and optionally save the figure."""
    if prediction_column not in dataset.columns:
        raise ValueError(f'Dataset is missing the prediction column: {prediction_column}')

    plot_data = dataset.copy()
    if date_column in plot_data.columns and hour_column in plot_data.columns:
        timestamps = pd.to_datetime(plot_data[date_column]) + pd.to_timedelta(
            plot_data[hour_column], unit='h'
        )
        x_values = timestamps
        x_label = 'Timestamp'
    else:
        x_values = plot_data.index
        x_label = 'Row Index'

    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(x_values, plot_data[prediction_column], color='#3498db', linewidth=1.5)
    axis.set_title('Predicted Hourly Bike Demand')
    axis.set_xlabel(x_label)
    axis.set_ylabel('Predicted Rental Count (cnt)')
    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches='tight')

    return figure, axis