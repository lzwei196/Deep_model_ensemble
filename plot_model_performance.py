import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import os
from datetime import datetime
import glob

def load_observed_data(year_folder):
    """Load observed data from year folder"""
    observed_data = {}

    for file in glob.glob(os.path.join(year_folder, "*.txt")):
        df = pd.read_csv(file, sep='\t')

        # Extract station name from filename
        filename = os.path.basename(file)
        if 'bengbu' in filename.lower():
            station = 'Bengbu'
        elif 'xixian' in filename.lower():
            station = 'Xixian'
        elif 'linyi' in filename.lower():
            station = 'Linyi'
        elif 'huaibin' in filename.lower():
            station = 'Huaibin'
        elif 'wangjiaba' in filename.lower():
            station = 'Wangjiaba'
        elif 'jiangjiaji' in filename.lower():
            station = 'Jiangjiaji'
        elif 'zhoukou' in filename.lower():
            station = 'Zhoukou'
        else:
            continue

        df['dates'] = pd.to_datetime(df['dates'])
        df = df[['dates', 'Q']].rename(columns={'Q': 'observed_flow'})
        observed_data[station] = df

    return observed_data

def load_predictions(station_name):
    """Load extended predictions for a station"""
    pred_file = f"{station_name}_extended_predictions/extended_predictions_clean.csv"

    if os.path.exists(pred_file):
        df = pd.read_csv(pred_file)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None

def create_interactive_plot():
    """Create interactive matplotlib plot"""

    # Load all available data
    years = ['2003', '2007', '2020']
    all_observed = {}
    all_predictions = {}

    # Get list of stations with predictions
    prediction_dirs = glob.glob("*_extended_predictions")
    stations_with_predictions = [d.replace('_extended_predictions', '') for d in prediction_dirs]

    print("Loading data...")

    # Load observed data for each year
    for year in years:
        year_path = f"observed_year/{year}"
        if os.path.exists(year_path):
            all_observed[year] = load_observed_data(year_path)

    # Load predictions for each station
    for station in stations_with_predictions:
        pred_data = load_predictions(station)
        if pred_data is not None:
            all_predictions[station] = pred_data
            print(f"Loaded predictions for {station}")

    # Get list of models from first station's predictions
    if all_predictions:
        first_station = list(all_predictions.keys())[0]
        model_columns = [col for col in all_predictions[first_station].columns
                        if col.endswith('_predicted') or col in ['ensemble_mean', 'ensemble_median']]
        model_names = [col.replace('_predicted', '').replace('_', ' ').title() for col in model_columns]
    else:
        print("No predictions found!")
        return

    # Create the figure and initial plot
    fig = plt.figure(figsize=(15, 10))

    # Main plot axes
    ax_main = plt.axes([0.1, 0.35, 0.75, 0.55])
    ax_scatter = plt.axes([0.1, 0.05, 0.35, 0.25])
    ax_metrics = plt.axes([0.5, 0.05, 0.35, 0.25])

    # Initialize with first available data
    initial_year = years[0]
    initial_station = None
    initial_model = model_columns[0]

    # Find first station that has both predictions and observed data
    for station in stations_with_predictions:
        if station in all_observed.get(initial_year, {}):
            initial_station = station
            break

    if initial_station is None:
        print(f"No matching station found for year {initial_year}")
        return

    # Plot initial data
    def update_plot(year, station, model):
        ax_main.clear()
        ax_scatter.clear()
        ax_metrics.clear()

        if station not in all_predictions:
            ax_main.text(0.5, 0.5, f'No predictions available for {station}',
                        ha='center', va='center', transform=ax_main.transAxes)
            return

        if year not in all_observed or station not in all_observed[year]:
            ax_main.text(0.5, 0.5, f'No observed data for {station} in {year}',
                        ha='center', va='center', transform=ax_main.transAxes)
            return

        # Get data
        pred_df = all_predictions[station]
        obs_df = all_observed[year][station]

        # Merge on date
        merged = pd.merge(obs_df, pred_df, left_on='dates', right_on='date', how='inner')

        if merged.empty:
            ax_main.text(0.5, 0.5, 'No overlapping dates found',
                        ha='center', va='center', transform=ax_main.transAxes)
            return

        # Time series plot
        ax_main.plot(merged['dates'], merged['observed_flow_x'], 'b-', label='Observed', linewidth=2)

        if model == 'all':
            # Plot all models
            colors = plt.cm.tab10(np.linspace(0, 1, len(model_columns)))
            for i, col in enumerate(model_columns):
                if col in merged.columns:
                    ax_main.plot(merged['dates'], merged[col], alpha=0.7,
                               label=col.replace('_predicted', '').replace('_', ' ').title(),
                               color=colors[i])
        else:
            # Plot single model
            if model in merged.columns:
                ax_main.plot(merged['dates'], merged[model], 'r-', label=model.replace('_predicted', '').replace('_', ' ').title(), linewidth=2)

        ax_main.set_xlabel('Date')
        ax_main.set_ylabel('Flow (m³/s)')
        ax_main.set_title(f'{station} - {year} - Model Performance')
        ax_main.legend(loc='upper right')
        ax_main.grid(True, alpha=0.3)

        # Scatter plot
        if model != 'all' and model in merged.columns:
            ax_scatter.scatter(merged['observed_flow_x'], merged[model], alpha=0.5, s=10)

            # Add 1:1 line
            max_val = max(merged['observed_flow_x'].max(), merged[model].max())
            min_val = min(merged['observed_flow_x'].min(), merged[model].min())
            ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

            ax_scatter.set_xlabel('Observed Flow')
            ax_scatter.set_ylabel('Predicted Flow')
            ax_scatter.set_title('Observed vs Predicted')
            ax_scatter.grid(True, alpha=0.3)

            # Calculate metrics
            obs = merged['observed_flow_x'].values
            pred = merged[model].values

            # Remove NaN values
            mask = ~(np.isnan(obs) | np.isnan(pred))
            obs = obs[mask]
            pred = pred[mask]

            if len(obs) > 0:
                # Calculate metrics
                mae = np.mean(np.abs(obs - pred))
                rmse = np.sqrt(np.mean((obs - pred) ** 2))
                mape = np.mean(np.abs((obs - pred) / obs)) * 100 if np.all(obs != 0) else np.nan

                # Nash-Sutcliffe Efficiency
                nse = 1 - (np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2))

                # R-squared
                correlation = np.corrcoef(obs, pred)[0, 1] if len(obs) > 1 else 0
                r2 = correlation ** 2

                # Display metrics
                metrics_text = f"Performance Metrics:\n"
                metrics_text += f"MAE: {mae:.2f}\n"
                metrics_text += f"RMSE: {rmse:.2f}\n"
                metrics_text += f"MAPE: {mape:.2f}%\n" if not np.isnan(mape) else "MAPE: N/A\n"
                metrics_text += f"NSE: {nse:.3f}\n"
                metrics_text += f"R²: {r2:.3f}\n"
                metrics_text += f"Correlation: {correlation:.3f}"

                ax_metrics.text(0.1, 0.5, metrics_text, transform=ax_metrics.transAxes,
                              fontsize=10, verticalalignment='center',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax_metrics.axis('off')

        plt.draw()

    # Initial plot
    update_plot(initial_year, initial_station, initial_model)

    # Add radio buttons for year selection
    rax_year = plt.axes([0.88, 0.7, 0.1, 0.15])
    radio_year = RadioButtons(rax_year, years, active=years.index(initial_year))

    # Add radio buttons for station selection
    available_stations = [s for s in stations_with_predictions
                         if any(s in all_observed.get(y, {}) for y in years)]
    rax_station = plt.axes([0.88, 0.45, 0.1, 0.2])
    radio_station = RadioButtons(rax_station, available_stations,
                                 active=available_stations.index(initial_station) if initial_station in available_stations else 0)

    # Add radio buttons for model selection
    model_options = ['all'] + model_columns
    model_labels = ['All Models'] + model_names
    rax_model = plt.axes([0.88, 0.15, 0.1, 0.25])
    radio_model = RadioButtons(rax_model, model_labels, active=model_options.index(initial_model))

    # Update function for radio buttons
    def update(val):
        year = radio_year.value_selected
        station = radio_station.value_selected
        model_idx = model_labels.index(radio_model.value_selected)
        model = model_options[model_idx]
        update_plot(year, station, model)

    radio_year.on_clicked(update)
    radio_station.on_clicked(update)
    radio_model.on_clicked(update)

    plt.suptitle('Model Performance Comparison - Interactive Dashboard', fontsize=14, fontweight='bold')

    return fig

if __name__ == "__main__":
    print("Creating interactive model performance plots...")
    fig = create_interactive_plot()

    # Keep the plot open
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        print("\nClosing plots...")
    finally:
        plt.close('all')