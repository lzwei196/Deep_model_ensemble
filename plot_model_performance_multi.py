import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
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

        # Extract station name from filename (case-insensitive matching)
        filename = os.path.basename(file).lower()
        if 'bengbu' in filename:
            station = 'Bengbu'
        elif 'xixian' in filename:
            station = 'Xixian'
        elif 'linyi' in filename:
            station = 'Linyi'
        elif 'huaibin' in filename:
            station = 'Huaibin'
        elif 'wangjiaba' in filename:
            station = 'Wangjiaba'
        elif 'jiangjiaji' in filename:
            station = 'Jiangjiaji'
        elif 'zhoukou' in filename:
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


def create_station_plot(station, all_observed, all_predictions):
    """Create interactive plot for a single station"""

    years = sorted(all_observed.keys())

    # Check if station has predictions
    if station not in all_predictions:
        print(f"No predictions available for {station}")
        return None

    pred_df = all_predictions[station]

    # Get model columns
    model_columns = [col for col in pred_df.columns
                     if col.endswith('_predicted') or col in ['ensemble_mean', 'ensemble_median']]

    # Remove empty model columns (like LSTM if it's all NaN)
    model_columns = [col for col in model_columns if not pred_df[col].isna().all()]

    model_names = [col.replace('_predicted', '').replace('_', ' ').title() for col in model_columns]

    # Create figure for this station
    fig = plt.figure(figsize=(16, 10), num=f"{station} Model Performance")

    # Main plot axes
    ax_main = plt.axes([0.1, 0.35, 0.75, 0.55])
    ax_scatter = plt.axes([0.1, 0.05, 0.35, 0.25])
    ax_metrics = plt.axes([0.5, 0.05, 0.35, 0.25])

    # Find initial year where this station has observed data
    initial_year = None
    for year in years:
        if station in all_observed[year]:
            initial_year = year
            break

    if initial_year is None:
        plt.close(fig)
        print(f"No observed data found for {station}")
        return None

    initial_model = 'all'

    # Plot function
    def update_plot(year, model_selection):
        ax_main.clear()
        ax_scatter.clear()
        ax_metrics.clear()

        if year not in all_observed or station not in all_observed[year]:
            ax_main.text(0.5, 0.5, f'No observed data for {station} in {year}',
                         ha='center', va='center', transform=ax_main.transAxes)
            return

        # Get data
        obs_df = all_observed[year][station]

        # Merge on date
        merged = pd.merge(obs_df, pred_df, left_on='dates', right_on='date', how='inner', suffixes=('_obs', '_pred'))

        if merged.empty:
            ax_main.text(0.5, 0.5, 'No overlapping dates found',
                         ha='center', va='center', transform=ax_main.transAxes)
            return

        # Treat negative predicted values as 0 for all model columns
        for col in model_columns:
            if col in merged.columns:
                merged[col] = merged[col].clip(lower=0)

        # Time series plot
        ax_main.plot(merged['dates'], merged['observed_flow_obs'], 'b-', label='Observed', linewidth=2, zorder=10)

        if model_selection == 'all':
            # Plot all models with different colors and styles
            colors = plt.cm.tab10(np.linspace(0, 1, len(model_columns)))
            line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

            for i, col in enumerate(model_columns):
                if col in merged.columns and not merged[col].isna().all():
                    label = col.replace('_predicted', '').replace('_', ' ').title()
                    ax_main.plot(merged['dates'], merged[col],
                                 linestyle=line_styles[i % len(line_styles)],
                                 color=colors[i],
                                 label=label,
                                 linewidth=1.5,
                                 alpha=0.8)

            ax_main.set_title(f'{station} - {year} - All Models Performance')

            # Show combined metrics for all models
            metrics_text = "Model Performance Summary:\n\n"
            for col in model_columns:
                if col in merged.columns and not merged[col].isna().all():
                    obs = merged['observed_flow_obs'].values
                    pred = merged[col].values
                    mask = ~(np.isnan(obs) | np.isnan(pred))
                    if mask.any():
                        obs_clean = obs[mask]
                        pred_clean = pred[mask]
                        rmse = np.sqrt(np.mean((obs_clean - pred_clean) ** 2))

                        # Nash-Sutcliffe Efficiency
                        nse = 1 - (np.sum((obs_clean - pred_clean) ** 2) / np.sum(
                            (obs_clean - np.mean(obs_clean)) ** 2))

                        if len(obs_clean) > 1:
                            corr = np.corrcoef(obs_clean, pred_clean)[0, 1]
                        else:
                            corr = 0
                        model_name = col.replace('_predicted', '').replace('_', ' ').title()
                        metrics_text += f"{model_name}:\n"
                        metrics_text += f"  RMSE: {rmse:.1f}, NSE: {nse:.3f}, R: {corr:.3f}\n"

            ax_metrics.text(0.05, 0.5, metrics_text, transform=ax_metrics.transAxes,
                            fontsize=9, verticalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            ax_metrics.axis('off')

            # Create a combined scatter plot in ax_scatter
            ax_scatter.text(0.5, 0.5, 'Select individual model\nfor scatter plot',
                            ha='center', va='center', transform=ax_scatter.transAxes,
                            fontsize=11, color='gray')
            ax_scatter.set_xlim(0, 1)
            ax_scatter.set_ylim(0, 1)
            ax_scatter.axis('off')

        else:
            # Plot single model
            if model_selection in merged.columns and not merged[model_selection].isna().all():
                ax_main.plot(merged['dates'], merged[model_selection], 'r-',
                             label=model_selection.replace('_predicted', '').replace('_', ' ').title(),
                             linewidth=2)

                # Scatter plot for single model
                ax_scatter.scatter(merged['observed_flow_obs'], merged[model_selection], alpha=0.5, s=10)

                # Add 1:1 line
                max_val = max(merged['observed_flow_obs'].max(), merged[model_selection].max())
                min_val = min(merged['observed_flow_obs'].min(), merged[model_selection].min())
                ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

                ax_scatter.set_xlabel('Observed Flow (m³/s)')
                ax_scatter.set_ylabel('Predicted Flow (m³/s)')
                ax_scatter.set_title('Observed vs Predicted')
                ax_scatter.grid(True, alpha=0.3)

                # Calculate metrics
                obs = merged['observed_flow_obs'].values
                pred = merged[model_selection].values

                # Remove NaN values
                mask = ~(np.isnan(obs) | np.isnan(pred))
                obs = obs[mask]
                pred = pred[mask]

                if len(obs) > 0:
                    # Calculate metrics
                    mae = np.mean(np.abs(obs - pred))
                    rmse = np.sqrt(np.mean((obs - pred) ** 2))

                    # MAPE with zero handling
                    non_zero_mask = obs != 0
                    if np.any(non_zero_mask):
                        mape = np.mean(np.abs((obs[non_zero_mask] - pred[non_zero_mask]) / obs[non_zero_mask])) * 100
                    else:
                        mape = np.nan

                    # Nash-Sutcliffe Efficiency
                    nse = 1 - (np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2))

                    # R-squared
                    if len(obs) > 1:
                        correlation = np.corrcoef(obs, pred)[0, 1]
                        r2 = correlation ** 2
                    else:
                        correlation = 0
                        r2 = 0

                    # Display metrics
                    metrics_text = f"Performance Metrics:\n"
                    metrics_text += f"MAE: {mae:.2f}\n"
                    metrics_text += f"RMSE: {rmse:.2f}\n"
                    metrics_text += f"MAPE: {mape:.2f}%\n" if not np.isnan(mape) else "MAPE: N/A\n"
                    metrics_text += f"NSE: {nse:.3f}\n"
                    metrics_text += f"R²: {r2:.3f}\n"
                    metrics_text += f"Correlation: {correlation:.3f}\n"
                    metrics_text += f"N samples: {len(obs)}"

                    ax_metrics.text(0.1, 0.5, metrics_text, transform=ax_metrics.transAxes,
                                    fontsize=11, verticalalignment='center',
                                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax_metrics.axis('off')
                ax_main.set_title(
                    f'{station} - {year} - {model_selection.replace("_predicted", "").replace("_", " ").title()}')
            else:
                ax_main.text(0.5, 0.5, f'No data available for {model_selection}',
                             ha='center', va='center', transform=ax_main.transAxes)
                ax_scatter.axis('off')
                ax_metrics.axis('off')

        ax_main.set_xlabel('Date')
        ax_main.set_ylabel('Flow (m³/s)')
        ax_main.legend(loc='best', fontsize=9)
        ax_main.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        for label in ax_main.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

        plt.draw()

    # Initial plot
    update_plot(initial_year, initial_model)

    # Add radio buttons for year selection
    years_with_data = [y for y in years if station in all_observed.get(y, {})]
    if years_with_data:
        rax_year = plt.axes([0.88, 0.7, 0.1, 0.15])
        radio_year = RadioButtons(rax_year, years_with_data, active=years_with_data.index(initial_year))

    # Add radio buttons for model selection
    model_options = ['all'] + model_columns
    model_labels = ['All Models'] + model_names
    rax_model = plt.axes([0.88, 0.3, 0.1, 0.35])
    radio_model = RadioButtons(rax_model, model_labels, active=0)

    # Update function for radio buttons
    def update(val):
        year = radio_year.value_selected
        model_idx = model_labels.index(radio_model.value_selected)
        model = model_options[model_idx]
        update_plot(year, model)

    radio_year.on_clicked(update)
    radio_model.on_clicked(update)

    plt.suptitle(f'{station} - Model Performance Comparison Dashboard', fontsize=14, fontweight='bold')

    return fig


def main():
    """Main function to create plots for all stations"""

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
            print(f"Loaded observed data for year {year}: {list(all_observed[year].keys())}")

    # Load predictions for each station
    for station in stations_with_predictions:
        pred_data = load_predictions(station)
        if pred_data is not None:
            all_predictions[station] = pred_data
            print(f"Loaded predictions for {station}")

    if not all_predictions:
        print("No predictions found!")
        return

    # Create plot for each station that has both predictions and observed data
    figures = []
    for station in stations_with_predictions:
        # Check if station has any observed data
        has_observed = any(station in all_observed.get(year, {}) for year in years)

        if has_observed:
            print(f"\nCreating plot for {station}...")
            fig = create_station_plot(station, all_observed, all_predictions)
            if fig is not None:
                figures.append(fig)
        else:
            print(f"\n{station} has predictions but no matching observed data")

    if figures:
        print(f"\nCreated {len(figures)} interactive plot window(s)")
        print("Use the radio buttons to switch between years and models")
        print("Close all windows or press Ctrl+C to exit")
    else:
        print("\nNo plots could be created - no matching data found")


if __name__ == "__main__":
    print("Creating interactive model performance plots...")
    main()

    # Keep plots open
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        print("\nClosing plots...")
    finally:
        plt.close('all')
