# extended_prediction.py
"""
Extended prediction for complete forcing data - works with clean ML pipeline (no flow features)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


class ExtendedPredictor:
    """Generate predictions for full period using clean ML pipeline (no flow-based features)"""

    def __init__(self, pipeline, results_dir=None):
        self.pipeline = pipeline
        self.results_dir = results_dir

        # Load best model configuration if results_dir provided
        if results_dir and os.path.exists(results_dir):
            try:
                self.config = load_best_model_config(results_dir)
                self.best_model_name = self.config["best_model"]["name"]
                print(f"Loaded configuration - Best model: {self.best_model_name}")

                if self.best_model_name == 'LSTM':
                    self.sequence_length = self.config["model_specific"]["sequence_length"]
                    print(f"LSTM sequence length: {self.sequence_length}")
                else:
                    self.sequence_length = 0
            except Exception as e:
                print(f"Warning: Could not load model config: {e}")
                self.config = None
                self.best_model_name = None
                self.sequence_length = getattr(pipeline, 'sequence_length', 10)
        else:
            self.config = None
            self.best_model_name = None
            self.sequence_length = getattr(pipeline, 'sequence_length', 10)

    def prepare_full_period_data(self):
        """Prepare data for full period using clean feature engineering (no flow features)"""
        print("\n" + "=" * 60)
        print("PREPARING FULL PERIOD DATA")
        print("=" * 60)

        # Read the forcing data
        df_full = pd.read_csv(self.pipeline.filepath)
        df_full['time'] = pd.to_datetime(df_full['time'])

        print(f"Original data shape: {df_full.shape}")
        print(f"Data period: {df_full['time'].min()} to {df_full['time'].max()}")
        print(f"Columns: {list(df_full.columns)}")

        # Identify observed vs future periods
        valid_flow = (df_full['flow(m^3/s)'] > self.pipeline.flow_threshold) & \
                     (df_full['flow(m^3/s)'] != -99)

        print(f"Observed flow: {valid_flow.sum()} days")
        print(f"Missing flow (future): {(~valid_flow).sum()} days")

        if valid_flow.sum() > 0:
            last_obs_date = df_full[valid_flow]['time'].max()
            first_pred_date = df_full[~valid_flow]['time'].min() if (~valid_flow).sum() > 0 else None
            print(f"Last observed: {last_obs_date}")
            if first_pred_date:
                print(f"First prediction: {first_pred_date}")

        # Create features using pipeline's method (now clean - no flow features!)
        print("\nCreating features using clean pipeline...")
        try:
            features_list = self.pipeline._create_features(df_full)
            print(f"Features created: {len(features_list)}")

            # Verify no flow features (should be clean now)
            flow_features = [f for f in self.pipeline.feature_names if 'flow' in f.lower()]
            if flow_features:
                print(f"WARNING: Found {len(flow_features)} flow features: {flow_features[:5]}...")
                print("This suggests the pipeline still has flow-based features!")
            else:
                print("✓ Confirmed: Clean pipeline with no flow-based features")

        except Exception as e:
            print(f"Error in feature creation: {e}")
            return False

        # Remove rows with NaN (from lagged meteorological features)
        print(f"Before removing NaN: {len(df_full)} rows")
        df_clean = df_full.dropna(subset=self.pipeline.feature_names)
        print(f"After removing NaN: {len(df_clean)} rows")

        if len(df_clean) == 0:
            print("ERROR: No clean data available")
            return False

        # Prepare arrays
        X_full = df_clean[self.pipeline.feature_names].values
        X_full_scaled = self.pipeline.scaler_X.transform(X_full)

        dates_full = df_clean['time'].values
        y_observed = df_clean['flow(m^3/s)'].values.copy()

        # Mark observed vs prediction periods
        has_observed = (y_observed > self.pipeline.flow_threshold) & (y_observed != -99)
        y_observed[~has_observed] = np.nan

        # Store results
        self.X_full_scaled = X_full_scaled
        self.dates_full = dates_full
        self.y_observed = y_observed
        self.has_observed = has_observed
        self.df_clean = df_clean

        print(f"\nFinal data summary:")
        print(f"Period: {dates_full[0]} to {dates_full[-1]}")
        print(f"Total samples: {len(dates_full)}")
        print(f"Observed samples: {has_observed.sum()}")
        print(f"Prediction samples: {(~has_observed).sum()}")

        return True

    def generate_predictions(self):
        """Generate predictions for all models"""
        print("\n" + "=" * 60)
        print("GENERATING PREDICTIONS")
        print("=" * 60)

        predictions = {}

        for model_name, model in self.pipeline.models.items():
            print(f"\nPredicting with {model_name}...")

            try:
                if model_name == 'LSTM':
                    pred = self._predict_lstm(model)
                elif model_name == 'SVM':
                    pred = self._predict_svm(model)
                else:
                    # Tree-based models (XGBoost, LightGBM, RandomForest, etc.)
                    pred = model.predict(self.X_full_scaled)

                predictions[model_name] = pred

                # DEBUG: Check prediction values
                valid_preds = ~np.isnan(pred)
                if valid_preds.sum() > 0:
                    print(
                        f"  DEBUG: Prediction stats - Min: {np.min(pred[valid_preds]):.2f}, Max: {np.max(pred[valid_preds]):.2f}, Mean: {np.mean(pred[valid_preds]):.2f}")

                    # Check if predictions are reasonable
                    if np.max(pred[valid_preds]) > 20000:  # Very high flow values
                        print(f"  ⚠️  WARNING: Extremely high predictions detected!")

                    if np.min(pred[valid_preds]) < -1000:  # Negative flow values
                        print(f"  ⚠️  WARNING: Negative flow predictions detected!")

                # Show stats
                future_mask = ~self.has_observed
                future_preds = pred[future_mask & valid_preds]

                print(f"  ✓ Total predictions: {valid_preds.sum()}")
                print(f"  ✓ Future predictions: {len(future_preds)}")
                if len(future_preds) > 0:
                    print(f"    Future range: {np.min(future_preds):.2f} to {np.max(future_preds):.2f} m³/s")

                # Validate observed period performance
                if self.has_observed.sum() > 0:
                    obs_true = self.y_observed[self.has_observed]
                    obs_pred = pred[self.has_observed]
                    valid_obs = ~np.isnan(obs_true) & ~np.isnan(obs_pred)

                    if valid_obs.sum() > 5:  # Need minimum samples
                        obs_true_clean = obs_true[valid_obs]
                        obs_pred_clean = obs_pred[valid_obs]

                        # DEBUG: Check observed period predictions
                        print(
                            f"  DEBUG: Observed period - True range: {np.min(obs_true_clean):.2f} to {np.max(obs_true_clean):.2f}")
                        print(
                            f"  DEBUG: Observed period - Pred range: {np.min(obs_pred_clean):.2f} to {np.max(obs_pred_clean):.2f}")

                        # Calculate NSE for validation
                        ss_res = np.sum((obs_true_clean - obs_pred_clean) ** 2)
                        ss_tot = np.sum((obs_true_clean - np.mean(obs_true_clean)) ** 2)
                        nse = 1 - (ss_res / ss_tot)
                        rmse = np.sqrt(np.mean((obs_true_clean - obs_pred_clean) ** 2))

                        print(f"    Observed period performance: NSE={nse:.3f}, RMSE={rmse:.2f}")

            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")
                import traceback
                traceback.print_exc()
                predictions[model_name] = np.full(len(self.X_full_scaled), np.nan)

        self.predictions = predictions

        # ADDITIONAL DEBUG: Compare with original training data
        print(f"\nDEBUG: Data comparison:")
        print(
            f"Original observed flow range: {np.min(self.y_observed[~np.isnan(self.y_observed)]):.2f} to {np.max(self.y_observed[~np.isnan(self.y_observed)]):.2f}")
        if hasattr(self.pipeline, 'y_train'):
            print(f"Training data range: {np.min(self.pipeline.y_train):.2f} to {np.max(self.pipeline.y_train):.2f}")
        if hasattr(self.pipeline, 'y_test'):
            print(f"Test data range: {np.min(self.pipeline.y_test):.2f} to {np.max(self.pipeline.y_test):.2f}")

        return predictions

    def _predict_lstm(self, model):
        """LSTM prediction with sequences using saved configuration"""

        # Use sequence length from config if available
        if self.config and "model_specific" in self.config:
            sequence_length = self.config["model_specific"]["sequence_length"]
            print(f"    Using config sequence length: {sequence_length}")
        elif hasattr(self.pipeline, 'X_train_seq') and self.pipeline.X_train_seq is not None:
            sequence_length = self.pipeline.X_train_seq.shape[1]
            print(f"    Using training sequence length: {sequence_length}")
        elif 'LSTM' in self.pipeline.results and 'sequence_length' in self.pipeline.results['LSTM']:
            sequence_length = self.pipeline.results['LSTM']['sequence_length']
            print(f"    Using stored sequence length: {sequence_length}")
        else:
            sequence_length = getattr(self.pipeline, 'sequence_length', 10)
            print(f"    Using default sequence length: {sequence_length}")

        predictions = np.full(len(self.X_full_scaled), np.nan)

        if len(self.X_full_scaled) >= sequence_length:
            X_seq = []
            valid_indices = []

            for i in range(len(self.X_full_scaled) - sequence_length + 1):
                X_seq.append(self.X_full_scaled[i:i + sequence_length])
                valid_indices.append(i + sequence_length - 1)

            if len(X_seq) > 0:
                X_seq = np.array(X_seq)
                print(f"    LSTM input shape: {X_seq.shape}")
                print(f"    Model expects: {model.input_shape}")

                # Apply physical constraints to predictions
                y_pred_scaled = model.predict(X_seq, verbose=0).ravel()
                y_pred_raw = self.pipeline.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

                # Apply physical constraints (no negative flows)
                y_pred = np.maximum(y_pred_raw, 0)  # Force non-negative

                # Log corrections
                negative_count = np.sum(y_pred_raw < 0)
                if negative_count > 0:
                    print(f"    Corrected {negative_count} negative LSTM predictions")

                predictions[valid_indices] = y_pred

        return predictions

    def _predict_svm(self, model):
        """SVM prediction with proper scaling"""
        y_pred_scaled = model.predict(self.X_full_scaled)
        y_pred = self.pipeline.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return y_pred

    def export_predictions(self, output_dir=None):
        """Export predictions to CSV and create plots"""
        if output_dir is None:
            output_dir = 'extended_predictions_clean'

        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("EXPORTING RESULTS")
        print("=" * 60)

        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': self.dates_full,
            'observed_flow': self.y_observed,
            'has_observed': self.has_observed,
        })

        # Add model predictions
        for model_name, pred in self.predictions.items():
            results_df[f'{model_name}_predicted'] = pred

        # Calculate ensemble predictions
        pred_cols = [col for col in results_df.columns if '_predicted' in col]
        if len(pred_cols) > 1:
            results_df['ensemble_mean'] = results_df[pred_cols].mean(axis=1, skipna=True)
            results_df['ensemble_median'] = results_df[pred_cols].median(axis=1, skipna=True)
        elif len(pred_cols) == 1:
            results_df['ensemble_mean'] = results_df[pred_cols[0]]

        # Add period indicator
        results_df['period'] = results_df['has_observed'].map({True: 'observed', False: 'predicted'})

        # Save main results
        csv_file = os.path.join(output_dir, 'extended_predictions_clean.csv')
        results_df.to_csv(csv_file, index=False)
        print(f"Saved: {csv_file}")

        # Save separate files for observed and predicted periods
        if self.has_observed.sum() > 0:
            obs_df = results_df[results_df['has_observed']].copy()

            # Add performance metrics for observed period
            for col in pred_cols:
                if col in obs_df.columns:
                    obs_df[f'{col}_error'] = obs_df[col] - obs_df['observed_flow']
                    obs_df[f'{col}_abs_error'] = np.abs(obs_df[f'{col}_error'])

            obs_file = os.path.join(output_dir, 'observed_period_validation.csv')
            obs_df.to_csv(obs_file, index=False)
            print(f"Saved: {obs_file}")

        if (~self.has_observed).sum() > 0:
            pred_df = results_df[~results_df['has_observed']].copy()
            pred_file = os.path.join(output_dir, 'future_predictions_clean.csv')
            pred_df.to_csv(pred_file, index=False)
            print(f"Saved: {pred_file}")

        # Create visualization
        self._create_clean_plots(results_df, output_dir)

        # Print comprehensive summary
        self._print_comprehensive_summary(results_df)

        return results_df

    def _print_comprehensive_summary(self, results_df):
        """Print detailed summary of results"""
        print(f"\nCOMPREHENSIVE RESULTS SUMMARY:")
        print(f"=" * 50)

        # Time period summary
        print(f"Total period: {results_df['date'].min()} to {results_df['date'].max()}")
        print(f"Total samples: {len(results_df)}")
        print(f"Observed samples: {self.has_observed.sum()}")
        print(f"Future prediction samples: {(~self.has_observed).sum()}")

        # Performance on observed period
        if self.has_observed.sum() > 0:
            print(f"\nObserved Period Model Performance:")
            obs_data = results_df[results_df['has_observed']]

            for model_name in self.predictions.keys():
                col = f'{model_name}_predicted'
                if col in obs_data.columns:
                    obs_true = obs_data['observed_flow']
                    obs_pred = obs_data[col]

                    valid_mask = ~(np.isnan(obs_true) | np.isnan(obs_pred))
                    if valid_mask.sum() > 5:
                        obs_true_clean = obs_true[valid_mask]
                        obs_pred_clean = obs_pred[valid_mask]

                        # Calculate comprehensive metrics
                        rmse = np.sqrt(np.mean((obs_true_clean - obs_pred_clean) ** 2))
                        mae = np.mean(np.abs(obs_true_clean - obs_pred_clean))

                        # NSE
                        ss_res = np.sum((obs_true_clean - obs_pred_clean) ** 2)
                        ss_tot = np.sum((obs_true_clean - np.mean(obs_true_clean)) ** 2)
                        nse = 1 - (ss_res / ss_tot)

                        # R²
                        r2 = np.corrcoef(obs_true_clean, obs_pred_clean)[0, 1] ** 2

                        print(f"  {model_name}: NSE={nse:.3f}, R²={r2:.3f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

        # Future predictions summary
        if (~self.has_observed).sum() > 0:
            print(f"\nFuture Predictions Summary:")
            future_data = results_df[~results_df['has_observed']]

            if 'ensemble_mean' in future_data.columns:
                future_flows = future_data['ensemble_mean'].dropna()
                if len(future_flows) > 0:
                    print(f"  Ensemble predictions: {len(future_flows)} values")
                    print(f"  Range: {future_flows.min():.2f} to {future_flows.max():.2f} m³/s")
                    print(f"  Mean: {future_flows.mean():.2f} m³/s")
                    print(f"  Median: {future_flows.median():.2f} m³/s")

            # Model agreement analysis
            pred_cols = [col for col in future_data.columns if '_predicted' in col and 'ensemble' not in col]
            if len(pred_cols) > 1:
                future_std = future_data[pred_cols].std(axis=1, skipna=True)
                mean_std = future_std.mean()
                print(f"  Model agreement (avg std): {mean_std:.2f} m³/s")
                if mean_std > future_data['ensemble_mean'].mean() * 0.2:
                    print(f"  ⚠️  High model disagreement - interpret with caution")

    def _create_clean_plots(self, results_df, output_dir):
        """Create clean visualization plots"""
        print("\nCreating clean prediction plots...")

        # Create comprehensive figure
        fig = plt.figure(figsize=(18, 14))
        gs = plt.GridSpec(4, 2, hspace=0.35, wspace=0.25)

        # 1. Full time series overview
        ax1 = plt.subplot(gs[0, :])
        obs_data = results_df[results_df['has_observed']]
        pred_data = results_df[~results_df['has_observed']]

        if len(obs_data) > 0:
            ax1.plot(obs_data['date'], obs_data['observed_flow'], 'b-',
                     label='Observed', linewidth=2, alpha=0.8)

        if len(pred_data) > 0 and 'ensemble_mean' in pred_data.columns:
            ax1.plot(pred_data['date'], pred_data['ensemble_mean'], 'r-',
                     label='Future Predictions', linewidth=2, alpha=0.8)

        # Mark transition
        if self.has_observed.sum() > 0:
            transition_date = obs_data['date'].max()
            ax1.axvline(x=transition_date, color='gray', linestyle='--', alpha=0.7,
                        label='End of Observed Data')

        ax1.set_title('Clean ML Pipeline: Extended Flow Predictions', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Flow (m³/s)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Observed vs Predicted validation scatter
        ax2 = plt.subplot(gs[1, 0])
        if len(obs_data) > 0 and 'ensemble_mean' in obs_data.columns:
            obs_true = obs_data['observed_flow']
            obs_pred = obs_data['ensemble_mean']

            valid_mask = ~(np.isnan(obs_true) | np.isnan(obs_pred))
            if valid_mask.sum() > 0:
                obs_true_clean = obs_true[valid_mask]
                obs_pred_clean = obs_pred[valid_mask]

                ax2.scatter(obs_true_clean, obs_pred_clean, alpha=0.6, s=25)

                # Add 1:1 line
                min_val = min(obs_true_clean.min(), obs_pred_clean.min())
                max_val = max(obs_true_clean.max(), obs_pred_clean.max())
                ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

                # Calculate and show R²
                r2 = np.corrcoef(obs_true_clean, obs_pred_clean)[0, 1] ** 2
                ax2.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax2.transAxes,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)

        ax2.set_title('Model Validation: Observed vs Predicted')
        ax2.set_xlabel('Observed Flow (m³/s)')
        ax2.set_ylabel('Predicted Flow (m³/s)')
        ax2.grid(True, alpha=0.3)

        # 3. Residuals analysis
        ax3 = plt.subplot(gs[1, 1])
        if len(obs_data) > 0 and 'ensemble_mean' in obs_data.columns:
            residuals = obs_data['ensemble_mean'] - obs_data['observed_flow']
            valid_residuals = residuals.dropna()

            if len(valid_residuals) > 0:
                ax3.scatter(obs_data.loc[residuals.notna(), 'observed_flow'],
                            valid_residuals, alpha=0.6, s=25)
                ax3.axhline(y=0, color='r', linestyle='--', alpha=0.8, linewidth=2)

                rmse = np.sqrt(np.mean(valid_residuals ** 2))
                ax3.text(0.05, 0.95, f'RMSE = {rmse:.2f}', transform=ax3.transAxes,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)

        ax3.set_title('Residuals Analysis')
        ax3.set_xlabel('Observed Flow (m³/s)')
        ax3.set_ylabel('Residual (m³/s)')
        ax3.grid(True, alpha=0.3)

        # 4. Future predictions: Individual models
        ax4 = plt.subplot(gs[2, :])
        if len(pred_data) > 0:
            colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
            model_count = 0

            for model_name in self.predictions.keys():
                col = f'{model_name}_predicted'
                if col in pred_data.columns and not pred_data[col].isna().all():
                    ax4.plot(pred_data['date'], pred_data[col],
                             color=colors[model_count % len(colors)],
                             label=model_name, alpha=0.6, linewidth=1.5)
                    model_count += 1

            if 'ensemble_mean' in pred_data.columns:
                ax4.plot(pred_data['date'], pred_data['ensemble_mean'], 'k-',
                         label='Ensemble Mean', linewidth=3, alpha=0.9)

            # Add uncertainty band if multiple models
            pred_model_cols = [col for col in pred_data.columns if '_predicted' in col and 'ensemble' not in col]
            if len(pred_model_cols) > 1:
                lower = pred_data[pred_model_cols].quantile(0.25, axis=1)
                upper = pred_data[pred_model_cols].quantile(0.75, axis=1)
                ax4.fill_between(pred_data['date'], lower, upper,
                                 alpha=0.2, color='gray', label='Model Uncertainty (IQR)')

        ax4.set_title('Future Period: Individual Model Predictions & Uncertainty')
        ax4.set_ylabel('Flow (m³/s)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Monthly climatology comparison
        ax5 = plt.subplot(gs[3, 0])
        if len(results_df) > 0:
            results_df_copy = results_df.copy()
            results_df_copy['month'] = pd.to_datetime(results_df_copy['date']).dt.month

            monthly_obs = results_df_copy[results_df_copy['has_observed']].groupby('month')['observed_flow'].mean()

            months = range(1, 13)
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            if len(monthly_obs) > 0:
                ax5.plot(months, [monthly_obs.get(m, np.nan) for m in months],
                         'b-o', label='Observed', linewidth=2, markersize=6)

            if len(pred_data) > 0 and 'ensemble_mean' in results_df_copy.columns:
                monthly_pred = results_df_copy[~results_df_copy['has_observed']].groupby('month')[
                    'ensemble_mean'].mean()
                if len(monthly_pred) > 0:
                    ax5.plot(months, [monthly_pred.get(m, np.nan) for m in months],
                             'r-o', label='Predicted', linewidth=2, markersize=6)

        ax5.set_title('Monthly Flow Climatology')
        ax5.set_xlabel('Month')
        ax5.set_ylabel('Mean Flow (m³/s)')
        ax5.set_xticks(months)
        ax5.set_xticklabels(month_names, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Model performance comparison
        ax6 = plt.subplot(gs[3, 1])
        if len(obs_data) > 0:
            model_names = []
            nse_values = []

            for model_name in self.predictions.keys():
                col = f'{model_name}_predicted'
                if col in obs_data.columns:
                    obs_true = obs_data['observed_flow']
                    obs_pred = obs_data[col]

                    valid_mask = ~(np.isnan(obs_true) | np.isnan(obs_pred))
                    if valid_mask.sum() > 5:
                        obs_true_clean = obs_true[valid_mask]
                        obs_pred_clean = obs_pred[valid_mask]

                        # Calculate NSE
                        ss_res = np.sum((obs_true_clean - obs_pred_clean) ** 2)
                        ss_tot = np.sum((obs_true_clean - np.mean(obs_true_clean)) ** 2)
                        nse = 1 - (ss_res / ss_tot)

                        model_names.append(model_name)
                        nse_values.append(nse)

            if model_names:
                colors_bar = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
                bars = ax6.bar(model_names, nse_values, color=colors_bar, alpha=0.8)

                # Add value labels on bars
                for bar, nse in zip(bars, nse_values):
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                             f'{nse:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax6.set_title('Model Performance (NSE on Observed Period)')
        ax6.set_ylabel('Nash-Sutcliffe Efficiency')
        ax6.set_ylim(0, 1)
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')

        plt.suptitle(
            'Clean ML Pipeline: Extended Hydrological Predictions Analysis\n(No Flow-Based Features - Suitable for True Future Prediction)',
            fontsize=18, y=0.98)

        # Save plot
        plot_file = os.path.join(output_dir, 'clean_extended_predictions_analysis.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {plot_file}")


def load_best_model_config(results_dir):
    """Load best model configuration from results directory"""
    config_file = os.path.join(results_dir, 'best_model_config.json')

    if not os.path.exists(config_file):
        # Fallback to simple lookup
        simple_file = os.path.join(results_dir, 'model_lookup.json')
        if os.path.exists(simple_file):
            with open(simple_file, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"No model configuration found in {results_dir}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    return config


def run_extended_predictions(pipeline, output_dir=None, results_dir=None):
    """
    Main function to run extended predictions with best model configuration

    Args:
        pipeline: Trained ML pipeline
        output_dir: Output directory for extended predictions
        results_dir: Directory containing best_model_config.json (e.g., 'Bengbu_results')

    Returns:
        DataFrame with results
    """
    print("\n" + "=" * 60)
    print("EXTENDED PREDICTIONS WITH BEST MODEL CONFIG")
    print("=" * 60)

    if results_dir:
        print(f"Using model configuration from: {results_dir}")

    predictor = ExtendedPredictor(pipeline, results_dir)

    if not predictor.prepare_full_period_data():
        print("Failed to prepare data")
        return None

    predictions = predictor.generate_predictions()

    # Focus on best model if configuration is available
    if predictor.config and predictor.best_model_name:
        print(f"\n🏆 Best model focus: {predictor.best_model_name}")
        if predictor.best_model_name in predictions:
            best_pred = predictions[predictor.best_model_name]
            valid_preds = ~np.isnan(best_pred)
            print(f"   Valid predictions: {valid_preds.sum()}")
            if valid_preds.sum() > 0:
                print(f"   Range: {best_pred[valid_preds].min():.2f} to {best_pred[valid_preds].max():.2f} m³/s")

    results_df = predictor.export_predictions(output_dir)

    print("\n" + "=" * 60)
    print("EXTENDED PREDICTIONS COMPLETE!")
    if predictor.config:
        print(f"Used best model configuration: {predictor.best_model_name}")
        if predictor.best_model_name == 'LSTM':
            print(f"LSTM sequence length: {predictor.sequence_length}")
    print("=" * 60)

    return results_df


# Usage
if __name__ == "__main__":
    from enhanced_ml_pipeline import AdvancedHydrologicalMLPipeline

    # Load trained pipeline
    config_path = 'hyperparameter_config.json'
    pipeline = AdvancedHydrologicalMLPipeline(config_path)
    pipeline.load_and_prepare_data()

    # Train models using clean pipeline (no flow features)
    pipeline.optimize_all_models()

    # Run extended predictions
    results = run_extended_predictions(pipeline, 'extended_predictions_clean')
