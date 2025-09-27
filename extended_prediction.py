"""
Extended prediction for complete forcing data - works with clean ML pipeline (no flow features)
Fixed version: loads config from {site}_saved_models/best_parameters.json
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


class ExtendedPredictor:
    """Generate predictions for full period using clean ML pipeline (no flow-based features)"""

    def __init__(self, pipeline, site_name=None, results_dir=None):
        self.pipeline = pipeline
        self.site_name = site_name
        self.results_dir = results_dir

        # Load best model configuration from saved_models directory
        if site_name:
            saved_models_dir = f"{site_name}_saved_models"
            config_path = os.path.join(saved_models_dir, 'best_parameters.json')

            print(f'Looking for config at: {config_path}')

            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)

                    # Determine best model from config (you may need to add logic here)
                    # For now, we'll look for a best_model key or use all models
                    if "best_model" in self.config:
                        self.best_model_name = self.config["best_model"]["name"]
                    else:
                        # If no explicit best model, you might want to determine it
                        # based on validation scores or use ensemble
                        self.best_model_name = None
                        print("No explicit best model in config, will use ensemble")

                    # Get LSTM sequence length if LSTM is present
                    if 'LSTM' in self.config:
                        self.sequence_length = self.config["LSTM"].get("sequence_length", 45)
                        print(f"LSTM sequence length from config: {self.sequence_length}")
                    else:
                        self.sequence_length = 0

                    print(f"Successfully loaded configuration from {config_path}")
                    print(f"Available models in config: {list(self.config.keys())}")

                except Exception as e:
                    print(f"Warning: Could not load model config: {e}")
                    self.config = None
                    self.best_model_name = None
                    self.sequence_length = getattr(pipeline, 'sequence_length', 10)
            else:
                print(f"Config file not found at {config_path}")
                self.config = None
                self.best_model_name = None
                self.sequence_length = getattr(pipeline, 'sequence_length', 10)
        else:
            print("No site name provided, cannot load saved model config")
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
        """Generate predictions for all models - FIXED VERSION"""
        print("\n" + "=" * 60)
        print("GENERATING PREDICTIONS")
        print("=" * 60)

        predictions = {}

        for model_name, model in self.pipeline.models.items():
            print(f"\nPredicting with {model_name}...")

            try:
                if model_name == 'LSTM':
                    pred = self._predict_lstm_fixed(model)
                elif model_name == 'SVM':
                    pred = self._predict_svm(model)
                else:
                    # Tree-based models (XGBoost, LightGBM, RandomForest, etc.)
                    pred = model.predict(self.X_full_scaled)

                predictions[model_name] = pred

                # Check prediction values
                valid_preds = ~np.isnan(pred)
                if valid_preds.sum() > 0:
                    print(f"  Prediction stats - Min: {np.min(pred[valid_preds]):.2f}, "
                          f"Max: {np.max(pred[valid_preds]):.2f}, "
                          f"Mean: {np.mean(pred[valid_preds]):.2f}")
                    print(f"  Valid predictions: {valid_preds.sum()} out of {len(pred)}")

                    # Check for unreasonable values
                    if np.max(pred[valid_preds]) > 20000:
                        print(f"  ⚠️  WARNING: Extremely high predictions detected!")
                    if np.min(pred[valid_preds]) < -1000:
                        print(f"  ⚠️  WARNING: Negative flow predictions detected!")
                else:
                    print(f"  ⚠️  WARNING: No valid predictions from {model_name}!")

                # Validate on observed period
                if self.has_observed.sum() > 0:
                    obs_mask = self.has_observed
                    obs_true = self.y_observed[obs_mask]
                    obs_pred = pred[obs_mask]
                    valid_obs = ~np.isnan(obs_true) & ~np.isnan(obs_pred)

                    if valid_obs.sum() > 5:
                        obs_true_clean = obs_true[valid_obs]
                        obs_pred_clean = obs_pred[valid_obs]

                        # Calculate NSE
                        ss_res = np.sum((obs_true_clean - obs_pred_clean) ** 2)
                        ss_tot = np.sum((obs_true_clean - np.mean(obs_true_clean)) ** 2)
                        nse = 1 - (ss_res / ss_tot)
                        rmse = np.sqrt(np.mean((obs_true_clean - obs_pred_clean) ** 2))

                        print(f"  Observed period: NSE={nse:.3f}, RMSE={rmse:.2f}, "
                              f"Valid points={valid_obs.sum()}")
                    else:
                        print(f"  Observed period: Not enough valid points for evaluation")

            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")
                import traceback
                traceback.print_exc()
                predictions[model_name] = np.full(len(self.X_full_scaled), np.nan)

        self.predictions = predictions
        return predictions

    def _predict_lstm_fixed(self, model):
        """
        LSTM prediction with proper sequence handling and alignment
        This version correctly handles the sequence generation and prediction placement
        """

        # Get sequence length from config if available
        if self.config and 'LSTM' in self.config:
            sequence_length = self.config['LSTM'].get('sequence_length', 45)
            print(f"    Using config sequence length: {sequence_length}")
        elif hasattr(self.pipeline, 'sequence_length'):
            sequence_length = self.pipeline.sequence_length
            print(f"    Using pipeline sequence length: {sequence_length}")
        else:
            sequence_length = 30  # Your data shows 30
            print(f"    Using default sequence length: {sequence_length}")

        n_samples = len(self.X_full_scaled)
        n_features = self.X_full_scaled.shape[1]

        print(f"    Data shape: {n_samples} samples, {n_features} features")

        # Initialize predictions with NaN
        predictions = np.full(n_samples, np.nan)

        # Check if we have enough data
        if n_samples < sequence_length:
            print(f"    ERROR: Not enough data ({n_samples} < {sequence_length})")
            return predictions

        # Create sequences using the same approach as training
        X_sequences = []
        sequence_indices = []

        for i in range(sequence_length, n_samples + 1):
            # Extract sequence
            seq = self.X_full_scaled[i - sequence_length:i]
            X_sequences.append(seq)
            # The prediction will be for index i-1 (last element of the sequence)
            sequence_indices.append(i - 1)

        X_sequences = np.array(X_sequences)
        print(f"    Created {len(X_sequences)} sequences of shape {X_sequences.shape}")

        # Make predictions
        try:
            # Ensure correct shape (samples, timesteps, features)
            if len(X_sequences.shape) == 2:
                X_sequences = X_sequences.reshape(len(X_sequences), sequence_length, n_features)

            print(f"    LSTM input shape: {X_sequences.shape}")
            print(f"    Model expects: {model.input_shape}")

            # Get predictions
            y_pred_scaled = model.predict(X_sequences, verbose=0)

            # Handle different output shapes
            if len(y_pred_scaled.shape) > 1:
                y_pred_scaled = y_pred_scaled.ravel()

            print(f"    Raw predictions shape: {y_pred_scaled.shape}")

            # Inverse transform
            y_pred_raw = self.pipeline.scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()

            # Apply physical constraints
            y_pred = np.maximum(y_pred_raw, 0)

            # Count corrections
            negative_count = np.sum(y_pred_raw < 0)
            if negative_count > 0:
                print(f"    Corrected {negative_count} negative predictions")

            # Place predictions at correct indices
            for idx, pred_value in zip(sequence_indices, y_pred):
                predictions[idx] = pred_value

            # Summary
            valid_count = np.sum(~np.isnan(predictions))
            print(f"    Successfully placed {valid_count} predictions")
            print(f"    First {sequence_length} positions have no prediction (need history)")

            # Check prediction distribution
            if valid_count > 0:
                valid_preds = predictions[~np.isnan(predictions)]
                print(f"    Prediction range: {np.min(valid_preds):.2f} to {np.max(valid_preds):.2f}")

        except Exception as e:
            print(f"    ERROR in LSTM prediction: {str(e)}")
            import traceback
            traceback.print_exc()

        return predictions

    def _predict_svm(self, model):
        """SVM prediction with proper scaling"""
        y_pred_scaled = model.predict(self.X_full_scaled)
        y_pred = self.pipeline.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return y_pred

    def export_predictions(self, output_dir=None):
        """Export predictions to CSV with proper LSTM handling"""
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
            col_name = f'{model_name}_predicted'
            results_df[col_name] = pred

            # Check how many valid predictions for this model
            valid_count = results_df[col_name].notna().sum()
            print(f"{model_name}: {valid_count} valid predictions out of {len(results_df)}")

        # Calculate ensemble predictions (excluding NaN)
        pred_cols = [col for col in results_df.columns if '_predicted' in col]

        # For ensemble, only use models with valid predictions at each timestep
        results_df['ensemble_mean'] = results_df[pred_cols].mean(axis=1, skipna=True)
        results_df['ensemble_median'] = results_df[pred_cols].median(axis=1, skipna=True)
        results_df['ensemble_count'] = results_df[pred_cols].notna().sum(axis=1)

        print(f"\nEnsemble statistics:")
        print(f"  Points with all {len(pred_cols)} models: {(results_df['ensemble_count'] == len(pred_cols)).sum()}")
        print(f"  Points with at least 1 model: {(results_df['ensemble_count'] > 0).sum()}")

        # Add period indicator
        results_df['period'] = results_df['has_observed'].map({True: 'observed', False: 'predicted'})

        # Save main results
        csv_file = os.path.join(output_dir, 'extended_predictions_clean.csv')
        results_df.to_csv(csv_file, index=False)
        print(f"\nSaved: {csv_file}")

        # Save validation metrics
        if self.has_observed.sum() > 0:
            obs_df = results_df[results_df['has_observed']].copy()

            # Calculate errors for each model
            for col in pred_cols:
                if col in obs_df.columns:
                    # Only calculate error where both observed and predicted are valid
                    valid_mask = obs_df[col].notna() & obs_df['observed_flow'].notna()
                    obs_df.loc[valid_mask, f'{col}_error'] = (
                            obs_df.loc[valid_mask, col] - obs_df.loc[valid_mask, 'observed_flow']
                    )
                    obs_df.loc[valid_mask, f'{col}_abs_error'] = np.abs(
                        obs_df.loc[valid_mask, f'{col}_error']
                    )

            obs_file = os.path.join(output_dir, 'observed_period_validation.csv')
            obs_df.to_csv(obs_file, index=False)
            print(f"Saved: {obs_file}")

        # Save future predictions
        if (~self.has_observed).sum() > 0:
            pred_df = results_df[~results_df['has_observed']].copy()
            pred_file = os.path.join(output_dir, 'future_predictions_clean.csv')
            pred_df.to_csv(pred_file, index=False)
            print(f"Saved: {pred_file}")

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


def run_extended_predictions(pipeline, site_name, output_dir=None):
    """
    Main function to run extended predictions with best model configuration

    Args:
        pipeline: Trained ML pipeline
        site_name: Site name (e.g., 'Bengbu') to load config from {site}_saved_models/
        output_dir: Output directory for extended predictions

    Returns:
        DataFrame with results
    """
    print("\n" + "=" * 60)
    print("EXTENDED PREDICTIONS WITH BEST MODEL CONFIG")
    print("=" * 60)

    if site_name:
        print(f"Loading model configuration from: {site_name}_saved_models/")

    predictor = ExtendedPredictor(pipeline, site_name=site_name)

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
        if predictor.best_model_name:
            print(f"Used best model configuration: {predictor.best_model_name}")
        else:
            print(f"Using ensemble of all models from config")
        if 'LSTM' in predictor.config:
            print(f"LSTM sequence length: {predictor.config['LSTM'].get('sequence_length', 'not specified')}")
    print("=" * 60)

    return results_df


# Usage example
if __name__ == "__main__":
    from enhanced_ml_pipeline import AdvancedHydrologicalMLPipeline

    # Configuration - set these values directly for IDE execution
    SITE_NAME = "Bengbu"  # Change this to your site name
    CONFIG_PATH = 'hyperparameter_config.json'
    OUTPUT_DIR = f'{SITE_NAME}_extended_predictions_clean'

    # Load trained pipeline
    pipeline = AdvancedHydrologicalMLPipeline(CONFIG_PATH)
    pipeline.load_and_prepare_data()

    # Train models using clean pipeline (no flow features)
    print("Training models...")
    pipeline.optimize_all_models()

    # Run extended predictions with site-specific config
    print(f"\nRunning extended predictions for {SITE_NAME}...")
    results = run_extended_predictions(
        pipeline=pipeline,
        site_name=SITE_NAME,
        output_dir=OUTPUT_DIR
    )

    if results is not None:
        print(f"\nResults saved to {OUTPUT_DIR}/")
        print(f"Total predictions generated: {len(results)}")
    else:
        print("\nPrediction generation failed!")