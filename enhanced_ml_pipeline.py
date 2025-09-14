import os
import json
import pickle
import joblib
from datetime import datetime
import numpy as np
import pandas as pd

# For LSTM models
import tensorflow as tf
import keras
# For other models
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Import your main pipeline
from ML import AdvancedHydrologicalMLPipeline


class ModelSaver:
    """Utility class for saving and loading different types of models"""

    @staticmethod
    def save_model(model, model_name, save_dir):
        """Save model based on its type"""
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, model_name)

        if model_name == 'LSTM':
            # Use joblib for LSTM - much simpler and more reliable!
            joblib.dump(model, f"{model_path}_joblib.pkl")
            print(f"Saved LSTM model using joblib to {model_path}_joblib.pkl")

            # Optional: Also try to save in Keras native format as backup
            try:
                model.save(f"{model_path}_backup.keras")
                print(f"Also saved backup in Keras format")
            except Exception as e:
                print(f"Could not save Keras backup (not critical): {e}")

        elif model_name == 'XGBoost':
            # XGBoost native save
            model.save_model(f"{model_path}.json")
            print(f"Saved XGBoost model to {model_path}.json")

        elif model_name == 'LightGBM':
            # LightGBM can be either a Booster object or sklearn wrapper
            if hasattr(model, 'booster_'):
                # It's a sklearn wrapper (LGBMRegressor)
                model.booster_.save_model(f"{model_path}.txt")
                print(f"Saved LightGBM model (sklearn wrapper) to {model_path}.txt")
            elif isinstance(model, lgb.Booster):
                # It's already a Booster object
                model.save_model(f"{model_path}.txt")
                print(f"Saved LightGBM Booster to {model_path}.txt")
            else:
                # Fallback to joblib
                joblib.dump(model, f"{model_path}.pkl")
                print(f"Saved LightGBM model with joblib to {model_path}.pkl")

        elif model_name in ['RandomForest', 'SVM', 'CatBoost', 'Cubist']:
            # Use joblib for sklearn and similar models
            joblib.dump(model, f"{model_path}.pkl")
            print(f"Saved {model_name} model to {model_path}.pkl")

        else:
            # Fallback to joblib (better than pickle for most cases)
            joblib.dump(model, f"{model_path}.pkl")
            print(f"Saved {model_name} model to {model_path}.pkl")

    @staticmethod
    def load_model(model_name, save_dir):
        """Load model based on its type"""
        model_path = os.path.join(save_dir, model_name)

        if model_name == 'LSTM':
            # Try joblib first (most reliable)
            if os.path.exists(f"{model_path}_joblib.pkl"):
                try:
                    model = joblib.load(f"{model_path}_joblib.pkl")
                    print(f"Loaded LSTM model from joblib")

                    # Recompile if needed (sometimes necessary after loading)
                    if not model.compiled:
                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss='mse',
                            metrics=['mae']
                        )
                        print("Recompiled LSTM model")

                    return model
                except Exception as e:
                    print(f"Error loading LSTM with joblib: {e}")

            # Try Keras format as fallback
            if os.path.exists(f"{model_path}_backup.keras"):
                try:
                    model = tf.keras.models.load_model(f"{model_path}_backup.keras")
                    print(f"Loaded LSTM model from Keras format")
                    return model
                except Exception as e:
                    print(f"Error loading Keras backup: {e}")

            print(f"Could not load LSTM model - consider retraining")
            return None

        elif model_name == 'XGBoost':
            if os.path.exists(f"{model_path}.json"):
                model = xgb.XGBRegressor()
                model.load_model(f"{model_path}.json")
                print(f"Loaded XGBoost model from {model_path}.json")
                return model

        elif model_name == 'LightGBM':
            # Try native format first
            if os.path.exists(f"{model_path}.txt"):
                model = lgb.Booster(model_file=f"{model_path}.txt")
                print(f"Loaded LightGBM Booster from {model_path}.txt")
                return model
            # Try joblib format
            elif os.path.exists(f"{model_path}.pkl"):
                model = joblib.load(f"{model_path}.pkl")
                print(f"Loaded LightGBM model from {model_path}.pkl")
                return model

        elif model_name in ['RandomForest', 'SVM', 'CatBoost', 'Cubist']:
            if os.path.exists(f"{model_path}.pkl"):
                model = joblib.load(f"{model_path}.pkl")
                print(f"Loaded {model_name} model from {model_path}.pkl")
                return model

        else:
            if os.path.exists(f"{model_path}.pkl"):
                model = joblib.load(f"{model_path}.pkl")
                print(f"Loaded {model_name} model from {model_path}.pkl")
                return model

        print(f"Model {model_name} not found in {save_dir}")
        return None


class EnhancedMLPipeline(AdvancedHydrologicalMLPipeline):
    """Enhanced pipeline with save/load functionality"""

    def __init__(self, config_path='hyperparameter_config.json', save_dir='saved_models'):
        super().__init__(config_path)
        self.save_dir = save_dir
        self.model_saver = ModelSaver()

    def save_all_models(self):
        """Save all trained models"""
        print("\n" + "=" * 60)
        print("SAVING ALL TRAINED MODELS")
        print("=" * 60)

        # Create main save directory
        os.makedirs(self.save_dir, exist_ok=True)

        # Save each model
        for model_name, model in self.models.items():
            self.model_saver.save_model(model, model_name, self.save_dir)

        # Save best parameters
        params_file = os.path.join(self.save_dir, 'best_parameters.json')
        # Convert numpy types to Python types for JSON serialization
        params_to_save = {}
        for model_name, params in self.best_params.items():
            params_to_save[model_name] = {
                k: float(v) if isinstance(v, np.number) else v
                for k, v in params.items()
            }

        with open(params_file, 'w') as f:
            json.dump(params_to_save, f, indent=2)
        print(f"Saved best parameters to {params_file}")

        # Save results and metrics
        results_file = os.path.join(self.save_dir, 'results.pkl')
        joblib.dump(self.results, results_file)
        print(f"Saved results to {results_file}")

        # Save scalers (important for prediction!)
        scalers_file = os.path.join(self.save_dir, 'scalers.pkl')
        joblib.dump({
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }, scalers_file)
        print(f"Saved scalers to {scalers_file}")

        # Save feature names
        features_file = os.path.join(self.save_dir, 'feature_names.json')
        with open(features_file, 'w') as f:
            json.dump(self.feature_names, f)
        print(f"Saved feature names to {features_file}")

        # Save training metadata
        metadata_file = os.path.join(self.save_dir, 'training_metadata.json')
        metadata = {
            'training_date': datetime.now().isoformat(),
            'n_features': len(self.feature_names),
            'n_train_samples': len(self.X_train) if hasattr(self, 'X_train') else None,
            'n_test_samples': len(self.X_test) if hasattr(self, 'X_test') else None,
            'models_trained': list(self.models.keys())
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved training metadata to {metadata_file}")

        print("\n✅ All models and metadata saved successfully!")

    def load_all_models(self):
        """Load all saved models"""
        print("\n" + "=" * 60)
        print("LOADING SAVED MODELS")
        print("=" * 60)

        if not os.path.exists(self.save_dir):
            print(f"Save directory {self.save_dir} not found!")
            return False

        # Load best parameters
        params_file = os.path.join(self.save_dir, 'best_parameters.json')
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                self.best_params = json.load(f)
            print(f"Loaded best parameters from {params_file}")

        # Load results
        results_file = os.path.join(self.save_dir, 'results.pkl')
        if os.path.exists(results_file):
            self.results = joblib.load(results_file)
            print(f"Loaded results from {results_file}")

        # Load scalers
        scalers_file = os.path.join(self.save_dir, 'scalers.pkl')
        if os.path.exists(scalers_file):
            scalers = joblib.load(scalers_file)
            self.scaler_X = scalers['scaler_X']
            self.scaler_y = scalers['scaler_y']
            print(f"Loaded scalers from {scalers_file}")

        # Load feature names
        features_file = os.path.join(self.save_dir, 'feature_names.json')
        if os.path.exists(features_file):
            with open(features_file, 'r') as f:
                self.feature_names = json.load(f)
            print(f"Loaded feature names from {features_file}")

        # Load training metadata
        metadata_file = os.path.join(self.save_dir, 'training_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.training_metadata = json.load(f)
            print(f"Loaded training metadata from {metadata_file}")
            print(f"  Models were trained on: {self.training_metadata['training_date']}")
            print(f"  Number of features: {self.training_metadata['n_features']}")

        # Load each model
        model_names = list(self.best_params.keys())
        for model_name in model_names:
            model = self.model_saver.load_model(model_name, self.save_dir)
            if model is not None:
                self.models[model_name] = model

        print(f"\n✅ Loaded {len(self.models)} models successfully!")
        return True

    def predict_with_saved_models(self, X_new):
        """Make predictions with saved models"""
        predictions = {}

        # Scale input
        X_scaled = self.scaler_X.transform(X_new)

        for model_name, model in self.models.items():
            try:
                if model_name == 'LSTM':
                    # LSTM needs 3D input: (samples, timesteps, features)
                    # Assuming single timestep for simplicity
                    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                    pred_scaled = model.predict(X_lstm, verbose=0)
                    predictions[model_name] = self.scaler_y.inverse_transform(
                        pred_scaled.reshape(-1, 1)
                    ).ravel()
                elif model_name == 'SVM':
                    # SVM uses scaled targets
                    pred_scaled = model.predict(X_scaled)
                    predictions[model_name] = self.scaler_y.inverse_transform(
                        pred_scaled.reshape(-1, 1)
                    ).ravel()
                elif model_name == 'LightGBM':
                    # Handle different LightGBM types
                    if hasattr(model, 'predict'):
                        # Check if it's a Booster object
                        if isinstance(model, lgb.Booster):
                            # Booster object uses predict method with optional parameters
                            predictions[model_name] = model.predict(
                                X_scaled,
                                num_iteration=model.best_iteration if hasattr(model, 'best_iteration') else None
                            )
                        else:
                            # sklearn wrapper or other
                            predictions[model_name] = model.predict(X_scaled)
                    else:
                        print(f"⚠️ LightGBM model doesn't have predict method")
                        continue
                else:
                    predictions[model_name] = model.predict(X_scaled)

                print(f"✅ Generated predictions for {model_name}")

            except Exception as e:
                print(f"⚠️ Could not generate predictions for {model_name}: {e}")
                continue

        return predictions


def run_training_with_saves():
    """Main training function with model saving"""

    # Configuration
    config_file = 'hyperparameter_config.json'
    save_dir = 'Bantai_saved_models'
    checkpoint_dir = 'Bantai_training_checkpoints'

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize enhanced pipeline
    pipeline = EnhancedMLPipeline(config_path=config_file, save_dir=save_dir)

    # Check if we have saved models
    if os.path.exists(os.path.join(save_dir, 'best_parameters.json')):
        print("Found existing saved models!")
        response = input("Load existing models? (y/n): ")
        if response.lower() == 'y':
            pipeline.load_all_models()
            pipeline.print_summary()
            return pipeline

    # Load and prepare data
    pipeline.load_and_prepare_data()

    # Define models to train
    models_to_train = [
        ('XGBoost', pipeline.optimize_xgboost),
        ('LightGBM', pipeline.optimize_lightgbm),
        ('RandomForest', pipeline.optimize_random_forest),
        ('Cubist', pipeline.optimize_cubist),
        ('SVM', pipeline.optimize_svm),
        ('LSTM', pipeline.optimize_lstm)
    ]

    # Training with checkpoints
    for model_name, train_func in models_to_train:
        checkpoint_file = os.path.join(checkpoint_dir, f'{model_name}_done.txt')

        # Skip if already done
        if os.path.exists(checkpoint_file):
            print(f"\n⏭️ Skipping {model_name} - already completed")
            # Try to load the model if not already loaded
            if model_name not in pipeline.models:
                model = pipeline.model_saver.load_model(model_name, save_dir)
                if model is not None:
                    pipeline.models[model_name] = model
            continue

        print(f"\n🚀 Training {model_name}...")
        try:
            # Train the model
            train_func()

            # Save immediately after training
            pipeline.model_saver.save_model(
                pipeline.models[model_name],
                model_name,
                save_dir
            )

            # Save parameters
            temp_params_file = os.path.join(checkpoint_dir, f'{model_name}_params.json')
            with open(temp_params_file, 'w') as f:
                params = pipeline.best_params.get(model_name, {})
                params_to_save = {
                    k: float(v) if isinstance(v, np.number) else v
                    for k, v in params.items()
                }
                json.dump(params_to_save, f)

            # Mark as complete
            with open(checkpoint_file, 'w') as f:
                f.write(f"Completed at {datetime.now()}\n")
                f.write(f"Model saved to {save_dir}\n")
                if model_name in pipeline.results:
                    metrics = pipeline.results[model_name]['test_metrics']
                    f.write(f"Test NSE: {metrics['NSE']:.4f}\n")
                    f.write(f"Test R2: {metrics['R2']:.4f}\n")

            print(f"✅ {model_name} training completed and saved!")

        except Exception as e:
            print(f"❌ Failed to train {model_name}: {e}")
            response = input("Continue with next model? (y/n): ")
            if response.lower() != 'y':
                break

    # Save all models and metadata at the end
    pipeline.save_all_models()

    # Print final summary
    pipeline.print_summary()

    # Save summary to file
    summary_file = os.path.join(save_dir, 'training_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Training completed at {datetime.now()}\n")
        f.write("=" * 60 + "\n")
        f.write("Model Performance Summary\n")
        f.write("=" * 60 + "\n")
        for model_name in pipeline.results:
            metrics = pipeline.results[model_name]['test_metrics']
            f.write(f"\n{model_name}:\n")
            f.write(f"  Test NSE: {metrics['NSE']:.4f}\n")
            f.write(f"  Test R2: {metrics['R2']:.4f}\n")
            f.write(f"  Test RMSE: {metrics['RMSE']:.4f}\n")
            f.write(f"  Test KGE: {metrics['KGE']:.4f}\n")

    print(f"\n📊 Training summary saved to {summary_file}")

    return pipeline


def load_and_use_models():
    """Example of loading and using saved models"""

    save_dir = 'Bantai_saved_models'

    # Initialize pipeline
    pipeline = EnhancedMLPipeline(save_dir=save_dir)

    # Load saved models
    if not pipeline.load_all_models():
        print("No saved models found. Please train first!")
        return

    # Load data to get test samples
    pipeline.load_and_prepare_data()

    # Make predictions with first 10 test samples
    X_sample = pipeline.X_test[:10]

    print("\n" + "=" * 60)
    print("MAKING PREDICTIONS WITH SAVED MODELS")
    print("=" * 60)

    predictions = pipeline.predict_with_saved_models(X_sample)

    # Display predictions
    for model_name, preds in predictions.items():
        print(f"\n{model_name} predictions: {preds[:5]}...")  # Show first 5

    # Compare with actual values
    y_actual = pipeline.y_test[:10]
    print(f"\nActual values: {y_actual[:5]}...")

    # Calculate simple error metrics
    print("\n" + "=" * 60)
    print("PREDICTION ERRORS (first 10 samples)")
    print("=" * 60)

    for model_name, preds in predictions.items():
        mae = np.mean(np.abs(preds - y_actual))
        rmse = np.sqrt(np.mean((preds - y_actual) ** 2))
        print(f"\n{model_name}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")


if __name__ == "__main__":
    # Set input variables directly for IDE execution
    mode = "train"  # Options: "train", "load", "both"

    if mode == "train" or mode == "both":
        # Main training with saves
        pipeline = run_training_with_saves()

    if mode == "load" or mode == "both":
        # Load and use saved models
        load_and_use_models()