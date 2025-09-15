"""
Complete ML Training Pipeline with:
- Checkpoint support (resume interrupted training)
- Proper model saving/loading
- Comprehensive results export
- Error handling and recovery

This replaces and enhances your original checkpoint script.
"""

import os
import pickle
from datetime import datetime
from enhanced_ml_pipeline import EnhancedMLPipeline
from results_exporter import ResultsExporter
import yaml


class EnhancedTrainingPipeline:
    """
    Enhanced training pipeline with YAML configuration
    """

    def __init__(self, config_file='hyperparameter_config.json',
                 model_config_file='model_config.yaml',
                 site_name=None):
        self.config_file = config_file
        self.model_config_file = model_config_file

        # Load model configuration
        self.load_model_config()

        # Use provided site_name or get from config or use default
        if site_name:
            self.site_name = site_name
            # Optionally update the YAML config with this site name
            self.model_config['site_name'] = site_name
            self._save_model_config()
        else:
            self.site_name = self.model_config.get('site_name', 'default_site')

        # Set up directories using the site name
        self.checkpoint_dir = f'{self.site_name}_model_checkpoints'
        self.save_dir = f'{self.site_name}_saved_models'
        self.results_dir = f'{self.site_name}_results'

        # Create directories
        for dir_path in [self.checkpoint_dir, self.save_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)

        print(f"Using site name: {self.site_name}")
        # Initialize pipeline
        self.pipeline = EnhancedMLPipeline(
            config_path=self.config_file,
            save_dir=self.save_dir
        )

        # Integrate HydroDL models if available
        try:
            from hydrodl_models import integrate_hydrodl_models
            self.pipeline = integrate_hydrodl_models(self.pipeline)
            print("✓ HydroDL models integrated")
        except ImportError as e:
            print(f"⚠️ HydroDL models not available: {e}")

        # Build model list from YAML
        self.build_model_list()

    def load_model_config(self):
        """Load model configuration from YAML"""
        if os.path.exists(self.model_config_file):
            with open(self.model_config_file, 'r') as f:
                self.model_config = yaml.safe_load(f)
            print(f"Loaded model configuration from {self.model_config_file}")
        else:
            # Create default configuration
            self.create_default_model_config()

    def create_default_model_config(self):
        """Create default model configuration YAML"""
        default_config = {
            'site_name': 'Bengbu',
            'traditional_models': {
                'XGBoost': {'enabled': True, 'priority': 1},
                'LightGBM': {'enabled': True, 'priority': 2},
                'RandomForest': {'enabled': True, 'priority': 3},
                'Cubist': {'enabled': True, 'priority': 4},
                'SVM': {'enabled': True, 'priority': 5}
            },
            'deep_learning_models': {
                'LSTM': {'enabled': True, 'priority': 6}
            },
            'training_settings': {
                'parallel_training': False,
                'max_parallel_jobs': 2,
                'save_intermediate': True,
                'verbose': True
            }
        }

        with open(self.model_config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

        self.model_config = default_config
        print(f"Created default model configuration: {self.model_config_file}")
        print(f"Site name set to: {default_config['site_name']}")

    def _save_model_config(self):
        """Save current model configuration to YAML"""
        with open(self.model_config_file, 'w') as f:
            yaml.dump(self.model_config, f, default_flow_style=False, sort_keys=False)

    def build_model_list(self):
        """Build model training list from YAML configuration"""
        self.models_to_train = []

        # Map model names to methods
        model_methods = {
            # Traditional models
            'XGBoost': self.pipeline.optimize_xgboost,
            'LightGBM': self.pipeline.optimize_lightgbm,
            'RandomForest': self.pipeline.optimize_random_forest,
            'Cubist': self.pipeline.optimize_cubist,
            'SVM': self.pipeline.optimize_svm,

            # Deep learning models
            'LSTM': self.pipeline.optimize_lstm,

            # HydroDL models (check if methods exist)
            'HydroDL_LSTM': getattr(self.pipeline, 'optimize_hydrodl_lstm', None),
            'HydroDL_CNN_LSTM': getattr(self.pipeline, 'optimize_hydrodl_cnn_lstm', None),
            'HydroDL_MTS_LSTM': getattr(self.pipeline, 'optimize_hydrodl_mts_lstm', None),
        }

        # Collect all enabled models
        all_models = []

        for category in ['traditional_models', 'deep_learning_models', 'hydrodl_models']:
            if category in self.model_config:
                for model_name, config in self.model_config[category].items():
                    if config.get('enabled', False):
                        if model_name in model_methods and model_methods[model_name]:
                            all_models.append({
                                'name': model_name,
                                'method': model_methods[model_name],
                                'priority': config.get('priority', 999)
                            })
                        else:
                            print(f"⚠️ Model {model_name} is enabled but method not found")

        # Sort by priority
        all_models.sort(key=lambda x: x['priority'])

        # Build final list
        self.models_to_train = [(m['name'], m['method']) for m in all_models]

        print(f"\n📋 Models to train ({len(self.models_to_train)}):")
        for model_name, _ in self.models_to_train:
            print(f"   - {model_name}")

    def check_existing_models(self):
        """Check if we have previously saved models"""
        params_file = os.path.join(self.save_dir, 'best_parameters.json')
        if os.path.exists(params_file):
            print("\n" + "=" * 60)
            print("EXISTING MODELS FOUND")
            print("=" * 60)

            response = input("\nFound existing saved models. Options:\n"
                             "1. Load existing models and skip training (l)\n"
                             "2. Continue training remaining models (c)\n"
                             "3. Start fresh training (f)\n"
                             "Choice [l/c/f]: ").lower()

            if response == 'l':
                print("\nLoading existing models...")
                # CRITICAL FIX: Load data first, then models
                print("Loading and preparing data...")
                self.pipeline.load_and_prepare_data()

                # Now load the models
                self.pipeline.load_all_models()
                return 'loaded'
            elif response == 'c':
                print("\nContinuing with remaining models...")
                # Load data first
                print("Loading and preparing data...")
                self.pipeline.load_and_prepare_data()

                # Try to load existing models
                try:
                    self.pipeline.load_all_models()
                except:
                    print("Warning: Could not load all existing models")
                return 'continue'
            else:
                print("\nStarting fresh training...")
                # Clear checkpoints
                self._clear_checkpoints()
                return 'fresh'
        return 'fresh'

    def _clear_checkpoints(self):
        """Clear all checkpoint files"""
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('_done.txt'):
                os.remove(os.path.join(self.checkpoint_dir, file))
        print("Cleared all checkpoints")

    def train_with_checkpoints(self):
        """Main training loop with checkpoint support"""
        print("\n" + "=" * 60)
        print("STARTING ML PIPELINE TRAINING")
        print("=" * 60)

        # Load and prepare data (only if not already loaded)
        if not hasattr(self.pipeline, 'X_train'):
            print("\nLoading and preparing data...")
            self.pipeline.load_and_prepare_data()

        # Track completed models
        completed_models = []
        failed_models = []

        # Training loop
        for model_name, train_func in self.models_to_train:
            checkpoint_file = os.path.join(self.checkpoint_dir, f'{model_name}_done.txt')

            # Check if already completed
            if os.path.exists(checkpoint_file):
                print(f"\n⏭️  Skipping {model_name} - already completed")
                completed_models.append(model_name)

                # Try to load the model if not in memory
                if model_name not in self.pipeline.models:
                    try:
                        model = self.pipeline.model_saver.load_model(model_name, self.save_dir)
                        if model is not None:
                            self.pipeline.models[model_name] = model
                    except:
                        print(f"   Warning: Could not load saved {model_name} model")
                continue

            # Train the model
            print(f"\n" + "=" * 40)
            print(f"🚀 Training {model_name}...")
            print("=" * 40)

            try:
                start_time = datetime.now()

                # Execute training
                train_func()

                end_time = datetime.now()
                training_time = (end_time - start_time).total_seconds()

                # Save model immediately
                self.pipeline.model_saver.save_model(
                    self.pipeline.models[model_name],
                    model_name,
                    self.save_dir
                )

                # Save checkpoint with detailed info
                with open(checkpoint_file, 'w') as f:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Completed at: {end_time}\n")
                    f.write(f"Training time: {training_time:.2f} seconds\n")
                    f.write(f"Model saved to: {self.save_dir}\n")

                    if model_name in self.pipeline.results:
                        metrics = self.pipeline.results[model_name]['test_metrics']
                        f.write(f"\nPerformance Metrics:\n")
                        f.write(f"  Test NSE: {metrics['NSE']:.4f}\n")
                        f.write(f"  Test R2: {metrics['R2']:.4f}\n")
                        f.write(f"  Test RMSE: {metrics['RMSE']:.4f}\n")
                        f.write(f"  Test KGE: {metrics['KGE']:.4f}\n")

                # Save intermediate pipeline state
                self._save_pipeline_state()

                completed_models.append(model_name)
                print(f"✅ {model_name} completed successfully!")

            except Exception as e:
                print(f"❌ Failed to train {model_name}: {str(e)}")
                failed_models.append((model_name, str(e)))

                # Ask user whether to continue
                if len(self.models_to_train) > len(completed_models) + len(failed_models):
                    response = input("\nContinue with next model? (y/n): ").lower()
                    if response != 'y':
                        print("Training interrupted by user")
                        break

        # Final save of all models and metadata
        if completed_models:
            print("\n" + "=" * 60)
            print("SAVING ALL MODELS AND METADATA")
            print("=" * 60)
            self.pipeline.save_all_models()

        # Print training summary
        self._print_training_summary(completed_models, failed_models)

        return completed_models, failed_models

    def _save_pipeline_state(self):
        """Save intermediate pipeline state"""
        state_file = os.path.join(self.checkpoint_dir, 'pipeline_state.pkl')

        # Prepare state data
        state = {
            'timestamp': datetime.now(),
            'models_trained': list(self.pipeline.models.keys()),
            'best_params': self.pipeline.best_params,
            'results': self.pipeline.results
        }

        # Save state
        with open(state_file, 'wb') as f:
            pickle.dump(state, f)

    def _print_training_summary(self, completed_models, failed_models):
        """Print and save training summary"""
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        print(f"\nCompleted Models ({len(completed_models)}):")
        for model in completed_models:
            if model in self.pipeline.results:
                metrics = self.pipeline.results[model]['test_metrics']
                print(f"  ✅ {model}: NSE={metrics['NSE']:.4f}, R²={metrics['R2']:.4f}")
            else:
                print(f"  ✅ {model}")

        if failed_models:
            print(f"\nFailed Models ({len(failed_models)}):")
            for model, error in failed_models:
                print(f"  ❌ {model}: {error[:50]}...")

        # Save summary to file
        summary_file = os.path.join(self.checkpoint_dir, 'training_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Training Summary - {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Completed Models ({len(completed_models)}):\n")
            for model in completed_models:
                if model in self.pipeline.results:
                    metrics = self.pipeline.results[model]['test_metrics']
                    f.write(f"  {model}:\n")
                    for metric_name, value in metrics.items():
                        f.write(f"    {metric_name}: {value:.4f}\n")
                else:
                    f.write(f"  {model}: No results available\n")

            if failed_models:
                f.write(f"\nFailed Models ({len(failed_models)}):\n")
                for model, error in failed_models:
                    f.write(f"  {model}: {error}\n")

        print(f"\nSummary saved to: {summary_file}")

    def export_results(self):
        """Export all results using ResultsExporter"""
        print("\n" + "=" * 60)
        print("EXPORTING RESULTS")
        print("=" * 60)

        if not self.pipeline.results:
            print("No results to export. Please train models first.")
            return

        # Ensure data is loaded before exporting
        if not hasattr(self.pipeline, 'dates_train'):
            print("Loading data for export...")
            self.pipeline.load_and_prepare_data()

        exporter = ResultsExporter(self.pipeline, self.results_dir)
        exporter.export_all_results()

        print(f"\n✅ Results exported to: {self.results_dir}/")

    def run_complete_pipeline(self):
        """Run the complete pipeline with all features"""
        print("\n" + "=" * 80)
        print("ENHANCED ML PIPELINE WITH YAML CONFIGURATION")
        print("=" * 80)
        print(f"Configuration: {self.model_config_file}")
        print(f"Site: {self.site_name}")

        # Check for existing models
        status = self.check_existing_models()

        if status == 'loaded':
            # Models loaded, just show summary and export
            self.pipeline.print_summary()

            response = input("\nExport results? (y/n): ").lower()
            if response == 'y':
                self.export_results()
                # Ask about extended predictions
            response = input("\nExtend prediction? (y/n): ").lower()
            if response == 'y':
                from extended_prediction import run_extended_predictions

                # Use the results directory that contains the model config
                extended_results = run_extended_predictions(
                    self.pipeline,
                    output_dir=f'{self.site_name}_extended_predictions',
                    results_dir=self.results_dir  # This contains best_model_config.json
                )
                print(f"Extended predictions saved to {self.site_name}_extended_predictions/")
        else:
            # Train models (fresh or continuing)
            completed, failed = self.train_with_checkpoints()

            if completed:
                # Show performance summary
                self.pipeline.print_summary()

                # Export results
                response = input("\nExport comprehensive results? (y/n): ").lower()
                if response == 'y':
                    self.export_results()
            else:
                print("\nNo models were successfully trained.")

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE!")
        print("=" * 80)

        # Print final status
        print(f"\n📁 Models saved in: {self.save_dir}")
        print(f"📊 Results exported to: {self.results_dir}")
        print(f"📝 Checkpoints in: {self.checkpoint_dir}")

        return self.pipeline


def main():
    """Main entry point - replaces your original script"""

    # Check for configuration file
    config_file = 'hyperparameter_config.json'
    if not os.path.exists(config_file):
        print(f"❌ Configuration file {config_file} not found!")
        print("Please ensure hyperparameter_config.json exists with proper configuration!")
        return

    # Create and run complete pipeline
    training_pipeline = EnhancedTrainingPipeline(config_file)

    # Interactive model selection
    print("\nWould you like to modify model selection? (y/n): ", end='')
    if input().lower() == 'y':
        print("\nAvailable models:")
        all_models = []
        for category in ['traditional_models', 'deep_learning_models']:
            if category in training_pipeline.model_config:
                print(f"\n{category.replace('_', ' ').title()}:")
                for model_name, config in training_pipeline.model_config[category].items():
                    status = "✓" if config.get('enabled', False) else "✗"
                    print(f"  [{status}] {model_name}")
                    all_models.append(model_name)

        print("\nEnter model names to toggle (comma-separated) or press Enter to continue:")
        toggle_input = input().strip()

        if toggle_input:
            models_to_toggle = [m.strip() for m in toggle_input.split(',')]
            for model in models_to_toggle:
                if model in all_models:
                    # Toggle current status
                    for category in ['traditional_models', 'deep_learning_models']:
                        if category in training_pipeline.model_config:
                            if model in training_pipeline.model_config[category]:
                                current = training_pipeline.model_config[category][model].get('enabled', False)
                                training_pipeline.model_config[category][model]['enabled'] = not current
                                training_pipeline._save_model_config()
                                training_pipeline.build_model_list()
                                print(f"  {model}: {'enabled' if not current else 'disabled'}")
                                break

    pipeline = training_pipeline.run_complete_pipeline()

    # Optional: Return pipeline for further use
    return pipeline


if __name__ == "__main__":
    # This replaces your original run_with_checkpoints()
    pipeline = main()

# run_with_checkpoints_enhanced.py
# """
# Enhanced ML Training Pipeline with:
# - YAML-based model selection
# - HydroDL integration
# - Dynamic model loading
# - Checkpoint support
# """

# import os
# import yaml
# import pickle
# from datetime import datetime
# from enhanced_ml_pipeline import EnhancedMLPipeline
# from results_exporter import ResultsExporter
# from HydroDL import integrate_hydrodl_models
#
#
# class EnhancedTrainingPipeline:
#     """
#     Enhanced training pipeline with YAML configuration
#     """
#
#     def __init__(self, config_file='hyperparameter_config.json',
#                  model_config_file='model_config.yaml'):
#         self.config_file = config_file
#         self.model_config_file = model_config_file
#
#         # Load model configuration
#         self.load_model_config()
#
#         # Set up directories
#         site_name = self.model_config.get('site_name', 'default_site')
#         self.checkpoint_dir = f'{site_name}_model_checkpoints'
#         self.save_dir = f'{site_name}_saved_models'
#         self.results_dir = f'{site_name}_results'
#
#         # Create directories
#         for dir_path in [self.checkpoint_dir, self.save_dir, self.results_dir]:
#             os.makedirs(dir_path, exist_ok=True)
#
#         # Initialize pipeline
#         self.pipeline = EnhancedMLPipeline(
#             config_path=self.config_file,
#             save_dir=self.save_dir
#         )
#
#         # Integrate HydroDL models
#         self.pipeline = integrate_hydrodl_models(self.pipeline)
#
#         # Build model list from YAML
#         self.build_model_list()
#
#     def load_model_config(self):
#         """Load model configuration from YAML"""
#         if os.path.exists(self.model_config_file):
#             with open(self.model_config_file, 'r') as f:
#                 self.model_config = yaml.safe_load(f)
#             print(f"Loaded model configuration from {self.model_config_file}")
#         else:
#             # Create default configuration
#             self.create_default_model_config()
#
#     def create_default_model_config(self):
#         """Create default model configuration YAML"""
#         default_config = {
#             'site_name': 'Bengbu',
#             'traditional_models': {
#                 'XGBoost': {'enabled': True, 'priority': 1},
#                 'LightGBM': {'enabled': True, 'priority': 2},
#                 'RandomForest': {'enabled': True, 'priority': 3},
#                 'Cubist': {'enabled': True, 'priority': 4},
#                 'SVM': {'enabled': True, 'priority': 5}
#             },
#             'deep_learning_models': {
#                 'LSTM': {'enabled': True, 'priority': 6}
#             },
#             'hydrodl_models': {
#                 'HydroDL_LSTM': {'enabled': False, 'priority': 7},
#                 'HydroDL_CNN_LSTM': {'enabled': False, 'priority': 8},
#                 'HydroDL_MTS_LSTM': {'enabled': False, 'priority': 9}
#             },
#             'training_settings': {
#                 'parallel_training': False,
#                 'max_parallel_jobs': 2,
#                 'save_intermediate': True,
#                 'verbose': True
#             }
#         }
#
#         with open(self.model_config_file, 'w') as f:
#             yaml.dump(default_config, f, default_flow_style=False)
#
#         self.model_config = default_config
#         print(f"Created default model configuration: {self.model_config_file}")
#
#     def build_model_list(self):
#         """Build model training list from YAML configuration"""
#         self.models_to_train = []
#
#         # Map model names to methods
#         model_methods = {
#             # Traditional models
#             'XGBoost': self.pipeline.optimize_xgboost,
#             'LightGBM': self.pipeline.optimize_lightgbm,
#             'RandomForest': self.pipeline.optimize_random_forest,
#             'Cubist': self.pipeline.optimize_cubist,
#             'SVM': self.pipeline.optimize_svm,
#
#             # Deep learning models
#             'LSTM': self.pipeline.optimize_lstm,
#
#             # HydroDL models
#             'HydroDL_LSTM': self.pipeline.optimize_hydrodl_lstm if hasattr(self.pipeline,
#                                                                            'optimize_hydrodl_lstm') else None,
#             'HydroDL_CNN_LSTM': self.pipeline.optimize_hydrodl_cnn_lstm if hasattr(self.pipeline,
#                                                                                    'optimize_hydrodl_cnn_lstm') else None,
#             'HydroDL_MTS_LSTM': self.pipeline.optimize_hydrodl_mts_lstm if hasattr(self.pipeline,
#                                                                                    'optimize_hydrodl_mts_lstm') else None,
#         }
#
#         # Collect all enabled models
#         all_models = []
#
#         for category in ['traditional_models', 'deep_learning_models', 'hydrodl_models']:
#             if category in self.model_config:
#                 for model_name, config in self.model_config[category].items():
#                     if config.get('enabled', False):
#                         if model_name in model_methods and model_methods[model_name]:
#                             all_models.append({
#                                 'name': model_name,
#                                 'method': model_methods[model_name],
#                                 'priority': config.get('priority', 999)
#                             })
#                         else:
#                             print(f"⚠️ Model {model_name} is enabled but method not found")
#
#         # Sort by priority
#         all_models.sort(key=lambda x: x['priority'])
#
#         # Build final list
#         self.models_to_train = [(m['name'], m['method']) for m in all_models]
#
#         print(f"\n📋 Models to train ({len(self.models_to_train)}):")
#         for model_name, _ in self.models_to_train:
#             print(f"   - {model_name}")
#
#     def update_model_status(self, model_name, enabled=True):
#         """Update model training status in YAML"""
#         # Find and update model in config
#         for category in ['traditional_models', 'deep_learning_models', 'hydrodl_models']:
#             if category in self.model_config:
#                 if model_name in self.model_config[category]:
#                     self.model_config[category][model_name]['enabled'] = enabled
#                     break
#
#         # Save updated config
#         with open(self.model_config_file, 'w') as f:
#             yaml.dump(self.model_config, f, default_flow_style=False)
#
#         # Rebuild model list
#         self.build_model_list()
#
#         print(f"Updated {model_name} status to: {'enabled' if enabled else 'disabled'}")
#
#     def train_with_checkpoints(self):
#         """Main training loop with checkpoint support"""
#         print("\n" + "=" * 60)
#         print("STARTING ML PIPELINE TRAINING")
#         print("=" * 60)
#
#         # Load and prepare data
#         print("\nLoading and preparing data...")
#         self.pipeline.load_and_prepare_data()
#
#         # Track completed models
#         completed_models = []
#         failed_models = []
#
#         # Training settings
#         settings = self.model_config.get('training_settings', {})
#         verbose = settings.get('verbose', True)
#         save_intermediate = settings.get('save_intermediate', True)
#
#         # Training loop
#         for model_name, train_func in self.models_to_train:
#             checkpoint_file = os.path.join(self.checkpoint_dir, f'{model_name}_done.txt')
#
#             # Check if already completed
#             if os.path.exists(checkpoint_file):
#                 print(f"\n⏭️  Skipping {model_name} - already completed")
#                 completed_models.append(model_name)
#
#                 # Try to load the model if not in memory
#                 if model_name not in self.pipeline.models:
#                     try:
#                         model = self.pipeline.model_saver.load_model(model_name, self.save_dir)
#                         if model is not None:
#                             self.pipeline.models[model_name] = model
#                     except:
#                         print(f"   Warning: Could not load saved {model_name} model")
#                 continue
#
#             # Train the model
#             print(f"\n" + "=" * 40)
#             print(f"🚀 Training {model_name}...")
#             print("=" * 40)
#
#             try:
#                 start_time = datetime.now()
#
#                 # Execute training
#                 train_func()
#
#                 end_time = datetime.now()
#                 training_time = (end_time - start_time).total_seconds()
#
#                 # Save model immediately if configured
#                 if save_intermediate:
#                     self.pipeline.model_saver.save_model(
#                         self.pipeline.models[model_name],
#                         model_name,
#                         self.save_dir
#                     )
#
#                 # Save checkpoint with detailed info
#                 with open(checkpoint_file, 'w') as f:
#                     f.write(f"Model: {model_name}\n")
#                     f.write(f"Completed at: {end_time}\n")
#                     f.write(f"Training time: {training_time:.2f} seconds\n")
#                     f.write(f"Model saved to: {self.save_dir}\n")
#
#                     if model_name in self.pipeline.results:
#                         metrics = self.pipeline.results[model_name]['test_metrics']
#                         f.write(f"\nPerformance Metrics:\n")
#                         f.write(f"  Test NSE: {metrics['NSE']:.4f}\n")
#                         f.write(f"  Test R2: {metrics['R2']:.4f}\n")
#                         f.write(f"  Test RMSE: {metrics['RMSE']:.4f}\n")
#                         f.write(f"  Test KGE: {metrics['KGE']:.4f}\n")
#
#                 # Save intermediate pipeline state
#                 self._save_pipeline_state()
#
#                 completed_models.append(model_name)
#                 print(f"✅ {model_name} completed successfully!")
#
#             except Exception as e:
#                 print(f"❌ Failed to train {model_name}: {str(e)}")
#                 failed_models.append((model_name, str(e)))
#
#                 # Ask user whether to continue
#                 if len(self.models_to_train) > len(completed_models) + len(failed_models):
#                     response = input("\nContinue with next model? (y/n): ").lower()
#                     if response != 'y':
#                         print("Training interrupted by user")
#                         break
#
#         # Final save of all models and metadata
#         if completed_models:
#             print("\n" + "=" * 60)
#             print("SAVING ALL MODELS AND METADATA")
#             print("=" * 60)
#             self.pipeline.save_all_models()
#
#         # Print training summary
#         self._print_training_summary(completed_models, failed_models)
#
#         return completed_models, failed_models
#
#     def _save_pipeline_state(self):
#         """Save intermediate pipeline state"""
#         state_file = os.path.join(self.checkpoint_dir, 'pipeline_state.pkl')
#
#         state = {
#             'timestamp': datetime.now(),
#             'models_trained': list(self.pipeline.models.keys()),
#             'best_params': self.pipeline.best_params,
#             'results': self.pipeline.results,
#             'model_config': self.model_config
#         }
#
#         with open(state_file, 'wb') as f:
#             pickle.dump(state, f)
#
#     def _print_training_summary(self, completed_models, failed_models):
#         """Print and save training summary"""
#         print("\n" + "=" * 60)
#         print("TRAINING SUMMARY")
#         print("=" * 60)
#
#         print(f"\nCompleted Models ({len(completed_models)}):")
#         for model in completed_models:
#             if model in self.pipeline.results:
#                 metrics = self.pipeline.results[model]['test_metrics']
#                 print(f"  ✅ {model}: NSE={metrics['NSE']:.4f}, R²={metrics['R2']:.4f}")
#             else:
#                 print(f"  ✅ {model}")
#
#         if failed_models:
#             print(f"\nFailed Models ({len(failed_models)}):")
#             for model, error in failed_models:
#                 print(f"  ❌ {model}: {error[:50]}...")
#
#         # Save summary to file
#         summary_file = os.path.join(self.checkpoint_dir, 'training_summary.txt')
#         with open(summary_file, 'w') as f:
#             f.write(f"Training Summary - {datetime.now()}\n")
#             f.write("=" * 60 + "\n\n")
#
#             f.write(f"Configuration: {self.model_config_file}\n")
#             f.write(f"Site: {self.model_config.get('site_name', 'Unknown')}\n\n")
#
#             f.write(f"Completed Models ({len(completed_models)}):\n")
#             for model in completed_models:
#                 if model in self.pipeline.results:
#                     metrics = self.pipeline.results[model]['test_metrics']
#                     f.write(f"  {model}:\n")
#                     for metric_name, value in metrics.items():
#                         f.write(f"    {metric_name}: {value:.4f}\n")
#                 else:
#                     f.write(f"  {model}: No results available\n")
#
#             if failed_models:
#                 f.write(f"\nFailed Models ({len(failed_models)}):\n")
#                 for model, error in failed_models:
#                     f.write(f"  {model}: {error}\n")
#
#         print(f"\nSummary saved to: {summary_file}")
#
#     def run_complete_pipeline(self):
#         """Run the complete pipeline with all features"""
#         print("\n" + "=" * 80)
#         print("ENHANCED ML PIPELINE WITH YAML CONFIGURATION")
#         print("=" * 80)
#         print(f"Configuration: {self.model_config_file}")
#         print(f"Site: {self.model_config.get('site_name', 'Unknown')}")
#
#         # Check for existing models
#         status = self.check_existing_models()
#
#         if status == 'loaded':
#             # Models loaded, just show summary and export
#             self.pipeline.print_summary()
#
#             response = input("\nExport results? (y/n): ").lower()
#             if response == 'y':
#                 self.export_results()
#
#         else:
#             # Train models (fresh or continuing)
#             completed, failed = self.train_with_checkpoints()
#
#             if completed:
#                 # Show performance summary
#                 self.pipeline.print_summary()
#
#                 # Export results
#                 response = input("\nExport comprehensive results? (y/n): ").lower()
#                 if response == 'y':
#                     self.export_results()
#             else:
#                 print("\nNo models were successfully trained.")
#
#         print("\n" + "=" * 80)
#         print("PIPELINE COMPLETE!")
#         print("=" * 80)
#
#         return self.pipeline
#
#     def check_existing_models(self):
#         """Check if we have previously saved models"""
#         params_file = os.path.join(self.save_dir, 'best_parameters.json')
#         if os.path.exists(params_file):
#             print("\n" + "=" * 60)
#             print("EXISTING MODELS FOUND")
#             print("=" * 60)
#
#             response = input("\nFound existing saved models. Options:\n"
#                              "1. Load existing models and skip training (l)\n"
#                              "2. Continue training remaining models (c)\n"
#                              "3. Start fresh training (f)\n"
#                              "Choice [l/c/f]: ").lower()
#
#             if response == 'l':
#                 print("\nLoading existing models...")
#                 self.pipeline.load_all_models()
#                 return 'loaded'
#             elif response == 'c':
#                 print("\nContinuing with remaining models...")
#                 try:
#                     self.pipeline.load_all_models()
#                 except:
#                     print("Warning: Could not load all existing models")
#                 return 'continue'
#             else:
#                 print("\nStarting fresh training...")
#                 self._clear_checkpoints()
#                 return 'fresh'
#         return 'fresh'
#
#     def _clear_checkpoints(self):
#         """Clear all checkpoint files"""
#         for file in os.listdir(self.checkpoint_dir):
#             if file.endswith('_done.txt'):
#                 os.remove(os.path.join(self.checkpoint_dir, file))
#         print("Cleared all checkpoints")
#
#     def export_results(self):
#         """Export all results using ResultsExporter"""
#         print("\n" + "=" * 60)
#         print("EXPORTING RESULTS")
#         print("=" * 60)
#
#         if not self.pipeline.results:
#             print("No results to export. Please train models first.")
#             return
#
#         exporter = ResultsExporter(self.pipeline, self.results_dir)
#         exporter.export_all_results()
#
#         print(f"\n✅ Results exported to: {self.results_dir}/")
#
#
# def main():
#     """Main entry point with YAML configuration"""
#
#     # Set configuration files
#     config_file = 'hyperparameter_config.json'
#     model_config_file = 'model_config.yaml'
#
#     # Check for hyperparameter configuration
#     if not os.path.exists(config_file):
#         print(f"❌ Configuration file {config_file} not found!")
#         print("Please ensure hyperparameter_config.json exists!")
#         return
#
#     # Create and run enhanced pipeline
#     training_pipeline = EnhancedTrainingPipeline(
#         config_file=config_file,
#         model_config_file=model_config_file
#     )
#
#     # Interactive model selection
#     print("\nWould you like to modify model selection? (y/n): ", end='')
#     if input().lower() == 'y':
#         print("\nAvailable models:")
#         all_models = []
#         for category in ['traditional_models', 'deep_learning_models', 'hydrodl_models']:
#             if category in training_pipeline.model_config:
#                 print(f"\n{category.replace('_', ' ').title()}:")
#                 for model_name, config in training_pipeline.model_config[category].items():
#                     status = "✓" if config.get('enabled', False) else "✗"
#                     print(f"  [{status}] {model_name}")
#                     all_models.append(model_name)
#
#         print("\nEnter model names to toggle (comma-separated) or press Enter to continue:")
#         toggle_input = input().strip()
#
#         if toggle_input:
#             models_to_toggle = [m.strip() for m in toggle_input.split(',')]
#             for model in models_to_toggle:
#                 if model in all_models:
#                     # Toggle current status
#                     current_status = False
#                     for category in ['traditional_models', 'deep_learning_models', 'hydrodl_models']:
#                         if category in training_pipeline.model_config:
#                             if model in training_pipeline.model_config[category]:
#                                 current_status = training_pipeline.model_config[category][model].get('enabled', False)
#                                 break
#
#                     training_pipeline.update_model_status(model, not current_status)
#                 else:
#                     print(f"⚠️ Model '{model}' not found")
#
#     # Run pipeline
#     pipeline = training_pipeline.run_complete_pipeline()
#
#     return pipeline
#
#
# if __name__ == "__main__":
#     # Run the enhanced pipeline
#     pipeline = main()