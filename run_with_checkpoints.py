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


class CompleteTrainingPipeline:
    """
    Complete training pipeline that combines:
    1. Checkpoint functionality (from your original script)
    2. Proper model saving (from enhanced_ml_pipeline)
    3. Results export (from results_exporter)
    """

    def __init__(self, config_file='hyperparameter_config.json'):
        self.config_file = config_file
        site_name = curren_site_name
        self.checkpoint_dir = f'{site_name}_model_checkpoints'
        self.save_dir = f'{site_name}_saved_models'
        self.results_dir = f'{site_name}_results'

        # Create all necessary directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize the enhanced pipeline
        self.pipeline = EnhancedMLPipeline(
            config_path=self.config_file,
            save_dir=self.save_dir
        )

        # Model training configuration
        self.models_to_train = [
            ('XGBoost', self.pipeline.optimize_xgboost),
            ('LightGBM', self.pipeline.optimize_lightgbm),
            ('RandomForest', self.pipeline.optimize_random_forest),
            ('Cubist', self.pipeline.optimize_cubist),
            ('SVM', self.pipeline.optimize_svm),
            ('LSTM', self.pipeline.optimize_lstm)
        ]

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
                self.pipeline.load_all_models()
                return 'loaded'
            elif response == 'c':
                print("\nContinuing with remaining models...")
                # Load what we have
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

        # Load and prepare data
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

        exporter = ResultsExporter(self.pipeline, self.results_dir)
        exporter.export_all_results()

        print(f"\n✅ Results exported to: {self.results_dir}/")

    def run_complete_pipeline(self):
        """Run the complete pipeline with all features"""
        print("\n" + "=" * 80)
        print("COMPLETE ML PIPELINE WITH CHECKPOINTS, SAVING, AND EXPORT")
        print("=" * 80)

        # Check for existing models
        status = self.check_existing_models()

        if status == 'loaded':
            # Models loaded, just show summary and export
            self.pipeline.print_summary()

            response = input("\nExport results? (y/n): ").lower()
            if response == 'y':
                self.export_results()

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
    training_pipeline = CompleteTrainingPipeline(config_file)
    pipeline = training_pipeline.run_complete_pipeline()

    # Optional: Return pipeline for further use
    return pipeline


if __name__ == "__main__":
    curren_site_name = 'Bengbu'
    # This replaces your original run_with_checkpoints()
    pipeline = main()