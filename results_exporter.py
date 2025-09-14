import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class ResultsExporter:
    """Export and visualize ML pipeline results comprehensively"""

    def __init__(self, pipeline, output_dir='Bantai_results'):
        self.pipeline = pipeline
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_all_results(self):
        """Export all results in multiple formats"""
        print("\n" + "=" * 60)
        print("EXPORTING COMPREHENSIVE RESULTS")
        print("=" * 60)

        # 1. Export predictions for full dataset
        self._export_predictions()

        # 2. Export performance metrics
        self._export_metrics()

        # 3. Export model comparisons
        self._export_model_comparison()

        # 4. Export best model config for extended simulation  # ADD THIS
        self._export_best_model_config()  # ADD THIS

        # 5. Generate visualizations
        self._create_visualizations()

        # 6. Export feature importance
        self._export_feature_importance()

        # 7. Create summary report
        self._create_summary_report()

        print(f"\n✅ All results exported to {self.output_dir}/")

    def _export_best_model_config(self):
        """Export best model configuration for extended simulation use"""
        print("\nExporting best model configuration...")

        if not self.pipeline.results:
            print("  No results to export")
            return

        # Find best model by test NSE
        best_model_name = max(self.pipeline.results.keys(),
                              key=lambda k: self.pipeline.results[k]['test_metrics']['NSE'])

        best_result = self.pipeline.results[best_model_name]

        # Create configuration for extended simulation
        config = {
            "best_model": {
                "name": best_model_name,
                "test_nse": best_result['test_metrics']['NSE'],
                "test_r2": best_result['test_metrics']['R2'],
                "test_rmse": best_result['test_metrics']['RMSE'],
                "test_kge": best_result['test_metrics']['KGE']
            },
            "model_specific": {},
            "data_info": {
                "feature_names": self.pipeline.feature_names,
                "num_features": len(self.pipeline.feature_names),
                "train_samples": len(self.pipeline.X_train),
                "test_samples": len(self.pipeline.X_test),
                "flow_threshold": self.pipeline.flow_threshold,
                "test_size": self.pipeline.test_size
            },
            "scaling": {
                "scaler_X_type": type(self.pipeline.scaler_X).__name__,
                "scaler_y_type": type(self.pipeline.scaler_y).__name__
            }
        }

        # Add model-specific configuration
        if best_model_name == 'LSTM':
            config["model_specific"] = {
                "sequence_length": best_result.get('sequence_length', self.pipeline.sequence_length),
                "architecture": best_result.get('architecture', 'unknown'),
                "input_shape": [best_result.get('sequence_length', self.pipeline.sequence_length),
                                len(self.pipeline.feature_names)],
                "requires_sequences": True,
                "dates_offset": best_result.get('sequence_length', self.pipeline.sequence_length)
            }
        elif best_model_name == 'SVM':
            config["model_specific"] = {
                "requires_sequences": False,
                "uses_scaled_targets": True,
                "dates_offset": 0
            }
        else:
            config["model_specific"] = {
                "requires_sequences": False,
                "uses_scaled_targets": False,
                "dates_offset": 0
            }

        # Add hyperparameters if available
        if hasattr(self.pipeline, 'best_params') and best_model_name in self.pipeline.best_params:
            config["hyperparameters"] = self.pipeline.best_params[best_model_name]

        # Add all model performances for comparison
        config["all_models_performance"] = {}
        for model_name, result in self.pipeline.results.items():
            config["all_models_performance"][model_name] = {
                "test_nse": result['test_metrics']['NSE'],
                "test_r2": result['test_metrics']['R2'],
                "test_rmse": result['test_metrics']['RMSE'],
                "test_kge": result['test_metrics']['KGE'],
                "rank": 0  # Will be filled below
            }

        # Add rankings
        sorted_models = sorted(config["all_models_performance"].items(),
                               key=lambda x: x[1]['test_nse'], reverse=True)
        for rank, (model_name, _) in enumerate(sorted_models, 1):
            config["all_models_performance"][model_name]["rank"] = rank

        # Export metadata
        config["export_info"] = {
            "timestamp": datetime.now().isoformat(),
            "total_models_trained": len(self.pipeline.results),
            "data_file": getattr(self.pipeline, 'filepath', 'unknown'),
            "config_version": "1.0"
        }

        # Save JSON
        json_file = os.path.join(self.output_dir, 'best_model_config.json')
        with open(json_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        print(f"  Saved best model config to {json_file}")
        print(f"  Best model: {best_model_name} (NSE: {best_result['test_metrics']['NSE']:.4f})")

        if best_model_name == 'LSTM':
            seq_len = config["model_specific"]["sequence_length"]
            print(f"  LSTM sequence length: {seq_len}")

        # Also create a simple lookup file for quick access
        simple_config = {
            "best_model_name": best_model_name,
            "sequence_length": config["model_specific"].get("sequence_length", 0),
            "requires_sequences": config["model_specific"]["requires_sequences"],
            "dates_offset": config["model_specific"]["dates_offset"],
            "test_nse": config["best_model"]["test_nse"]
        }

        simple_file = os.path.join(self.output_dir, 'model_lookup.json')
        with open(simple_file, 'w') as f:
            json.dump(simple_config, f, indent=2)

        print(f"  Saved simple lookup to {simple_file}")

        return config

    def _export_predictions(self):
        """Export predictions for entire dataset with length validation"""
        print("Exporting predictions...")

        for model_name in self.pipeline.results:
            try:
                result = self.pipeline.results[model_name]

                # Prepare data based on model type with length validation
                if model_name == 'LSTM':
                    # LSTM has different sequence lengths
                    train_dates = self.pipeline.dates_train[self.pipeline.sequence_length:]
                    test_dates = self.pipeline.dates_test[self.pipeline.sequence_length:]
                    y_train_actual = result.get('y_train_seq', self.pipeline.y_train[self.pipeline.sequence_length:])
                    y_test_actual = result.get('y_test_seq', self.pipeline.y_test[self.pipeline.sequence_length:])
                else:
                    train_dates = self.pipeline.dates_train
                    test_dates = self.pipeline.dates_test
                    y_train_actual = self.pipeline.y_train
                    y_test_actual = self.pipeline.y_test

                # Get predictions
                y_pred_train = result['y_pred_train']
                y_pred_test = result['y_pred_test']

                # Validate and fix training data lengths
                min_train_len = min(len(train_dates), len(y_train_actual), len(y_pred_train))
                if len(train_dates) != len(y_train_actual) or len(train_dates) != len(y_pred_train):
                    print(f"  Warning: Length mismatch in {model_name} training data")
                    print(
                        f"    Dates: {len(train_dates)}, Actual: {len(y_train_actual)}, Predicted: {len(y_pred_train)}")
                    print(f"    Using minimum length: {min_train_len}")

                    # Truncate to minimum length
                    train_dates = train_dates[:min_train_len]
                    y_train_actual = y_train_actual[:min_train_len]
                    y_pred_train = y_pred_train[:min_train_len]

                # Validate and fix test data lengths
                min_test_len = min(len(test_dates), len(y_test_actual), len(y_pred_test))
                if len(test_dates) != len(y_test_actual) or len(test_dates) != len(y_pred_test):
                    print(f"  Warning: Length mismatch in {model_name} test data")
                    print(f"    Dates: {len(test_dates)}, Actual: {len(y_test_actual)}, Predicted: {len(y_pred_test)}")
                    print(f"    Using minimum length: {min_test_len}")

                    # Truncate to minimum length
                    test_dates = test_dates[:min_test_len]
                    y_test_actual = y_test_actual[:min_test_len]
                    y_pred_test = y_pred_test[:min_test_len]

                # Create comprehensive DataFrames with validated lengths
                train_df = pd.DataFrame({
                    'date': train_dates,
                    'observed_flow': y_train_actual,
                    'predicted_flow': y_pred_train,
                    'split': 'train'
                })

                test_df = pd.DataFrame({
                    'date': test_dates,
                    'observed_flow': y_test_actual,
                    'predicted_flow': y_pred_test,
                    'split': 'test'
                })

                # Combine and add error metrics
                full_df = pd.concat([train_df, test_df], ignore_index=True)
                full_df['error'] = full_df['predicted_flow'] - full_df['observed_flow']
                full_df['absolute_error'] = np.abs(full_df['error'])

                # Handle division by zero for relative error
                full_df['relative_error'] = np.where(
                    full_df['observed_flow'] != 0,
                    (full_df['error'] / full_df['observed_flow'] * 100),
                    np.nan
                )
                full_df['relative_error'] = full_df['relative_error'].replace([np.inf, -np.inf], np.nan)

                # Add time-based features for analysis
                full_df['date'] = pd.to_datetime(full_df['date'])
                full_df['year'] = full_df['date'].dt.year
                full_df['month'] = full_df['date'].dt.month
                full_df['season'] = full_df['month'].apply(lambda x: 'Winter' if x in [12, 1, 2]
                else 'Spring' if x in [3, 4, 5]
                else 'Summer' if x in [6, 7, 8]
                else 'Fall')

                # Save to CSV
                output_file = os.path.join(self.output_dir, f'{model_name}_predictions.csv')
                full_df.to_csv(output_file, index=False)
                print(f"  Saved {model_name} predictions to {output_file}")

                # Also save as Excel with formatting
                excel_file = os.path.join(self.output_dir, f'{model_name}_predictions.xlsx')
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    full_df.to_excel(writer, sheet_name='Predictions', index=False)

                    # Add summary statistics sheet
                    try:
                        summary_stats = full_df.groupby('split').agg({
                            'observed_flow': ['mean', 'std', 'min', 'max'],
                            'predicted_flow': ['mean', 'std', 'min', 'max'],
                            'absolute_error': ['mean', 'std', 'min', 'max'],
                            'relative_error': ['mean', 'std']
                        }).round(3)
                        summary_stats.to_excel(writer, sheet_name='Summary_Stats')

                        # Add seasonal performance
                        seasonal_stats = full_df.groupby(['split', 'season']).agg({
                            'absolute_error': 'mean',
                            'relative_error': 'mean'
                        }).round(3)
                        seasonal_stats.to_excel(writer, sheet_name='Seasonal_Performance')
                    except Exception as e:
                        print(f"    Warning: Could not create summary sheets for {model_name}: {e}")

                print(f"  Saved {model_name} Excel report to {excel_file}")

            except Exception as e:
                print(f"  Error exporting predictions for {model_name}: {str(e)}")
                print(f"    Skipping {model_name} and continuing with other models...")
                continue

    def _export_metrics(self):
        """Export detailed performance metrics"""
        print("\nExporting performance metrics...")

        metrics_data = []
        for model_name in self.pipeline.results:
            result = self.pipeline.results[model_name]

            # Combine train and test metrics
            row = {'Model': model_name}

            for split in ['train', 'test']:
                metrics = result[f'{split}_metrics']
                for metric_name, value in metrics.items():
                    row[f'{split}_{metric_name}'] = value

            # Add overfit indicator
            row['NSE_overfit_gap'] = row['train_NSE'] - row['test_NSE']
            row['R2_overfit_gap'] = row['train_R2'] - row['test_R2']

            metrics_data.append(row)

        # Create DataFrame and sort by test NSE
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.sort_values('test_NSE', ascending=False)

        # Save metrics
        metrics_file = os.path.join(self.output_dir, 'all_models_metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)
        print(f"  Saved metrics to {metrics_file}")

        # Also save as formatted Excel
        excel_file = os.path.join(self.output_dir, 'all_models_metrics.xlsx')
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            metrics_df.to_excel(writer, sheet_name='All_Metrics', index=False)

            # Add ranking sheet
            ranking_df = metrics_df[['Model', 'test_NSE', 'test_R2', 'test_RMSE', 'test_KGE']].copy()
            ranking_df['Rank'] = range(1, len(ranking_df) + 1)
            ranking_df.to_excel(writer, sheet_name='Model_Ranking', index=False)

        print(f"  Saved metrics Excel to {excel_file}")

        return metrics_df

    def _export_model_comparison(self):
        """Export model comparison for best performing period"""
        print("\nExporting model comparison...")

        # Get common test dates (accounting for LSTM sequence length)
        min_test_length = min(len(self.pipeline.results[m]['y_pred_test'])
                              for m in self.pipeline.results)

        comparison_data = {
            'date': self.pipeline.dates_test[:min_test_length],
            'observed': self.pipeline.y_test[:min_test_length]
        }

        # Add predictions from each model
        for model_name in self.pipeline.results:
            comparison_data[f'{model_name}_predicted'] = (
                self.pipeline.results[model_name]['y_pred_test'][:min_test_length]
            )

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df['date'] = pd.to_datetime(comparison_df['date'])

        # Save comparison
        comparison_file = os.path.join(self.output_dir, 'model_comparison_test.csv')
        comparison_df.to_csv(comparison_file, index=False)
        print(f"  Saved model comparison to {comparison_file}")

        # Calculate ensemble predictions (simple average)
        model_cols = [col for col in comparison_df.columns if '_predicted' in col]
        comparison_df['ensemble_mean'] = comparison_df[model_cols].mean(axis=1)
        comparison_df['ensemble_median'] = comparison_df[model_cols].median(axis=1)

        # Save with ensemble
        ensemble_file = os.path.join(self.output_dir, 'model_comparison_with_ensemble.csv')
        comparison_df.to_csv(ensemble_file, index=False)
        print(f"  Saved ensemble predictions to {ensemble_file}")

        return comparison_df

    def _create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")

        # 1. Time series plots for each model
        for model_name in self.pipeline.results:
            self._plot_time_series(model_name)

        # 2. Model comparison plot
        self._plot_model_comparison()

        # 3. Performance comparison bar chart
        self._plot_performance_comparison()

        # 4. Scatter plots (observed vs predicted)
        self._plot_scatter_comparison()

        # 5. Error distribution plots
        self._plot_error_distributions()

        print(f"  All plots saved to {self.output_dir}/")

    def _plot_time_series(self, model_name):
        """Create time series plot for a model"""
        result = self.pipeline.results[model_name]

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Training period
        if model_name == 'LSTM':
            train_dates = self.pipeline.dates_train[self.pipeline.sequence_length:]
            y_train = result.get('y_train_seq', self.pipeline.y_train[self.pipeline.sequence_length:])
        else:
            train_dates = self.pipeline.dates_train
            y_train = self.pipeline.y_train

        axes[0].plot(train_dates, y_train, 'b-', label='Observed', alpha=0.7)
        axes[0].plot(train_dates, result['y_pred_train'][:len(y_train)], 'r-', label='Predicted', alpha=0.7)
        axes[0].set_title(f'{model_name} - Training Period')
        axes[0].set_ylabel('Flow (m³/s)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Testing period
        if model_name == 'LSTM':
            test_dates = self.pipeline.dates_test[self.pipeline.sequence_length:]
            y_test = result.get('y_test_seq', self.pipeline.y_test[self.pipeline.sequence_length:])
        else:
            test_dates = self.pipeline.dates_test
            y_test = self.pipeline.y_test

        axes[1].plot(test_dates, y_test, 'b-', label='Observed', alpha=0.7)
        axes[1].plot(test_dates, result['y_pred_test'][:len(y_test)], 'r-', label='Predicted', alpha=0.7)
        axes[1].set_title(f'{model_name} - Testing Period (NSE: {result["test_metrics"]["NSE"]:.3f})')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Flow (m³/s)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{model_name}_timeseries.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_model_comparison(self):
        """Create interactive comparison plot"""
        comparison_df = self._export_model_comparison()

        # Create interactive plot with plotly
        fig = go.Figure()

        # Add observed data
        fig.add_trace(go.Scatter(
            x=comparison_df['date'],
            y=comparison_df['observed'],
            mode='lines',
            name='Observed',
            line=dict(color='black', width=2)
        ))

        # Add each model's predictions
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i, model_name in enumerate(self.pipeline.results):
            if f'{model_name}_predicted' in comparison_df.columns:
                fig.add_trace(go.Scatter(
                    x=comparison_df['date'],
                    y=comparison_df[f'{model_name}_predicted'],
                    mode='lines',
                    name=model_name,
                    line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.7
                ))

        # Add ensemble if exists
        if 'ensemble_mean' in comparison_df.columns:
            fig.add_trace(go.Scatter(
                x=comparison_df['date'],
                y=comparison_df['ensemble_mean'],
                mode='lines',
                name='Ensemble Mean',
                line=dict(color='darkgreen', width=2, dash='dash')
            ))

        fig.update_layout(
            title='Model Comparison - Test Period',
            xaxis_title='Date',
            yaxis_title='Flow (m³/s)',
            hovermode='x unified',
            height=600
        )

        html_file = os.path.join(self.output_dir, 'interactive_comparison.html')
        fig.write_html(html_file)
        print(f"  Saved interactive plot to {html_file}")

    def _plot_performance_comparison(self):
        """Create performance comparison bar chart"""
        metrics_df = self._export_metrics()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        metrics_to_plot = ['NSE', 'R2', 'RMSE', 'KGE']

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]

            x = range(len(metrics_df))
            width = 0.35

            train_values = metrics_df[f'train_{metric}'].values
            test_values = metrics_df[f'test_{metric}'].values

            ax.bar([i - width / 2 for i in x], train_values, width, label='Train', alpha=0.7)
            ax.bar([i + width / 2 for i in x], test_values, width, label='Test', alpha=0.7)

            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_df['Model'].values, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_scatter_comparison(self):
        """Create scatter plots for all models"""
        n_models = len(self.pipeline.results)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, model_name in enumerate(self.pipeline.results):
            if idx >= 6:  # Maximum 6 subplots
                break

            ax = axes[idx]
            result = self.pipeline.results[model_name]

            # Use appropriate y values based on model
            if model_name == 'LSTM':
                y_test = result.get('y_test_seq', self.pipeline.y_test[self.pipeline.sequence_length:])
            else:
                y_test = self.pipeline.y_test

            y_pred = result['y_pred_test'][:len(y_test)]

            ax.scatter(y_test, y_pred, alpha=0.5, s=10)

            # Add 1:1 line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            ax.set_xlabel('Observed Flow (m³/s)')
            ax.set_ylabel('Predicted Flow (m³/s)')
            ax.set_title(f'{model_name} (R²={result["test_metrics"]["R2"]:.3f})')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_models, 6):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'scatter_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_error_distributions(self):
        """Plot error distributions for all models"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for model_name in self.pipeline.results:
            result = self.pipeline.results[model_name]

            # Calculate errors
            if model_name == 'LSTM':
                y_test = result.get('y_test_seq', self.pipeline.y_test[self.pipeline.sequence_length:])
            else:
                y_test = self.pipeline.y_test

            errors = result['y_pred_test'][:len(y_test)] - y_test
            relative_errors = (errors / y_test * 100)[y_test > 0]  # Avoid division by zero

            # Absolute errors histogram
            axes[0].hist(errors, bins=50, alpha=0.5, label=model_name, density=True)

            # Relative errors histogram
            axes[1].hist(relative_errors, bins=50, alpha=0.5, label=model_name, density=True, range=(-100, 100))

        axes[0].set_xlabel('Absolute Error (m³/s)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Absolute Error Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Relative Error (%)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Relative Error Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'error_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _export_feature_importance(self):
        """Export feature importance for tree-based models"""
        print("\nExporting feature importance...")

        importance_data = {}

        for model_name in ['XGBoost', 'LightGBM', 'RandomForest']:
            if model_name in self.pipeline.models:
                model = self.pipeline.models[model_name]
                if hasattr(model, 'feature_importances_'):
                    importance_data[model_name] = pd.DataFrame({
                        'feature': self.pipeline.feature_names,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)

        if importance_data:
            # Save to Excel with multiple sheets
            excel_file = os.path.join(self.output_dir, 'feature_importance.xlsx')
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for model_name, df in importance_data.items():
                    df.to_excel(writer, sheet_name=model_name, index=False)
            print(f"  Saved feature importance to {excel_file}")

            # Create visualization
            fig, axes = plt.subplots(1, len(importance_data), figsize=(15, 6))
            if len(importance_data) == 1:
                axes = [axes]

            for idx, (model_name, df) in enumerate(importance_data.items()):
                top_features = df.head(15)
                axes[idx].barh(range(len(top_features)), top_features['importance'].values)
                axes[idx].set_yticks(range(len(top_features)))
                axes[idx].set_yticklabels(top_features['feature'].values)
                axes[idx].set_xlabel('Importance')
                axes[idx].set_title(f'{model_name} - Top 15 Features')
                axes[idx].invert_yaxis()

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()

    # Fix for _create_summary_report method in results_exporter.py

    def _create_summary_report(self):
        """Create comprehensive text summary report"""
        print("\nCreating summary report...")

        report_file = os.path.join(self.output_dir, 'SUMMARY_REPORT.txt')

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("HYDROLOGICAL ML PIPELINE - COMPREHENSIVE RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Report metadata
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data File: {self.pipeline.filepath}\n")
            f.write(
                f"Train/Test Split: {100 * (1 - self.pipeline.test_size):.0f}/{100 * self.pipeline.test_size:.0f}\n")
            f.write(f"Number of Features: {len(self.pipeline.feature_names)}\n")
            f.write(f"Training Samples: {len(self.pipeline.X_train)}\n")
            f.write(f"Test Samples: {len(self.pipeline.X_test)}\n\n")

            # Data period information
            f.write("DATA PERIOD INFORMATION:\n")
            f.write("-" * 40 + "\n")

            # Convert numpy datetime64 to pandas datetime for proper handling
            train_start = pd.to_datetime(self.pipeline.dates_train[0])
            train_end = pd.to_datetime(self.pipeline.dates_train[-1])
            test_start = pd.to_datetime(self.pipeline.dates_test[0])
            test_end = pd.to_datetime(self.pipeline.dates_test[-1])

            f.write(f"Training Period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}\n")
            f.write(f"Testing Period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}\n")

            # Calculate total years properly
            total_days = (test_end - train_start).days
            total_years = total_days / 365.25
            f.write(f"Total Years: {total_years:.1f}\n\n")

            # Model performance summary
            f.write("MODEL PERFORMANCE SUMMARY (Test Set):\n")
            f.write("-" * 40 + "\n")

            # Sort models by test NSE
            sorted_models = sorted(self.pipeline.results.items(),
                                   key=lambda x: x[1]['test_metrics']['NSE'],
                                   reverse=True)

            for rank, (model_name, result) in enumerate(sorted_models, 1):
                metrics = result['test_metrics']
                f.write(f"\n{rank}. {model_name}:\n")
                f.write(f"   NSE:  {metrics['NSE']:.4f}\n")
                f.write(f"   R²:   {metrics['R2']:.4f}\n")
                f.write(f"   RMSE: {metrics['RMSE']:.4f} m³/s\n")
                f.write(f"   MAE:  {metrics['MAE']:.4f} m³/s\n")
                f.write(f"   KGE:  {metrics['KGE']:.4f}\n")
                f.write(f"   RE:   {metrics['RE']:.2f}%\n")

                # Add architecture info for LSTM
                if model_name == 'LSTM':
                    arch = result.get('architecture', 'unknown')
                    f.write(f"   Architecture: {arch}\n")

            # Best model details
            f.write("\n" + "=" * 80 + "\n")
            f.write("BEST MODEL DETAILS:\n")
            f.write("-" * 40 + "\n")
            best_model_name = sorted_models[0][0]
            best_model_metrics = sorted_models[0][1]['test_metrics']

            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Test NSE: {best_model_metrics['NSE']:.4f}\n")
            f.write(f"Test R²: {best_model_metrics['R2']:.4f}\n\n")

            f.write("Hyperparameters:\n")
            if best_model_name in self.pipeline.best_params:
                for param, value in self.pipeline.best_params[best_model_name].items():
                    # Format the value based on its type
                    if isinstance(value, float):
                        f.write(f"  {param}: {value:.6f}\n")
                    else:
                        f.write(f"  {param}: {value}\n")

            # Overfitting analysis
            f.write("\n" + "=" * 80 + "\n")
            f.write("OVERFITTING ANALYSIS:\n")
            f.write("-" * 40 + "\n")

            for model_name, result in sorted_models:
                train_nse = result['train_metrics']['NSE']
                test_nse = result['test_metrics']['NSE']
                gap = train_nse - test_nse

                f.write(f"{model_name:15} Train NSE: {train_nse:.4f}, Test NSE: {test_nse:.4f}, Gap: {gap:.4f}")

                if gap > 0.15:
                    f.write(" Potential overfitting\n")
                elif gap < 0.05 and test_nse > 0.95:
                    f.write(" Suspiciously low gap - check for data leakage\n")
                else:
                    f.write(" Good generalization\n")

            # Files generated
            f.write("\n" + "=" * 80 + "\n")
            f.write("OUTPUT FILES GENERATED:\n")
            f.write("-" * 40 + "\n")

            output_files = os.listdir(self.output_dir)
            for file in sorted(output_files):
                f.write(f"  - {file}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"  Summary report saved to {report_file}")


def export_results(pipeline, output_dir='Bantai_results'):
    """Main function to export all results"""
    exporter = ResultsExporter(pipeline, output_dir)
    exporter.export_all_results()

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\nAll results have been exported to: {output_dir}/")
    print("\nFiles created:")
    print("  - Individual model predictions (CSV & Excel)")
    print("  - Model comparison with ensemble predictions")
    print("  - Performance metrics summary")
    print("  - Time series plots for each model")
    print("  - Interactive comparison plot (HTML)")
    print("  - Feature importance analysis")
    print("  - Comprehensive summary report")

    return output_dir


# Usage example
if __name__ == "__main__":
    # Assuming you have a trained pipeline
    from ML import AdvancedHydrologicalMLPipeline

    # Load pipeline and train models (or load saved models)
    pipeline = AdvancedHydrologicalMLPipeline()
    pipeline.load_and_prepare_data()

    # If models are already trained, just export results
    # Otherwise, train first:
    # pipeline.optimize_all_models()

    # Export all results
    output_directory = export_results(pipeline)

    print(f"\n✅ Results ready for analysis in: {output_directory}/")
