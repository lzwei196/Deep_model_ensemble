# hydrodl_integration.py
"""
Integration module for HydroDL models into the main pipeline
Ensures consistent naming between YAML config and implementation
"""

import yaml
import torch
from hydrodl_models import (
    HydroDL_LSTM,
    HydroDL_CNN1D,
    HydroDL_CudnnLSTM,
    HydroDL_MTS_LSTM,
    integrate_hydrodl_models
)

# Model mapping - connects YAML names to actual classes
HYDRODL_MODEL_MAPPING = {
    'HydroDL_LSTM': {
        'class': HydroDL_LSTM,
        'method': 'optimize_hydrodl_lstm',
        'requires_gpu': False
    },
    'HydroDL_CNN1D': {
        'class': HydroDL_CNN1D,
        'method': 'optimize_hydrodl_cnn1d',
        'requires_gpu': False
    },
    'HydroDL_CudnnLSTM': {
        'class': HydroDL_CudnnLSTM,
        'method': 'optimize_hydrodl_cudnn_lstm',
        'requires_gpu': True
    },
    'HydroDL_MTS_LSTM': {
        'class': HydroDL_MTS_LSTM,
        'method': 'optimize_hydrodl_mts_lstm',
        'requires_gpu': False
    }
}


def load_config(config_path='model_config.yaml'):
    """Load model configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Warning: {config_path} not found, using default configuration")
        return get_default_config()


def get_default_config():
    """Default configuration if YAML file is not found"""
    return {
        'hydrodl_models': {
            'HydroDL_LSTM': {'enabled': True, 'priority': 11},
            'HydroDL_CNN1D': {'enabled': True, 'priority': 12},
            'HydroDL_CudnnLSTM': {'enabled': torch.cuda.is_available(), 'priority': 13},
            'HydroDL_MTS_LSTM': {'enabled': True, 'priority': 14}
        },
        'training_settings': {
            'optuna_trials': 30,
            'max_epochs_dl': 150
        }
    }


def get_enabled_hydrodl_models(config):
    """Get list of enabled HydroDL models sorted by priority"""
    hydrodl_models = config.get('hydrodl_models', {})
    enabled_models = []

    for model_name, model_config in hydrodl_models.items():
        if model_config.get('enabled', False):
            # Check GPU requirements
            model_info = HYDRODL_MODEL_MAPPING.get(model_name, {})
            if model_info.get('requires_gpu', False) and not torch.cuda.is_available():
                print(f"Warning: {model_name} requires GPU but CUDA not available, skipping")
                continue

            enabled_models.append({
                'name': model_name,
                'priority': model_config.get('priority', 999),
                'method': model_info.get('method'),
                'class': model_info.get('class')
            })

    # Sort by priority
    enabled_models.sort(key=lambda x: x['priority'])
    return enabled_models


def setup_hydrodl_pipeline(pipeline, config_path='model_config.yaml'):
    """
    Set up HydroDL models in the pipeline according to configuration

    Args:
        pipeline: The main ML pipeline object
        config_path: Path to YAML configuration file

    Returns:
        Updated pipeline with HydroDL methods attached
    """

    # Load configuration
    config = load_config(config_path)

    # Integrate HydroDL models into pipeline
    pipeline = integrate_hydrodl_models(pipeline)

    # Get enabled models
    enabled_models = get_enabled_hydrodl_models(config)

    # Store configuration in pipeline
    pipeline.hydrodl_config = config
    pipeline.hydrodl_enabled_models = enabled_models

    # Add method to run all enabled HydroDL models
    def run_hydrodl_models():
        """Run all enabled HydroDL models in priority order"""
        print(f"\n{'=' * 80}")
        print("STARTING HYDRODL MODELS TRAINING")
        print(f"{'=' * 80}")

        for model_info in pipeline.hydrodl_enabled_models:
            model_name = model_info['name']
            method_name = model_info['method']

            if hasattr(pipeline, method_name):
                print(f"\n{'-' * 60}")
                print(f"Training {model_name} (Priority: {model_info['priority']})")
                print(f"{'-' * 60}")

                try:
                    method = getattr(pipeline, method_name)
                    method()
                    print(f"✓ {model_name} completed successfully")
                except Exception as e:
                    print(f"✗ Error training {model_name}: {str(e)}")
                    continue
            else:
                print(f"Warning: Method {method_name} not found for {model_name}")

        print(f"\n{'=' * 80}")
        print("HYDRODL MODELS TRAINING COMPLETED")
        print(f"{'=' * 80}")

    # Attach the method to pipeline
    pipeline.run_hydrodl_models = run_hydrodl_models

    return pipeline


def validate_config(config_path='model_config.yaml'):
    """Validate the configuration file"""
    config = load_config(config_path)

    issues = []

    # Check if hydrodl_models section exists
    if 'hydrodl_models' not in config:
        issues.append("Missing 'hydrodl_models' section in configuration")
        return issues

    # Check each enabled model
    for model_name, model_config in config['hydrodl_models'].items():
        if model_config.get('enabled', False):
            # Check if model exists in our mapping
            if model_name not in HYDRODL_MODEL_MAPPING:
                issues.append(f"Unknown HydroDL model: {model_name}")

            # Check GPU requirements
            model_info = HYDRODL_MODEL_MAPPING.get(model_name, {})
            if model_info.get('requires_gpu', False) and not torch.cuda.is_available():
                issues.append(f"{model_name} requires GPU but CUDA not available")

    if not issues:
        print("✓ Configuration validation passed")
        enabled_count = sum(1 for m in config['hydrodl_models'].values() if m.get('enabled', False))
        print(f"✓ {enabled_count} HydroDL models enabled")
    else:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  ✗ {issue}")

    return issues


# Example usage
if __name__ == "__main__":
    # Validate configuration
    validate_config()

    # Show available models
    print("\nAvailable HydroDL Models:")
    for model_name, model_info in HYDRODL_MODEL_MAPPING.items():
        gpu_req = " (GPU required)" if model_info['requires_gpu'] else ""
        print(f"  - {model_name}{gpu_req}")