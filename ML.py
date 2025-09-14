import numpy as np
import pandas as pd
import json
import os
import warnings


# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Advanced ML
import xgboost as xgb
import lightgbm as lgb
import cubist
# Try to import Cubist, use CatBoost as fallback
try:
    from cubist import Cubist

    CUBIST_AVAILABLE = True
except ImportError:
    print("Warning: Cubist not available, using CatBoost as alternative")
    from catboost import CatBoostRegressor

    CUBIST_AVAILABLE = False

# Deep Learning - Fixed imports
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import (LSTM, Dense, Dropout, Input, Bidirectional,
                          GRU, Conv1D, MaxPooling1D, Flatten, concatenate,
                          GlobalMaxPooling1D, GlobalAveragePooling1D)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

# Try to import Attention layers (not available in all versions)
try:
    from keras.layers import Attention, MultiHeadAttention, LayerNormalization

    ATTENTION_AVAILABLE = True
except ImportError:
    print("Warning: Attention layers not available in this Keras version")
    ATTENTION_AVAILABLE = False

# Hyperparameter Optimization
import optuna
import keras_tuner as kt

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Utilities
import joblib
from tqdm import tqdm

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)


class ConfigManager:
    """Manages configuration loading and validation"""

    def __init__(self, config_path: str = 'hyperparameter_config.json'):
        """Load configuration from JSON file"""
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            print(f"Configuration loaded from {self.config_path}")
        else:
            raise FileNotFoundError(f"Configuration file {self.config_path} not found!")

    def get(self, *keys):
        """Get nested configuration value"""
        value = self.config
        for key in keys:
            value = value[key]
        return value


class AdvancedHydrologicalMLPipeline:
    """Advanced ML pipeline with complete model suite"""

    def __init__(self, config_path: str = 'hyperparameter_config.json'):
        """Initialize pipeline with configuration"""
        self.config = ConfigManager(config_path)
        self.models = {}
        self.results = {}
        self.best_params = {}
        self.studies = {}
        self.tuners = {}

        # Load data configuration
        self.filepath = self.config.get('data', 'filepath')
        self.flow_threshold = self.config.get('data', 'flow_threshold')
        self.test_size = self.config.get('data', 'test_size')
        self.sequence_length = self.config.get('data', 'sequence_length')

    def load_and_prepare_data(self):
        """Load and prepare data with advanced feature engineering"""
        print("Loading and preparing data...")
        df = pd.read_csv(self.filepath)
        df['time'] = pd.to_datetime(df['time'])

        # Filter valid flow data
        valid_mask = (df['flow(m^3/s)'] > self.flow_threshold) & (df['flow(m^3/s)'] != -99)
        df_valid = df[valid_mask].copy()

        print(f"Valid flow data: {len(df_valid)}/{len(df)} ({len(df_valid) / len(df) * 100:.1f}%)")

        # Advanced feature engineering
        features = self._create_features(df_valid)

        # Remove NaN rows
        df_valid = df_valid.dropna()

        # Prepare arrays
        self.feature_names = features
        self.X = df_valid[features].values
        self.y = df_valid['flow(m^3/s)'].values
        self.dates = df_valid['time'].values

        # Chronological split
        split_idx = int(len(self.X) * (1 - self.test_size))

        self.X_train = self.X[:split_idx]
        self.X_test = self.X[split_idx:]
        self.y_train = self.y[:split_idx]
        self.y_test = self.y[split_idx:]
        self.dates_train = self.dates[:split_idx]
        self.dates_test = self.dates[split_idx:]

        # Scale features
        self.scaler_X = StandardScaler()
        self.X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler_X.transform(self.X_test)

        # Scale targets
        self.scaler_y = MinMaxScaler()
        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train.reshape(-1, 1)).ravel()
        self.y_test_scaled = self.scaler_y.transform(self.y_test.reshape(-1, 1)).ravel()

        print(f"Features: {len(features)}")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")

    def _create_features(self, df):
        """Create comprehensive features"""
        features = []

        # Basic meteorological variables
        basic_vars = ['prcp(mm/day)', 'pet(mm/day)']
        for var in basic_vars:
            if var in df.columns:
                features.append(var)

        # Temperature variables
        temp_vars = ['tmax(C)', 'tmin(C)', 'tavg(C)']
        for var in temp_vars:
            if var in df.columns:
                features.append(var)

        # Additional meteorological variables
        other_vars = ['wind(m/s)', 'rh(%)', 'rs(W/m2)', 'snow(mm)', 'swe(mm)']
        for var in other_vars:
            if var in df.columns:
                features.append(var)

        # Temporal features
        df['month'] = df['time'].dt.month
        df['day_of_year'] = df['time'].dt.dayofyear
        df['season'] = df['month'].apply(lambda x: (x - 1) // 3 + 1)
        df['week_of_year'] = df['time'].dt.isocalendar().week

        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        # Lagged features
        lag_days = [1, 2, 3, 5, 7, 10, 14, 21, 30]
        for lag in lag_days:
            df[f'prcp_lag_{lag}'] = df['prcp(mm/day)'].shift(lag)
            df[f'flow_lag_{lag}'] = df['flow(m^3/s)'].shift(lag)
            df[f'pet_lag_{lag}'] = df['pet(mm/day)'].shift(lag)
            features.extend([f'prcp_lag_{lag}', f'flow_lag_{lag}', f'pet_lag_{lag}'])

        # Rolling statistics
        windows = [3, 7, 14, 30, 60]
        for window in windows:
            df[f'prcp_roll_mean_{window}'] = df['prcp(mm/day)'].rolling(window=window, min_periods=1).mean()
            df[f'prcp_roll_sum_{window}'] = df['prcp(mm/day)'].rolling(window=window, min_periods=1).sum()
            df[f'prcp_roll_max_{window}'] = df['prcp(mm/day)'].rolling(window=window, min_periods=1).max()
            df[f'prcp_roll_std_{window}'] = df['prcp(mm/day)'].rolling(window=window, min_periods=1).std()

            df[f'pet_roll_mean_{window}'] = df['pet(mm/day)'].rolling(window=window, min_periods=1).mean()
            df[f'pet_roll_sum_{window}'] = df['pet(mm/day)'].rolling(window=window, min_periods=1).sum()

            df[f'flow_roll_mean_{window}'] = df['flow(m^3/s)'].rolling(window=window, min_periods=1).mean()
            df[f'flow_roll_std_{window}'] = df['flow(m^3/s)'].rolling(window=window, min_periods=1).std()

            features.extend([
                f'prcp_roll_mean_{window}', f'prcp_roll_sum_{window}',
                f'prcp_roll_max_{window}', f'prcp_roll_std_{window}',
                f'pet_roll_mean_{window}', f'pet_roll_sum_{window}',
                f'flow_roll_mean_{window}', f'flow_roll_std_{window}'
            ])

        # Antecedent Precipitation Index
        df['api_7'] = df['prcp(mm/day)'].rolling(window=7, min_periods=1).apply(
            lambda x: np.sum(x * np.exp(-np.arange(len(x))[::-1] / 7))
        )
        df['api_30'] = df['prcp(mm/day)'].rolling(window=30, min_periods=1).apply(
            lambda x: np.sum(x * np.exp(-np.arange(len(x))[::-1] / 30))
        )

        # Water balance indicators
        df['prcp_pet_ratio'] = df['prcp(mm/day)'] / (df['pet(mm/day)'] + 0.001)
        df['cum_water_deficit'] = (df['prcp(mm/day)'] - df['pet(mm/day)']).cumsum()

        # Add all features
        features.extend([
            'month', 'day_of_year', 'season', 'week_of_year',
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'api_7', 'api_30', 'prcp_pet_ratio', 'cum_water_deficit'
        ])

        return features

    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics"""
        mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {metric: -999 for metric in ['RMSE', 'MAE', 'R2', 'NSE', 'RE', 'KGE']}

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Nash-Sutcliffe Efficiency
        nse = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

        # Relative Error
        re = (np.sum(y_pred) - np.sum(y_true)) / np.sum(y_true) * 100 if np.sum(y_true) > 0 else 0

        # Kling-Gupta Efficiency
        r = np.corrcoef(y_true, y_pred)[0, 1]
        alpha = np.std(y_pred) / np.std(y_true)
        beta = np.mean(y_pred) / np.mean(y_true)
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'NSE': nse,
            'RE': re,
            'KGE': kge
        }

    def optimize_cubist(self):
        """Optimize Cubist (or CatBoost if Cubist unavailable)"""
        if CUBIST_AVAILABLE:
            print("\n" + "=" * 60)
            print("Optimizing Cubist with Optuna...")
            print("=" * 60)

            def objective(trial):
                params = {
                    'n_rules': trial.suggest_int('n_rules',
                                                 *self.config.get('cubist', 'search_space', 'n_rules')),
                    'n_committees': trial.suggest_int('n_committees',
                                                      *self.config.get('cubist', 'search_space', 'n_committees')),
                    'neighbors': trial.suggest_int('neighbors',
                                                   *self.config.get('cubist', 'search_space', 'neighbors')),
                    'extrapolation': trial.suggest_float('extrapolation',
                                                         *self.config.get('cubist', 'search_space', 'extrapolation')),
                }

                params.update(self.config.get('cubist', 'fixed_params'))

                try:
                    model = Cubist(**params)

                    # Cross-validation with smaller fold for Cubist
                    cv_scores = cross_val_score(
                        model, self.X_train_scaled, self.y_train,
                        cv=self.config.get('cubist', 'optimization', 'cv_folds'),
                        scoring='neg_mean_squared_error',
                        n_jobs=1  # Cubist may not support parallel
                    )

                    return -cv_scores.mean()
                except:
                    return float('inf')

            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )

            study.optimize(
                objective,
                n_trials=self.config.get('cubist', 'optimization', 'n_trials'),
                timeout=self.config.get('cubist', 'optimization', 'timeout'),
                show_progress_bar=True
            )

            # Train best model
            best_params = study.best_params
            best_params.update(self.config.get('cubist', 'fixed_params'))

            self.models['Cubist'] = Cubist(**best_params)
            self.models['Cubist'].fit(self.X_train_scaled, self.y_train)

            self.best_params['Cubist'] = best_params
            self.studies['Cubist'] = study

            self._evaluate_model('Cubist')

        else:
            # Use CatBoost as alternative
            print("\n" + "=" * 60)
            print("Optimizing CatBoost (Cubist alternative) with Optuna...")
            print("=" * 60)

            def objective(trial):
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                    'random_strength': trial.suggest_float('random_strength', 0, 1),
                    'verbose': False
                }

                model = CatBoostRegressor(**params)

                cv_scores = cross_val_score(
                    model, self.X_train_scaled, self.y_train,
                    cv=3,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )

                return -cv_scores.mean()

            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )

            study.optimize(objective, n_trials=50, show_progress_bar=True)

            best_params = study.best_params
            best_params['verbose'] = False

            self.models['CatBoost'] = CatBoostRegressor(**best_params)
            self.models['CatBoost'].fit(self.X_train_scaled, self.y_train)

            self.best_params['CatBoost'] = best_params
            self.studies['CatBoost'] = study

            self._evaluate_model('CatBoost')

    def optimize_xgboost(self):
        """Optimize XGBoost using Optuna"""
        print("\n" + "=" * 60)
        print("Optimizing XGBoost with Optuna...")
        print("=" * 60)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators',
                                                  *self.config.get('xgboost', 'search_space', 'n_estimators')),
                'max_depth': trial.suggest_int('max_depth',
                                               *self.config.get('xgboost', 'search_space', 'max_depth')),
                'learning_rate': trial.suggest_float('learning_rate',
                                                     *self.config.get('xgboost', 'search_space', 'learning_rate'),
                                                     log=True),
                'subsample': trial.suggest_float('subsample',
                                                 *self.config.get('xgboost', 'search_space', 'subsample')),
                'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                        *self.config.get('xgboost', 'search_space',
                                                                         'colsample_bytree')),
                'gamma': trial.suggest_float('gamma',
                                             *self.config.get('xgboost', 'search_space', 'gamma')),
                'reg_alpha': trial.suggest_float('reg_alpha',
                                                 *self.config.get('xgboost', 'search_space', 'reg_alpha')),
                'reg_lambda': trial.suggest_float('reg_lambda',
                                                  *self.config.get('xgboost', 'search_space', 'reg_lambda')),
                'min_child_weight': trial.suggest_int('min_child_weight',
                                                      *self.config.get('xgboost', 'search_space', 'min_child_weight')),
            }

            params.update(self.config.get('xgboost', 'fixed_params'))

            model = xgb.XGBRegressor(**params)

            cv_scores = cross_val_score(
                model, self.X_train_scaled, self.y_train,
                cv=self.config.get('xgboost', 'optimization', 'cv_folds'),
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )

            return -cv_scores.mean()

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        study.optimize(
            objective,
            n_trials=self.config.get('xgboost', 'optimization', 'n_trials'),
            timeout=self.config.get('xgboost', 'optimization', 'timeout'),
            show_progress_bar=True
        )

        best_params = study.best_params
        best_params.update(self.config.get('xgboost', 'fixed_params'))

        self.models['XGBoost'] = xgb.XGBRegressor(**best_params)
        self.models['XGBoost'].fit(self.X_train_scaled, self.y_train)

        self.best_params['XGBoost'] = best_params
        self.studies['XGBoost'] = study

        self._evaluate_model('XGBoost')

    def optimize_lightgbm(self):
        """Optimize LightGBM using Optuna"""
        print("\n" + "=" * 60)
        print("Optimizing LightGBM with Optuna...")
        print("=" * 60)

        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves',
                                                *self.config.get('lightgbm', 'search_space', 'num_leaves')),
                'max_depth': trial.suggest_int('max_depth',
                                               *self.config.get('lightgbm', 'search_space', 'max_depth')),
                'learning_rate': trial.suggest_float('learning_rate',
                                                     *self.config.get('lightgbm', 'search_space', 'learning_rate'),
                                                     log=True),
                'n_estimators': trial.suggest_int('n_estimators',
                                                  *self.config.get('lightgbm', 'search_space', 'n_estimators')),
                'min_child_samples': trial.suggest_int('min_child_samples',
                                                       *self.config.get('lightgbm', 'search_space',
                                                                        'min_child_samples')),
                'subsample': trial.suggest_float('subsample',
                                                 *self.config.get('lightgbm', 'search_space', 'subsample')),
                'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                        *self.config.get('lightgbm', 'search_space',
                                                                         'colsample_bytree')),
                'reg_alpha': trial.suggest_float('reg_alpha',
                                                 *self.config.get('lightgbm', 'search_space', 'reg_alpha')),
                'reg_lambda': trial.suggest_float('reg_lambda',
                                                  *self.config.get('lightgbm', 'search_space', 'reg_lambda')),
            }

            params.update(self.config.get('lightgbm', 'fixed_params'))

            model = lgb.LGBMRegressor(**params)

            cv_scores = cross_val_score(
                model, self.X_train_scaled, self.y_train,
                cv=self.config.get('lightgbm', 'optimization', 'cv_folds'),
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )

            return -cv_scores.mean()

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        study.optimize(
            objective,
            n_trials=self.config.get('lightgbm', 'optimization', 'n_trials'),
            timeout=self.config.get('lightgbm', 'optimization', 'timeout'),
            show_progress_bar=True
        )

        best_params = study.best_params
        best_params.update(self.config.get('lightgbm', 'fixed_params'))

        self.models['LightGBM'] = lgb.LGBMRegressor(**best_params)
        self.models['LightGBM'].fit(self.X_train_scaled, self.y_train)

        self.best_params['LightGBM'] = best_params
        self.studies['LightGBM'] = study

        self._evaluate_model('LightGBM')

    def optimize_random_forest(self):
        """Optimize Random Forest using Optuna"""
        print("\n" + "=" * 60)
        print("Optimizing Random Forest with Optuna...")
        print("=" * 60)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators',
                                                  *self.config.get('random_forest', 'search_space', 'n_estimators')),
                'max_depth': trial.suggest_int('max_depth',
                                               *self.config.get('random_forest', 'search_space', 'max_depth')),
                'min_samples_split': trial.suggest_int('min_samples_split',
                                                       *self.config.get('random_forest', 'search_space',
                                                                        'min_samples_split')),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf',
                                                      *self.config.get('random_forest', 'search_space',
                                                                       'min_samples_leaf')),
                'max_features': trial.suggest_categorical('max_features',
                                                          self.config.get('random_forest', 'search_space',
                                                                          'max_features')),
                'bootstrap': trial.suggest_categorical('bootstrap',
                                                       self.config.get('random_forest', 'search_space', 'bootstrap')),
            }

            if params['bootstrap']:
                max_samples = self.config.get('random_forest', 'search_space', 'max_samples')
                params['max_samples'] = trial.suggest_categorical('max_samples', max_samples)

            params.update(self.config.get('random_forest', 'fixed_params'))

            model = RandomForestRegressor(**params)

            cv_scores = cross_val_score(
                model, self.X_train_scaled, self.y_train,
                cv=self.config.get('random_forest', 'optimization', 'cv_folds'),
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )

            return -cv_scores.mean()

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        study.optimize(
            objective,
            n_trials=self.config.get('random_forest', 'optimization', 'n_trials'),
            timeout=self.config.get('random_forest', 'optimization', 'timeout'),
            show_progress_bar=True
        )

        best_params = study.best_params
        best_params.update(self.config.get('random_forest', 'fixed_params'))

        self.models['RandomForest'] = RandomForestRegressor(**best_params)
        self.models['RandomForest'].fit(self.X_train_scaled, self.y_train)

        self.best_params['RandomForest'] = best_params
        self.studies['RandomForest'] = study

        self._evaluate_model('RandomForest')

    def optimize_svm(self):
        """Optimize SVM using Optuna"""
        print("\n" + "=" * 60)
        print("Optimizing SVM with Optuna...")
        print("=" * 60)

        def objective(trial):
            kernel = trial.suggest_categorical('kernel',
                                               self.config.get('svm', 'search_space', 'kernel'))

            params = {
                'kernel': kernel,
                'C': trial.suggest_float('C',
                                         *self.config.get('svm', 'search_space', 'C'), log=True),
                'epsilon': trial.suggest_float('epsilon',
                                               *self.config.get('svm', 'search_space', 'epsilon'), log=True),
            }

            if kernel in ['rbf', 'poly', 'sigmoid']:
                params['gamma'] = trial.suggest_float('gamma',
                                                      *self.config.get('svm', 'search_space', 'gamma'), log=True)

            if kernel == 'poly':
                params['degree'] = trial.suggest_int('degree',
                                                     *self.config.get('svm', 'search_space', 'degree'))
                params['coef0'] = trial.suggest_float('coef0',
                                                      *self.config.get('svm', 'search_space', 'coef0'))

            params.update(self.config.get('svm', 'fixed_params'))

            model = SVR(**params)

            # Use fewer samples for SVM
            sample_size = min(2000, len(self.X_train_scaled))
            indices = np.random.choice(len(self.X_train_scaled), sample_size, replace=False)

            cv_scores = cross_val_score(
                model,
                self.X_train_scaled[indices],
                self.y_train_scaled[indices],
                cv=self.config.get('svm', 'optimization', 'cv_folds'),
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )

            return -cv_scores.mean()

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        study.optimize(
            objective,
            n_trials=self.config.get('svm', 'optimization', 'n_trials'),
            timeout=self.config.get('svm', 'optimization', 'timeout'),
            show_progress_bar=True
        )

        best_params = study.best_params
        best_params.update(self.config.get('svm', 'fixed_params'))

        self.models['SVM'] = SVR(**best_params)
        self.models['SVM'].fit(self.X_train_scaled, self.y_train_scaled)

        self.best_params['SVM'] = best_params
        self.studies['SVM'] = study

        # Evaluate with inverse scaling
        y_pred_train = self.scaler_y.inverse_transform(
            self.models['SVM'].predict(self.X_train_scaled).reshape(-1, 1)
        ).ravel()
        y_pred_test = self.scaler_y.inverse_transform(
            self.models['SVM'].predict(self.X_test_scaled).reshape(-1, 1)
        ).ravel()

        train_metrics = self.calculate_metrics(self.y_train, y_pred_train)
        test_metrics = self.calculate_metrics(self.y_test, y_pred_test)

        self.results['SVM'] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test
        }

        print(f"Best params: {best_params}")
        print(f"Test NSE: {test_metrics['NSE']:.4f}, R2: {test_metrics['R2']:.4f}")

    def create_lstm_sequences(self, X, y):
        """Create sequences for LSTM"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def build_advanced_lstm_model(self, hp):
        """Build advanced LSTM model with multiple architectures"""
        architecture = hp.Choice('architecture',
                                 ['lstm_simple', 'lstm_bidirectional', 'lstm_attention',
                                  'gru_based', 'cnn_lstm', 'hybrid'])

        input_shape = (self.sequence_length, self.X_train_seq.shape[2])
        inputs = Input(shape=input_shape)

        if architecture == 'lstm_simple':
            # Simple stacked LSTM
            x = inputs
            n_layers = hp.Int('n_layers', 2, 4)

            for i in range(n_layers):
                units = hp.Int(f'units_layer_{i + 1}', 32, 256)
                dropout = hp.Float('dropout_rate', 0.0, 0.5)
                return_sequences = (i < n_layers - 1)

                x = LSTM(units, return_sequences=return_sequences,
                         dropout=dropout)(x)

        elif architecture == 'lstm_bidirectional':
            # Bidirectional LSTM
            x = inputs
            n_layers = hp.Int('n_layers', 2, 3)

            for i in range(n_layers):
                units = hp.Int(f'units_layer_{i + 1}', 32, 128)
                dropout = hp.Float('dropout_rate', 0.0, 0.5)
                return_sequences = (i < n_layers - 1)

                x = Bidirectional(LSTM(units, return_sequences=return_sequences,
                                       dropout=dropout))(x)

        elif architecture == 'lstm_attention' and ATTENTION_AVAILABLE:
            # LSTM with attention
            units1 = hp.Int('units_layer_1', 64, 256)
            units2 = hp.Int('units_layer_2', 32, 128)
            dropout = hp.Float('dropout_rate', 0.0, 0.5)

            lstm_out = LSTM(units1, return_sequences=True, dropout=dropout)(inputs)
            lstm_out = LSTM(units2, return_sequences=True, dropout=dropout)(lstm_out)

            attention_out = MultiHeadAttention(
                num_heads=hp.Int('attention_heads', 2, 8),
                key_dim=hp.Int('key_dim', 16, 64)
            )(lstm_out, lstm_out)

            attention_out = LayerNormalization()(attention_out + lstm_out)
            x = GlobalAveragePooling1D()(attention_out)

        elif architecture == 'gru_based':
            # GRU-based model
            x = inputs
            n_layers = hp.Int('n_layers', 2, 3)

            for i in range(n_layers):
                units = hp.Int(f'units_layer_{i + 1}', 32, 128)
                dropout = hp.Float('dropout_rate', 0.0, 0.5)
                return_sequences = (i < n_layers - 1)

                x = GRU(units, return_sequences=return_sequences,
                        dropout=dropout)(x)

        elif architecture == 'cnn_lstm':
            # CNN-LSTM hybrid
            filters = hp.Int('conv_filters', 32, 128)
            kernel_size = hp.Int('kernel_size', 3, 7)

            x = Conv1D(filters=filters, kernel_size=kernel_size,
                       activation='relu', padding='same')(inputs)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(hp.Float('conv_dropout', 0.0, 0.3))(x)

            lstm_units = hp.Int('lstm_units', 32, 128)
            x = LSTM(lstm_units, dropout=hp.Float('lstm_dropout', 0.0, 0.5))(x)

        else:  # hybrid
            # Complex hybrid architecture
            # CNN branch
            cnn_branch = Conv1D(filters=hp.Int('conv_filters', 32, 64),
                                kernel_size=hp.Int('kernel_size', 3, 5),
                                activation='relu', padding='same')(inputs)
            cnn_branch = GlobalMaxPooling1D()(cnn_branch)

            # LSTM branch
            lstm_branch = LSTM(hp.Int('lstm_units', 64, 128),
                               dropout=hp.Float('dropout_rate', 0.0, 0.5))(inputs)

            # GRU branch
            gru_branch = GRU(hp.Int('gru_units', 32, 64),
                             dropout=hp.Float('dropout_rate', 0.0, 0.5))(inputs)

            # Concatenate branches
            x = concatenate([cnn_branch, lstm_branch, gru_branch])

        # Dense layers
        x = Dense(hp.Int('dense_units', 32, 128),
                  activation=hp.Choice('activation', ['relu', 'tanh']))(x)
        x = Dropout(hp.Float('dense_dropout', 0.0, 0.5))(x)

        # Output layer
        outputs = Dense(1)(x)

        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=Adam(hp.Float('learning_rate', 0.0001, 0.01, sampling='log')),
            loss='mse',
            metrics=['mae']
        )

        return model

    def optimize_lstm(self):
        """Optimize advanced LSTM architectures using Keras Tuner"""
        print("\n" + "=" * 60)
        print("Optimizing Advanced LSTM Architectures with Keras Tuner...")
        print("=" * 60)

        # Prepare sequences
        self.X_train_seq, self.y_train_seq = self.create_lstm_sequences(
            self.X_train_scaled, self.y_train_scaled
        )
        self.X_test_seq, self.y_test_seq = self.create_lstm_sequences(
            self.X_test_scaled, self.y_test_scaled
        )

        print(f"Sequence shape: {self.X_train_seq.shape}")

        # Create tuner
        tuner = kt.RandomSearch(
            self.build_advanced_lstm_model,
            objective='val_loss',
            max_trials=self.config.get('lstm', 'optimization', 'max_trials'),
            executions_per_trial=self.config.get('lstm', 'optimization', 'executions_per_trial'),
            directory=self.config.get('lstm', 'optimization', 'directory'),
            project_name=self.config.get('lstm', 'optimization', 'project_name'),
            overwrite=self.config.get('lstm', 'optimization', 'overwrite')
        )

        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.config.get('lstm', 'fixed_params', 'patience'),
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.get('lstm', 'fixed_params', 'reduce_lr_factor'),
            patience=self.config.get('lstm', 'fixed_params', 'reduce_lr_patience'),
            min_lr=self.config.get('lstm', 'fixed_params', 'min_lr')
        )

        # Search
        tuner.search(
            self.X_train_seq, self.y_train_seq,
            epochs=self.config.get('lstm', 'fixed_params', 'epochs'),
            validation_split=self.config.get('lstm', 'fixed_params', 'validation_split'),
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # Get best model
        best_model = tuner.get_best_models()[0]
        best_params = tuner.get_best_hyperparameters()[0].values

        self.models['LSTM'] = best_model
        self.best_params['LSTM'] = best_params
        self.tuners['LSTM'] = tuner

        # Evaluate
        y_pred_train_scaled = best_model.predict(self.X_train_seq, verbose=0).ravel()
        y_pred_test_scaled = best_model.predict(self.X_test_seq, verbose=0).ravel()

        y_pred_train = self.scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
        y_pred_test = self.scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()

        y_train_lstm = self.scaler_y.inverse_transform(self.y_train_seq.reshape(-1, 1)).ravel()
        y_test_lstm = self.scaler_y.inverse_transform(self.y_test_seq.reshape(-1, 1)).ravel()

        train_metrics = self.calculate_metrics(y_train_lstm, y_pred_train)
        test_metrics = self.calculate_metrics(y_test_lstm, y_pred_test)

        self.results['LSTM'] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'y_train_seq': y_train_lstm,
            'y_test_seq': y_test_lstm,
            'architecture': best_params.get('architecture', 'unknown')
        }

        print(f"Best architecture: {best_params.get('architecture', 'unknown')}")
        print(f"Best params: {best_params}")
        print(f"Test NSE: {test_metrics['NSE']:.4f}, R2: {test_metrics['R2']:.4f}")

    def _evaluate_model(self, model_name):
        """Evaluate a trained model"""
        y_pred_train = self.models[model_name].predict(self.X_train_scaled)
        y_pred_test = self.models[model_name].predict(self.X_test_scaled)

        train_metrics = self.calculate_metrics(self.y_train, y_pred_train)
        test_metrics = self.calculate_metrics(self.y_test, y_pred_test)

        self.results[model_name] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test
        }

        if hasattr(self.models[model_name], 'feature_importances_'):
            self.results[model_name]['feature_importance'] = self.models[model_name].feature_importances_

        print(f"Best params: {self.best_params[model_name]}")
        print(f"Test NSE: {test_metrics['NSE']:.4f}, R2: {test_metrics['R2']:.4f}")

    def optimize_all_models(self):
        """Run optimization for all models"""
        self.optimize_xgboost()
        self.optimize_lightgbm()
        self.optimize_random_forest()
        self.optimize_cubist()  # Now included!
        self.optimize_svm()
        self.optimize_lstm()

    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "=" * 80)
        print("COMPLETE ML PIPELINE - PERFORMANCE SUMMARY")
        print("=" * 80)

        # Check if we have any results
        if not self.results:
            print("\n⚠️ No models have been evaluated yet.")
            print("Please train at least one model before printing summary.")
            return

        # Prepare summary data
        summary_data = []
        for model_name in self.results:
            # Check if the model has the required metrics
            if 'train_metrics' not in self.results[model_name] or 'test_metrics' not in self.results[model_name]:
                print(f"⚠️ Skipping {model_name} - incomplete metrics")
                continue

            try:
                summary_data.append([
                    model_name,
                    self.results[model_name]['train_metrics']['NSE'],
                    self.results[model_name]['test_metrics']['NSE'],
                    self.results[model_name]['train_metrics']['R2'],
                    self.results[model_name]['test_metrics']['R2'],
                    self.results[model_name]['train_metrics']['KGE'],
                    self.results[model_name]['test_metrics']['KGE'],
                    self.results[model_name]['test_metrics']['RMSE']
                ])
            except KeyError as e:
                print(f"⚠️ Skipping {model_name} - missing metric: {e}")
                continue

        # Check if we have any valid summary data
        if not summary_data:
            print("\n⚠️ No complete model results available for summary.")
            print("Models may have been trained but not properly evaluated.")

            # Show what models exist
            if self.models:
                print(f"\nModels loaded: {list(self.models.keys())}")
            if self.best_params:
                print(f"Parameters available for: {list(self.best_params.keys())}")

            return

        # Sort by test NSE
        summary_data.sort(key=lambda x: x[2], reverse=True)

        # Print table
        print("\n{:<12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "Model", "Train NSE", "Test NSE", "Train R2", "Test R2", "Train KGE", "Test KGE", "Test RMSE"
        ))
        print("-" * 95)

        for row in summary_data:
            print("{:<12} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(*row))

        # Best model
        best_model = summary_data[0][0]
        print(f"\n🏆 Best Model (by Test NSE): {best_model}")

        if best_model == 'LSTM':
            print(f"   Architecture: {self.results['LSTM'].get('architecture', 'unknown')}")

        # Print best hyperparameters if available
        if best_model in self.best_params:
            print("\nBest Hyperparameters:")
            for key, value in self.best_params[best_model].items():
                # Format the value based on its type
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"\n⚠️ Hyperparameters for {best_model} not available")

        print("\n" + "=" * 80)


# Main execution
if __name__ == "__main__":
    config_file = 'hyperparameter_config.json'

    if not os.path.exists(config_file):
        print(f"Configuration file {config_file} not found!")
        print("Please ensure hyperparameter_config.json exists with proper configuration!")
        exit(1)

    # Initialize pipeline
    print("=" * 80)
    print("COMPLETE ML PIPELINE WITH ALL MODELS AND ADVANCED ARCHITECTURES")
    print("=" * 80)

    pipeline = AdvancedHydrologicalMLPipeline(config_path=config_file)

    # Load and prepare data
    pipeline.load_and_prepare_data()

    # Run optimization for all models
    print("\nStarting hyperparameter optimization for all models...")
    print("This will optimize: XGBoost, LightGBM, RandomForest, Cubist/CatBoost, SVM, and Advanced LSTM")

    pipeline.optimize_all_models()

    # Print summary
    pipeline.print_summary()

    print("\n✅ OPTIMIZATION COMPLETE!")
    print("All models including Cubist and advanced LSTM architectures have been optimized!")