"""Enhanced batch trainer with multiple ML models and hyperparameter optimization.

Features:
- Support for multiple ML models: Random Forest, Gradient Boosting, Neural Networks
- Hyperparameter tuning with Optuna (fallback to GridSearchCV)
- Cross-validation with time series splits
- Model comparison and ensemble methods
- Feature importance analysis
- Comprehensive model evaluation metrics
- Model persistence and versioning
"""
import pandas as pd
import numpy as np
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Core ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib
    sklearn_available = True
except ImportError:
    sklearn_available = False
    logger.warning("scikit-learn not available, using minimal implementation")

# Optuna for hyperparameter optimization
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce optuna logging
    optuna_available = True
except ImportError:
    optuna_available = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[1] / 'model_store'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    """Configuration for ML models and hyperparameters."""
    name: str
    model_class: Any
    param_grid: Dict
    needs_scaling: bool = False
    default_params: Optional[Dict] = None

# Model configurations
MODEL_CONFIGS = {
    'random_forest': ModelConfig(
        name='Random Forest',
        model_class=RandomForestClassifier if sklearn_available else None,
        param_grid={
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'random_state': [42]
        },
        default_params={'n_estimators': 100, 'random_state': 42}
    ),
    'gradient_boosting': ModelConfig(
        name='Gradient Boosting',
        model_class=GradientBoostingClassifier if sklearn_available else None,
        param_grid={
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'random_state': [42]
        },
        default_params={'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
    ),
    'neural_network': ModelConfig(
        name='Neural Network (MLP)',
        model_class=MLPClassifier if sklearn_available else None,
        param_grid={
            'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [500],
            'random_state': [42]
        },
        needs_scaling=True,
        default_params={'hidden_layer_sizes': (100,), 'max_iter': 500, 'random_state': 42}
    ),
    'logistic_regression': ModelConfig(
        name='Logistic Regression',
        model_class=LogisticRegression if sklearn_available else None,
        param_grid={
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000],
            'random_state': [42]
        },
        needs_scaling=True,
        default_params={'max_iter': 1000, 'random_state': 42}
    )
}

def create_features(df: pd.DataFrame, lookback_periods: List[int] = [5, 10, 20]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create comprehensive feature set from OHLCV data.
    
    Args:
        df: OHLCV DataFrame
        lookback_periods: Periods for rolling window features
        
    Returns:
        Tuple of (features_df, target_series)
    """
    logger.info(f"Creating features from {len(df)} bars with lookback periods: {lookback_periods}")
    
    df = df.copy()
    
    # Basic price features
    df['ret'] = df['close'].pct_change()
    df['vol_change'] = df['volume'].pct_change() if 'volume' in df.columns else 0
    
    # Rolling window features
    for period in lookback_periods:
        df[f'ma_{period}'] = df['close'].rolling(period).mean()
        df[f'std_{period}'] = df['close'].rolling(period).std()
        df[f'ret_mean_{period}'] = df['ret'].rolling(period).mean()
        df[f'ret_std_{period}'] = df['ret'].rolling(period).std()
        
        if 'volume' in df.columns:
            df[f'vol_mean_{period}'] = df['volume'].rolling(period).mean()
    
    # Technical indicators
    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down.replace(0, 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    df['bb_middle'] = df['close'].rolling(bb_period).mean()
    bb_std_val = df['close'].rolling(bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
    df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # MACD
    ema_fast = df['close'].ewm(span=12).mean()
    ema_slow = df['close'].ewm(span=26).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Price position features
    for period in [10, 20, 50]:
        df[f'high_{period}'] = df['high'].rolling(period).max()
        df[f'low_{period}'] = df['low'].rolling(period).min()
        df[f'price_position_{period}'] = (df['close'] - df[f'low_{period}']) / (df[f'high_{period}'] - df[f'low_{period}'])
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    if len(df) == 0:
        raise ValueError("No data remaining after feature creation")
    
    # Select feature columns (exclude OHLCV and intermediate calculations)
    feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    
    # Create target: next bar direction (1 for up, 0 for down)
    target = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Remove last row (no target available)
    X = df[feature_cols].iloc[:-1]
    y = target.iloc[:-1]
    
    logger.info(f"Created {len(feature_cols)} features for {len(X)} samples")
    logger.info(f"Feature names: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

class OptimizedModelTrainer:
    """Enhanced model trainer with hyperparameter optimization and model comparison."""
    
    def __init__(self, use_optuna: bool = True, n_trials: int = 50, cv_folds: int = 5):
        self.use_optuna = use_optuna and optuna_available
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.results = {}
        self.best_models = {}
        
    def optimize_hyperparameters_optuna(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optimize hyperparameters using Optuna."""
        config = MODEL_CONFIGS[model_name]
        
        def objective(trial):
            # Suggest hyperparameters
            params = {}
            param_grid = config.param_grid
            
            for param, values in param_grid.items():
                if param == 'random_state':
                    params[param] = 42
                elif isinstance(values[0], int):
                    if None in values:
                        params[param] = trial.suggest_categorical(param, values)
                    else:
                        params[param] = trial.suggest_int(param, min(values), max(values))
                elif isinstance(values[0], float):
                    params[param] = trial.suggest_float(param, min(values), max(values))
                else:
                    params[param] = trial.suggest_categorical(param, values)
            
            # Create and evaluate model
            model = config.model_class(**params)
            
            if config.needs_scaling:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
            else:
                pipeline = model
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
            
            return scores.mean()
        
        logger.info(f"Starting Optuna optimization for {config.name} with {self.n_trials} trials")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        logger.info(f"Best score for {config.name}: {study.best_value:.4f}")
        logger.info(f"Best params for {config.name}: {study.best_params}")
        
        return study.best_params
    
    def optimize_hyperparameters_grid(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optimize hyperparameters using GridSearchCV."""
        config = MODEL_CONFIGS[model_name]
        
        logger.info(f"Starting GridSearchCV for {config.name}")
        
        model = config.model_class()
        
        if config.needs_scaling:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            # Adjust parameter names for pipeline
            param_grid = {f'model__{k}': v for k, v in config.param_grid.items()}
        else:
            pipeline = model
            param_grid = config.param_grid
        
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=tscv, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best score for {config.name}: {grid_search.best_score_:.4f}")
        
        # Extract model parameters from pipeline if needed
        if config.needs_scaling:
            best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
        else:
            best_params = grid_search.best_params_
        
        logger.info(f"Best params for {config.name}: {best_params}")
        
        return best_params
    
    def train_single_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, 
                          optimize_hyperparams: bool = True) -> Dict:
        """Train a single model with optional hyperparameter optimization."""
        config = MODEL_CONFIGS[model_name]
        
        if not config.model_class:
            logger.warning(f"Model {model_name} not available, skipping")
            return {'error': 'Model not available'}
        
        start_time = time.time()
        logger.info(f"Training {config.name}...")
        
        try:
            # Optimize hyperparameters
            if optimize_hyperparams:
                if self.use_optuna:
                    best_params = self.optimize_hyperparameters_optuna(model_name, X, y)
                else:
                    best_params = self.optimize_hyperparameters_grid(model_name, X, y)
            else:
                best_params = config.default_params or {}
            
            # Train final model with best parameters
            model = config.model_class(**best_params)
            
            if config.needs_scaling:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model.fit(X_scaled, y)
                
                # Create pipeline for saving
                final_model = Pipeline([
                    ('scaler', scaler),
                    ('model', model)
                ])
            else:
                model.fit(X, y)
                final_model = model
            
            # Evaluate model with cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            cv_scores = cross_val_score(final_model, X, y, cv=tscv, scoring='accuracy')
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
                # Sort by importance
                feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            # Save model
            model_filename = f'{model_name}_optimized.joblib'
            model_path = MODEL_DIR / model_filename
            joblib.dump(final_model, model_path)
            
            training_time = time.time() - start_time
            
            result = {
                'model_name': config.name,
                'best_params': best_params,
                'cv_mean_score': cv_scores.mean(),
                'cv_std_score': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'feature_importance': feature_importance,
                'model_path': str(model_path),
                'training_time_seconds': training_time,
                'model_object': final_model
            }
            
            logger.info(f"{config.name} training completed in {training_time:.1f}s")
            logger.info(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training {config.name}: {str(e)}")
            return {'error': str(e), 'model_name': config.name}
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, 
                        optimize_hyperparams: bool = True) -> Dict:
        """Train all available models and compare performance."""
        logger.info(f"Training {len(MODEL_CONFIGS)} models...")
        
        results = {}
        
        for model_name in MODEL_CONFIGS.keys():
            result = self.train_single_model(model_name, X, y, optimize_hyperparams)
            results[model_name] = result
        
        # Find best model
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if valid_results:
            best_model_name = max(valid_results.keys(), 
                                key=lambda k: valid_results[k]['cv_mean_score'])
            best_model_info = valid_results[best_model_name]
            
            logger.info("="*60)
            logger.info("MODEL COMPARISON RESULTS")
            logger.info("="*60)
            
            for model_name, result in results.items():
                if 'error' not in result:
                    score = result['cv_mean_score']
                    std = result['cv_std_score']
                    time_taken = result['training_time_seconds']
                    marker = " [BEST]" if model_name == best_model_name else ""
                    logger.info(f"{result['model_name']}: {score:.4f} (+/- {std*2:.4f}) - {time_taken:.1f}s{marker}")
                else:
                    logger.info(f"{model_name}: ERROR - {result['error']}")
            
            logger.info("="*60)
            logger.info(f"Best model: {best_model_info['model_name']}")
            logger.info(f"Best score: {best_model_info['cv_mean_score']:.4f}")
            logger.info("="*60)
            
            # Save best model info
            best_model_summary = {
                'best_model': best_model_name,
                'best_score': best_model_info['cv_mean_score'],
                'all_results': {k: {kk: vv for kk, vv in v.items() if kk != 'model_object'} 
                               for k, v in results.items()},
                'training_timestamp': time.time(),
                'feature_columns': list(X.columns),
                'dataset_size': len(X)
            }
            
            summary_path = MODEL_DIR / 'model_comparison_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(best_model_summary, f, indent=2, default=str)
            
            logger.info(f"Model comparison summary saved to {summary_path}")
        
        return results

def train_and_evaluate_models(bars_df: pd.DataFrame, optimize_hyperparams: bool = True,
                             use_optuna: bool = True, n_trials: int = 50) -> Dict:
    """
    Main function to train and evaluate multiple ML models.
    
    Args:
        bars_df: OHLCV DataFrame
        optimize_hyperparams: Whether to optimize hyperparameters
        use_optuna: Whether to use Optuna (vs GridSearchCV) for optimization
        n_trials: Number of trials for Optuna optimization
        
    Returns:
        Dictionary with training results and best model info
    """
    if not sklearn_available:
        logger.error("scikit-learn not available, cannot train models")
        return {'error': 'scikit-learn not available'}
    
    start_time = time.time()
    logger.info("Starting comprehensive model training and evaluation")
    logger.info(f"Dataset: {len(bars_df)} samples")
    logger.info(f"Hyperparameter optimization: {optimize_hyperparams}")
    logger.info(f"Optimization method: {'Optuna' if use_optuna else 'GridSearchCV'}")
    
    try:
        # Create features
        X, y = create_features(bars_df)
        
        if len(X) == 0:
            raise ValueError("No features created from data")
        
        # Initialize trainer
        trainer = OptimizedModelTrainer(
            use_optuna=use_optuna,
            n_trials=n_trials,
            cv_folds=5
        )
        
        # Train all models
        results = trainer.train_all_models(X, y, optimize_hyperparams)
        
        total_time = time.time() - start_time
        logger.info(f"Model training completed in {total_time:.1f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return {'error': str(e)}

# Legacy function for backward compatibility
def train_and_evaluate(bars_df: pd.DataFrame) -> Dict:
    """Legacy function for backward compatibility."""
    return train_and_evaluate_models(bars_df, optimize_hyperparams=False)
