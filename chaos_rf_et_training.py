#!/usr/bin/env python3
"""
CHAOS V1.0 - Local RF/ET Training Script
=========================================
GOOGLE DRIVE SYNCED VERSION - ALIGNED WITH COLAB

Path: G:\My Drive\chaos_v1.0\ (Windows) = /content/drive/MyDrive/chaos_v1.0/ (Colab)
Run with: python chaos_rf_et_training.py
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.utils.class_weight import compute_class_weight
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# CONFIGURATION - EXPLICIT, NO DYNAMIC SELECTION
# =============================================================================

# PATHS - Google Drive synced folder
BASE_DIR = Path(r"G:\My Drive\chaos_v1.0")
FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "rf_et_checkpoint.json"
PARAMS_DIR = MODELS_DIR / "optuna_params"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_DIR.mkdir(parents=True, exist_ok=True)

# EXPLICIT COLUMN NAMES - VERIFIED FROM DATA
TARGET_COL = 'target_3class_8'      # 3-class labels for 8-bar horizon
RETURNS_COL = 'target_return_8'     # Returns for 8-bar horizon

# FEATURE FILE PATTERN - _features.parquet files
FEATURE_FILE_PATTERN = "{pair}_{tf}_features.parquet"

# RANDOM SEED - MUST MATCH COLAB
RANDOM_SEED = 42

# Grid configuration - ALL 63 COMBINATIONS
ALL_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
             'USDCHF', 'NZDUSD', 'EURJPY', 'GBPJPY']  # 9 pairs

ALL_TIMEFRAMES = ['M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']  # 8 timeframes (M1 too large for 16GB RAM)

# Brain names
BRAINS = ['rf_optuna', 'et_optuna']

# =============================================================================
# OPTUNA CONFIGURATION BY TIMEFRAME
# =============================================================================
# More trials for smaller datasets (faster training)
# Fewer trials for larger datasets (time constraint)
# Early stopping patience to avoid wasting time when converged

TRIALS_CONFIG = {
    'M5':  {'trials': 50,  'patience': 20},   # ~1.1M rows - balance time vs quality
    'M15': {'trials': 75,  'patience': 25},   # ~375K rows - can afford more
    'M30': {'trials': 75,  'patience': 25},   # ~187K rows
    'H1':  {'trials': 100, 'patience': 30},   # ~93K rows - fast, maximize optimization
    'H4':  {'trials': 100, 'patience': 30},   # ~23K rows - very fast
    'D1':  {'trials': 100, 'patience': 30},   # ~3.9K rows - very fast
    'W1':  {'trials': 100, 'patience': 30},   # ~780 rows - instant
    'MN1': {'trials': 100, 'patience': 30},   # ~180 rows - instant
}


def get_optuna_config(timeframe: str) -> dict:
    """Get trials and patience configuration for a timeframe."""
    return TRIALS_CONFIG.get(timeframe, {'trials': 50, 'patience': 20})


# Trade frequency penalty - models must trade at least this % of the time
MIN_TRADE_RATIO = 0.10  # 10% minimum trades (non-FLAT predictions)

# =============================================================================
# GAUSSIAN SLIPPAGE MODEL
# =============================================================================
@dataclass
class SlippageConfig:
    """Per-pair slippage configuration."""
    pair: str
    mean_pips: float
    std_pips: float
    pip_value: float

    def sample(self, n_samples: int = 1) -> np.ndarray:
        np.random.seed(None)
        return np.random.normal(self.mean_pips, self.std_pips, n_samples)


SLIPPAGE_CONFIG = {
    'EURUSD': SlippageConfig('EURUSD', 0.25, 0.15, 0.0001),
    'GBPUSD': SlippageConfig('GBPUSD', 0.40, 0.25, 0.0001),
    'USDJPY': SlippageConfig('USDJPY', 0.30, 0.20, 0.01),
    'USDCHF': SlippageConfig('USDCHF', 0.35, 0.22, 0.0001),
    'USDCAD': SlippageConfig('USDCAD', 0.35, 0.22, 0.0001),
    'AUDUSD': SlippageConfig('AUDUSD', 0.40, 0.25, 0.0001),
    'NZDUSD': SlippageConfig('NZDUSD', 0.45, 0.28, 0.0001),
    'EURJPY': SlippageConfig('EURJPY', 0.45, 0.30, 0.01),
    'GBPJPY': SlippageConfig('GBPJPY', 0.50, 0.35, 0.01),
}


class GaussianSlippageModel:
    def __init__(self, configs: Dict[str, SlippageConfig] = None):
        self.configs = configs or SLIPPAGE_CONFIG.copy()

    def get_config(self, pair: str) -> SlippageConfig:
        return self.configs.get(pair, self.configs['EURUSD'])

    def apply_slippage(self, returns: np.ndarray, positions: np.ndarray,
                       pair: str) -> np.ndarray:
        n = len(returns)
        config = self.get_config(pair)
        slippage_pips = config.sample(n)
        if len(positions) != n:
            positions = positions[:n] if len(positions) > n else np.pad(positions, (0, n - len(positions)))
        position_changes = np.diff(np.concatenate([[0], positions]))
        slippage_impact = slippage_pips * config.pip_value * np.abs(position_changes)
        adjusted_returns = returns - slippage_impact
        return adjusted_returns


SLIPPAGE_MODEL = GaussianSlippageModel()

# =============================================================================
# CHECKPOINT MANAGEMENT - LIST FORMAT
# =============================================================================
def load_checkpoint() -> dict:
    """Load checkpoint or create new one."""
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("WARNING: Checkpoint corrupted, starting fresh")
            return {"completed": []}
    return {"completed": []}


def save_checkpoint(checkpoint: dict):
    """Save checkpoint."""
    checkpoint['timestamp'] = datetime.now().isoformat()
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def add_to_checkpoint(checkpoint: dict, model_key: str) -> dict:
    """Add model to checkpoint - LIST FORMAT (append), NOT DICT."""
    if model_key not in checkpoint['completed']:
        checkpoint['completed'].append(model_key)
    save_checkpoint(checkpoint)
    return checkpoint


# =============================================================================
# EXECUTION LAG - PREVENTS CIRCULAR DEPENDENCY
# =============================================================================
# Without lag: target predicts return T→T+8, evaluated on return T→T+8 = 100% profit
# With lag: target predicts return T→T+8, evaluated on return T+1→T+9 = realistic
EXECUTION_LAG_BARS = 1  # 1-bar execution delay (realistic)


# =============================================================================
# FEATURE LOADING - WITH EXECUTION LAG
# =============================================================================
def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature columns - exclude all targets, returns, metadata."""
    # Patterns to exclude
    exclude_patterns = ['target_', 'return', 'Open', 'High', 'Low', 'Close',
                        'Volume', 'timestamp', 'date', 'pair', 'symbol', 'tf',
                        'Bid_', 'Ask_', 'Spread_']

    feature_cols = []
    for col in df.columns:
        # Skip if matches exclude pattern
        if any(p in col for p in exclude_patterns):
            continue
        # Skip non-numeric
        if df[col].dtype not in ['int64', 'int32', 'float64', 'float32', 'Float64']:
            continue
        feature_cols.append(col)

    return feature_cols


def load_features(pair: str, tf: str, apply_lag: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load feature file with EXPLICIT column names and EXECUTION LAG.

    Execution lag breaks the circular dependency:
    - Target is defined as sign(return T→T+8)
    - Without lag: perfect prediction = 100% profit (unrealistic)
    - With lag: evaluate on return T+1→T+9 (realistic execution delay)
    """

    # Build filepath using pattern
    filename = FEATURE_FILE_PATTERN.format(pair=pair, tf=tf)
    filepath = FEATURES_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Feature file not found: {filepath}")

    df = pd.read_parquet(filepath)
    print(f"  Loaded {pair}_{tf}: {len(df):,} rows, {len(df.columns)} columns")

    # VALIDATE EXPLICIT COLUMNS EXIST
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found! Available: {[c for c in df.columns if 'target' in c.lower()]}")
    if RETURNS_COL not in df.columns:
        raise ValueError(f"Returns column '{RETURNS_COL}' not found! Available: {[c for c in df.columns if 'return' in c.lower()]}")

    # Get feature columns
    feature_cols = get_feature_columns(df)

    # APPLY EXECUTION LAG TO RETURNS
    if apply_lag and EXECUTION_LAG_BARS > 0:
        df['returns_lagged'] = df[RETURNS_COL].shift(-EXECUTION_LAG_BARS)
        df = df.dropna(subset=['returns_lagged'])
        returns_col_to_use = 'returns_lagged'
        print(f"  Applied {EXECUTION_LAG_BARS}-bar execution lag (breaks circular dependency)")
    else:
        returns_col_to_use = RETURNS_COL

    print(f"  Features: {len(feature_cols)}")

    # Prepare X
    X = df[feature_cols].values.astype('float32')
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # EXPLICIT target column
    y = df[TARGET_COL].values.copy()
    y = np.nan_to_num(y, nan=0.0)

    # Convert -1,0,1 to 0,1,2 if needed
    if y.min() < 0:
        y = y + 1
    y = y.astype('int64')

    # Returns with execution lag
    returns = df[returns_col_to_use].values.copy()
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Target: {TARGET_COL}, Returns: {returns_col_to_use}")

    # 80/20 chronological split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    returns_val = returns[split_idx:]

    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}")

    return X_train, y_train, X_val, y_val, returns_val, len(feature_cols)


# =============================================================================
# POSITION CONVERSION - CORRECT VERSION
# =============================================================================
def convert_to_positions(predictions: np.ndarray) -> np.ndarray:
    """
    Convert model predictions to trading positions.

    Input:  [0, 1, 2] where 0=short signal, 1=flat signal, 2=long signal
    Output: [-1, 0, +1] where -1=short position, 0=flat, +1=long position

    Math: position = prediction - 1
    """
    predictions = np.asarray(predictions).flatten()

    if len(predictions) == 0:
        return np.array([], dtype=np.float64)

    # Validate input range
    if predictions.min() < 0 or predictions.max() > 2:
        raise ValueError(
            f"Predictions must be in range [0, 1, 2]. "
            f"Got range [{predictions.min()}, {predictions.max()}]"
        )

    # Simple, correct conversion: 0→-1, 1→0, 2→+1
    positions = predictions.astype(np.float64) - 1.0

    return positions


# =============================================================================
# PROFIT FACTOR - WITH TRADE FREQUENCY PENALTY
# =============================================================================
def calculate_profit_factor(predictions: np.ndarray, returns: np.ndarray) -> float:
    """
    Calculate profit factor from predictions and forward returns.

    Profit Factor = Gross Profit / Gross Loss

    INCLUDES TRADE FREQUENCY PENALTY:
    - If trade_ratio < MIN_TRADE_RATIO, penalize the PF
    - This prevents models from only predicting FLAT to game the metric

    NO CLAMPING. Show the real value so bugs are visible.
    """
    predictions = np.asarray(predictions).flatten()
    returns = np.asarray(returns).flatten()

    if len(predictions) != len(returns):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, returns={len(returns)}"
        )

    if len(predictions) == 0:
        return 1.0

    # Convert predictions to positions
    positions = convert_to_positions(predictions)

    # Calculate trade ratio (non-FLAT predictions)
    n_trades = np.sum(positions != 0)
    trade_ratio = n_trades / len(positions)

    # PENALTY: If trade_ratio < MIN_TRADE_RATIO, heavily penalize
    if trade_ratio < MIN_TRADE_RATIO:
        # Return a value that discourages this behavior
        # Scale penalty based on how far below threshold
        penalty = trade_ratio / MIN_TRADE_RATIO  # 0 to 1
        return 0.5 * penalty  # Returns 0-0.5 for low trade models

    # Calculate PnL: position * return
    pnl = positions * returns

    # Gross profit = sum of winning trades
    gross_profit = np.sum(pnl[pnl > 0])

    # Gross loss = absolute value of sum of losing trades
    gross_loss = np.abs(np.sum(pnl[pnl < 0]))

    # Calculate PF
    if gross_loss < 1e-10:
        if gross_profit < 1e-10:
            return 1.0
        else:
            return 99.99  # Flag as suspicious

    pf = gross_profit / gross_loss

    return pf


def calculate_profit_factor_detailed(predictions: np.ndarray, returns: np.ndarray) -> dict:
    """Detailed PF calculation with all intermediate values for debugging."""
    predictions = np.asarray(predictions).flatten()
    returns = np.asarray(returns).flatten()

    positions = convert_to_positions(predictions)
    pnl = positions * returns

    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = np.abs(np.sum(pnl[pnl < 0]))

    if gross_loss < 1e-10:
        pf = 99.99 if gross_profit > 1e-10 else 1.0
    else:
        pf = gross_profit / gross_loss

    return {
        'pf': pf,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'total_pnl': pnl.sum(),
        'n_long': int(np.sum(positions == 1)),
        'n_short': int(np.sum(positions == -1)),
        'n_flat': int(np.sum(positions == 0)),
        'n_winning': int(np.sum(pnl > 0)),
        'n_losing': int(np.sum(pnl < 0)),
        'n_breakeven': int(np.sum(pnl == 0)),
    }


def calculate_profit_factor_with_stats(predictions: np.ndarray, returns: np.ndarray) -> Tuple[float, dict]:
    """
    Calculate PF with trade statistics for logging.

    Returns:
        (pf, stats_dict) where stats_dict contains trade ratio and counts
    """
    predictions = np.asarray(predictions).flatten()
    returns = np.asarray(returns).flatten()

    if len(predictions) == 0:
        return 1.0, {'trade_ratio': 0.0, 'n_trades': 0, 'n_total': 0, 'penalized': False}

    positions = convert_to_positions(predictions)

    n_trades = int(np.sum(positions != 0))
    n_total = len(positions)
    trade_ratio = n_trades / n_total

    # Check if penalized
    penalized = trade_ratio < MIN_TRADE_RATIO

    # Get raw PF (without penalty for logging)
    pnl = positions * returns
    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = np.abs(np.sum(pnl[pnl < 0]))

    if gross_loss < 1e-10:
        raw_pf = 99.99 if gross_profit > 1e-10 else 1.0
    else:
        raw_pf = gross_profit / gross_loss

    # Get actual PF (with penalty)
    actual_pf = calculate_profit_factor(predictions, returns)

    stats = {
        'trade_ratio': trade_ratio,
        'n_trades': n_trades,
        'n_total': n_total,
        'n_long': int(np.sum(positions == 1)),
        'n_short': int(np.sum(positions == -1)),
        'n_flat': int(np.sum(positions == 0)),
        'raw_pf': raw_pf,
        'penalized': penalized,
    }

    return actual_pf, stats


# =============================================================================
# EARLY STOPPING CALLBACK
# =============================================================================
class EarlyStoppingCallback:
    def __init__(self, patience: int = 100):
        self.patience = patience
        self.best_value = float('-inf')
        self.best_trial = 0

    def __call__(self, study, trial):
        if trial.value is not None and trial.value > self.best_value:
            self.best_value = trial.value
            self.best_trial = trial.number

        if trial.number - self.best_trial >= self.patience:
            print(f"    Early stopping: No improvement for {self.patience} trials")
            study.stop()


# =============================================================================
# RF/ET OBJECTIVES - WITH CLASS WEIGHTS
# =============================================================================
def create_rf_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf):
    """Create RF objective with class_weight='balanced' to handle class imbalance."""

    # Compute class weight dict for logging
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"  RF Class weights: {class_weight_dict}")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': 'balanced',  # CRITICAL: Handle class imbalance
            'n_jobs': -1,
            'random_state': RANDOM_SEED,
        }

        try:
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            pf, stats = calculate_profit_factor_with_stats(y_pred, returns_val)

            # Log every 10th trial
            if trial.number % 10 == 0:
                print(f"    Trial {trial.number}: PF={pf:.2f}, "
                      f"TradeRatio={stats['trade_ratio']:.1%}, "
                      f"L/S/F={stats['n_long']}/{stats['n_short']}/{stats['n_flat']}"
                      f"{' [PENALIZED]' if stats['penalized'] else ''}")

            return pf

        except Exception as e:
            print(f"    RF trial error: {e}")
            return 0.5

    return objective


def create_et_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf):
    """Create ET objective with class_weight='balanced' to handle class imbalance."""

    # Compute class weight dict for logging
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"  ET Class weights: {class_weight_dict}")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': 'balanced',  # CRITICAL: Handle class imbalance
            'n_jobs': -1,
            'random_state': RANDOM_SEED,
        }

        try:
            model = ExtraTreesClassifier(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            pf, stats = calculate_profit_factor_with_stats(y_pred, returns_val)

            # Log every 10th trial
            if trial.number % 10 == 0:
                print(f"    Trial {trial.number}: PF={pf:.2f}, "
                      f"TradeRatio={stats['trade_ratio']:.1%}, "
                      f"L/S/F={stats['n_long']}/{stats['n_short']}/{stats['n_flat']}"
                      f"{' [PENALIZED]' if stats['penalized'] else ''}")

            return pf

        except Exception as e:
            print(f"    ET trial error: {e}")
            return 0.5

    return objective


# =============================================================================
# STUDY DIAGNOSTICS
# =============================================================================
def print_study_diagnostics(study, brain: str):
    print(f"\n  OPTUNA DIAGNOSTICS ({brain}):")
    print(f"    Best trial: {study.best_trial.number} / {len(study.trials)}")
    print(f"    Best value: {study.best_value:.4f}")

    best_so_far = float('-inf')
    improvements = []
    for i, trial in enumerate(study.trials):
        if trial.value is not None and trial.value > best_so_far:
            best_so_far = trial.value
            improvements.append(i)

    if improvements:
        print(f"    Trials since last improvement: {len(study.trials) - improvements[-1]}")


# =============================================================================
# MODEL SAVING
# =============================================================================
def save_trained_model(brain: str, pair: str, tf: str, model,
                       scaler, best_params: dict, best_pf: float) -> Tuple[Path, int]:
    """Save model to .joblib file."""

    model_key = f"{pair}_{tf}_{brain}"
    model_file = MODELS_DIR / f'{model_key}.joblib'

    joblib.dump({
        'model': model,  # ACTUAL MODEL OBJECT
        'scaler': scaler,
        'brain': brain,
        'pair': pair,
        'tf': tf,
        'best_pf': best_pf,
        'target_col': TARGET_COL,
        'returns_col': RETURNS_COL,
    }, model_file)

    file_size = model_file.stat().st_size
    min_size = 10000  # 10KB minimum

    if file_size < min_size:
        raise ValueError(f"Model file too small: {file_size} bytes")

    print(f"  Saved: {model_file.name} ({file_size/1024/1024:.2f} MB)")

    # Save params
    params_file = PARAMS_DIR / f'{model_key}_params.json'
    clean_params = {}
    for k, v in best_params.items():
        if hasattr(v, 'item'):
            clean_params[k] = v.item()
        elif v is None:
            clean_params[k] = None
        else:
            clean_params[k] = v

    with open(params_file, 'w') as f:
        json.dump({
            'best_params': clean_params,
            'best_pf': float(best_pf),
            'brain': brain,
            'pair': pair,
            'tf': tf,
            'target_col': TARGET_COL,
            'returns_col': RETURNS_COL,
        }, f, indent=2)

    return model_file, file_size


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def train_rf_et_full_grid():
    print("=" * 70)
    print("CHAOS V1.0 - LOCAL RF/ET TRAINING (GOOGLE DRIVE SYNCED)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  BASE_DIR: {BASE_DIR}")
    print(f"  TARGET_COL: {TARGET_COL}")
    print(f"  RETURNS_COL: {RETURNS_COL}")
    print(f"  FILE_PATTERN: {FEATURE_FILE_PATTERN}")
    print("=" * 70)

    # Load checkpoint
    checkpoint = load_checkpoint()
    completed = checkpoint.get('completed', [])

    total_models = len(ALL_PAIRS) * len(ALL_TIMEFRAMES) * len(BRAINS)

    print(f"\nPairs: {len(ALL_PAIRS)} - {ALL_PAIRS}")
    print(f"Timeframes: {len(ALL_TIMEFRAMES)} - {ALL_TIMEFRAMES}")
    print(f"Brains: {len(BRAINS)} - {BRAINS}")
    print(f"Total models: {total_models}")
    print(f"Completed: {len(completed)}")
    print(f"Remaining: {total_models - len(completed)}")
    print("=" * 70)

    model_num = 0
    start_time = time.time()

    for pair in ALL_PAIRS:
        for tf in ALL_TIMEFRAMES:
            # Load data
            try:
                X_train, y_train, X_val, y_val, returns_val, n_features = load_features(pair, tf)
            except FileNotFoundError as e:
                print(f"\n[SKIP] {pair}_{tf}: File not found")
                continue
            except Exception as e:
                print(f"\n[ERROR] {pair}_{tf}: {e}")
                continue

            for brain in BRAINS:
                model_num += 1
                model_key = f"{pair}_{tf}_{brain}"

                if model_key in completed:
                    print(f"[{model_num}/{total_models}] {model_key}: SKIP (completed)")
                    continue

                print(f"\n[{model_num}/{total_models}] Training {model_key}...")

                # Get Optuna config for this timeframe
                optuna_config = get_optuna_config(tf)
                n_trials = optuna_config['trials']
                patience = optuna_config['patience']
                print(f"  Trials: {n_trials}, Early stop patience: {patience}")

                try:
                    if brain == 'rf_optuna':
                        objective = create_rf_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf)
                    else:
                        objective = create_et_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf)

                    study = optuna.create_study(
                        direction='maximize',
                        sampler=TPESampler(seed=RANDOM_SEED)
                    )

                    study.optimize(
                        objective,
                        n_trials=n_trials,
                        callbacks=[EarlyStoppingCallback(patience=patience)],
                        show_progress_bar=True,
                    )

                    print_study_diagnostics(study, brain)

                    # Retrain final model
                    print(f"  Retraining final model with best params...")
                    best_params = study.best_params.copy()
                    best_params['n_jobs'] = -1
                    best_params['random_state'] = RANDOM_SEED
                    best_params['class_weight'] = 'balanced'  # CRITICAL: Handle class imbalance

                    if brain == 'rf_optuna':
                        model = RandomForestClassifier(**best_params)
                    else:
                        model = ExtraTreesClassifier(**best_params)

                    model.fit(X_train, y_train)

                    # Verify
                    test_pred = model.predict(X_val[:10])
                    print(f"  Test predictions: {test_pred}")

                    # Save
                    model_file, file_size = save_trained_model(
                        brain, pair, tf, model, None,
                        study.best_params, study.best_value
                    )

                    # Update checkpoint - LIST FORMAT
                    checkpoint = add_to_checkpoint(checkpoint, model_key)
                    print(f"  Checkpoint: {len(checkpoint['completed'])} models")

                    # ETA
                    elapsed = time.time() - start_time
                    done = len(checkpoint['completed'])
                    remaining = total_models - done
                    if done > 0:
                        avg_time = elapsed / done
                        eta = remaining * avg_time
                        print(f"  ETA: {eta/60:.1f} minutes remaining")

                except KeyboardInterrupt:
                    print("\n\nInterrupted. Progress saved.")
                    return

                except Exception as e:
                    print(f"  [ERROR] {model_key}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Models completed: {len(checkpoint['completed'])}")
    print("=" * 70)


# =============================================================================
# PF VERIFICATION - RUN BEFORE TRAINING
# =============================================================================
def verify_pf_calculation():
    """Run this to verify PF functions are correct."""
    print("=" * 60)
    print("PF CALCULATION VERIFICATION")
    print("=" * 60)

    # Test 1: Known values
    print("\nTest 1: Known values")
    preds = np.array([0, 1, 2, 0, 2])  # short, flat, long, short, long
    rets = np.array([0.01, 0.01, 0.01, -0.01, -0.01])
    # positions = [-1, 0, 1, -1, 1]
    # pnl = [-0.01, 0, 0.01, 0.01, -0.01]
    # gp = 0.02, gl = 0.02, pf = 1.0

    result = calculate_profit_factor_detailed(preds, rets)
    print(f"  Predictions: {preds}")
    print(f"  Returns: {rets}")
    print(f"  PF: {result['pf']:.4f} (expected: 1.0)")
    assert abs(result['pf'] - 1.0) < 0.01, f"Test 1 FAILED: PF={result['pf']}"
    print("  PASSED")

    # Test 2: Random predictions
    print("\nTest 2: Random predictions (should be ~1.0)")
    np.random.seed(42)
    random_preds = np.random.randint(0, 3, size=10000)
    random_rets = np.random.randn(10000) * 0.01

    result = calculate_profit_factor_detailed(random_preds, random_rets)
    print(f"  PF: {result['pf']:.4f} (expected: 0.8-1.2)")
    assert 0.7 < result['pf'] < 1.3, f"Test 2 FAILED: PF={result['pf']}"
    print("  PASSED")

    # Test 3: Perfect prediction
    print("\nTest 3: Perfect prediction (should be high)")
    perfect_rets = np.random.randn(10000) * 0.01
    perfect_preds = np.where(perfect_rets > 0, 2, 0)

    result = calculate_profit_factor_detailed(perfect_preds, perfect_rets)
    print(f"  PF: {result['pf']:.4f} (expected: >5.0)")
    assert result['pf'] > 3.0, f"Test 3 FAILED: PF={result['pf']}"
    print("  PASSED")

    # Test 4: Anti-perfect
    print("\nTest 4: Anti-perfect prediction (should be low)")
    anti_preds = np.where(perfect_rets > 0, 0, 2)

    result = calculate_profit_factor_detailed(anti_preds, perfect_rets)
    print(f"  PF: {result['pf']:.4f} (expected: <0.3)")
    assert result['pf'] < 0.5, f"Test 4 FAILED: PF={result['pf']}"
    print("  PASSED")

    # Test 5: Position conversion
    print("\nTest 5: Position conversion")
    test_preds = np.array([0, 1, 2])
    expected_pos = np.array([-1.0, 0.0, 1.0])
    actual_pos = convert_to_positions(test_preds)
    print(f"  Input: {test_preds}")
    print(f"  Output: {actual_pos}")
    print(f"  Expected: {expected_pos}")
    assert np.allclose(actual_pos, expected_pos), "Test 5 FAILED"
    print("  PASSED")

    print("\n" + "=" * 60)
    print("ALL VERIFICATION TESTS PASSED")
    print("=" * 60)

    return True


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Verify PF calculation first
    verify_pf_calculation()
    print()

    # Then run training
    train_rf_et_full_grid()
