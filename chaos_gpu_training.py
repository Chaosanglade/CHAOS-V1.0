#!/usr/bin/env python3
"""
CHAOS V1.0 - GPU Training Script (All 19 Brains)
=================================================
Standalone script for training all GPU-based models.
Run with: python chaos_gpu_training.py

19 GPU Brains:
  - Gradient Boosting: lgb, xgb, cat (+ v2 variants) = 6
  - TabNet & MLP = 2
  - RNN/Transformer: lstm, gru, transformer = 3
  - CNN-based: cnn1d, tcn, wavenet = 3
  - Attention: attention_net, residual_mlp, ensemble_nn = 3
  - Time Series: nbeats, tft = 2
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import gc
import json
import time
import joblib
import warnings
import threading
import functools
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

# GPU/Neural Network imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Gradient Boosting
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# TabNet
try:
    from pytorch_tabnet.tab_model import TabNetClassifier as TabNetModel
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("WARNING: pytorch_tabnet not installed, tabnet_optuna will be skipped")

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# CONFIGURATION - EXPLICIT, NO DYNAMIC SELECTION
# =============================================================================

# Detect environment
if os.path.exists('/content/drive'):
    # Colab
    BASE_DIR = Path("/content/drive/MyDrive/chaos_v1.0")
else:
    # Local Windows
    BASE_DIR = Path(r"G:\My Drive\chaos_v1.0")

FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "gpu_checkpoint.json"
PARAMS_DIR = MODELS_DIR / "optuna_params"
OPTUNA_CHECKPOINT_DIR = MODELS_DIR / "optuna_checkpoints"
PROGRESS_LOG_PATH = MODELS_DIR / "training_progress.log"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_DIR.mkdir(parents=True, exist_ok=True)
OPTUNA_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum trials before we accept a partial Optuna checkpoint
MIN_TRIALS_FOR_RESUME = 10

# Heartbeat checkpoint interval (seconds)
HEARTBEAT_INTERVAL = 300  # 5 minutes

# Smart ordering: timeframes sorted by estimated training time (fastest first)
TF_SPEED_ORDER = ['D1', 'W1', 'MN1', 'H4', 'H1', 'M30', 'M15', 'M5', 'M1']

# EXPLICIT COLUMN NAMES - NEVER CHANGE THESE
TARGET_COL = 'target_3class_8'
RETURNS_COL = 'target_return_8'

# FILE PATTERN
FEATURE_FILE_PATTERN = "{pair}_{tf}_features.parquet"

# GRID - COMPLETE
ALL_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
             'USDCHF', 'NZDUSD', 'EURJPY', 'GBPJPY']

ALL_TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']

# GPU BRAINS - ALL 19
GPU_BRAINS = [
    # Gradient Boosting (6)
    'lgb_optuna', 'xgb_optuna', 'cat_optuna',
    'lgb_v2_optuna', 'xgb_v2_optuna', 'cat_v2_optuna',
    # TabNet & MLP (2)
    'tabnet_optuna', 'mlp_optuna',
    # RNN/Transformer (3)
    'lstm_optuna', 'gru_optuna', 'transformer_optuna',
    # CNN-based (3)
    'cnn1d_optuna', 'tcn_optuna', 'wavenet_optuna',
    # Attention & Residual (3)
    'attention_net_optuna', 'residual_mlp_optuna', 'ensemble_nn_optuna',
    # Time Series Specialized (2)
    'nbeats_optuna', 'tft_optuna'
]

# Trial counts by timeframe
TRIALS_CONFIG = {
    'M1': 50,
    'M5': 50,
    'M15': 50,
    'M30': 50,
    'H1': 50,
    'H4': 40,
    'D1': 30,
    'W1': 20,
    'MN1': 15,
}

# TabNet-specific trials (slower model, fewer trials)
TABNET_TRIALS_CONFIG = {
    'M1': 20,
    'M5': 20,
    'M15': 20,
    'M30': 20,
    'H1': 20,
    'H4': 15,
    'D1': 15,
    'W1': 10,
    'MN1': 10,
}

# TRAINING PARAMS
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 30

# Trade frequency penalty - models must trade at least this % of the time
MIN_TRADE_RATIO = 0.10  # 10% minimum trades (non-FLAT predictions)

# Set seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# GPU AVAILABILITY DETECTION
# =============================================================================
def detect_gpu_support():
    """Detect GPU support for each framework."""
    gpu_status = {
        'pytorch': torch.cuda.is_available(),
        'lightgbm': False,
        'xgboost': False,
        'catboost': False,
    }

    # Test LightGBM GPU
    try:
        import lightgbm as lgb
        params = {'device': 'gpu', 'verbose': -1, 'num_leaves': 31, 'n_estimators': 1}
        model = lgb.LGBMClassifier(**params)
        model.fit([[1,2,3], [4,5,6]], [0, 1])
        gpu_status['lightgbm'] = True
    except:
        gpu_status['lightgbm'] = False

    # Test XGBoost GPU
    try:
        import xgboost as xgb
        params = {'tree_method': 'hist', 'device': 'cuda', 'max_depth': 3, 'n_estimators': 1}
        model = xgb.XGBClassifier(**params)
        model.fit([[1,2,3], [4,5,6]], [0, 1])
        gpu_status['xgboost'] = True
    except:
        gpu_status['xgboost'] = False

    # Test CatBoost GPU
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=1, task_type='GPU', devices='0', verbose=0)
        model.fit([[1,2,3], [4,5,6]], [0, 1])
        gpu_status['catboost'] = True
    except:
        gpu_status['catboost'] = False

    return gpu_status

# Detect GPU support at import time
GPU_STATUS = None  # Will be populated on first run


def get_gpu_status():
    """Get or initialize GPU status."""
    global GPU_STATUS
    if GPU_STATUS is None:
        print("Detecting GPU support for each framework...")
        GPU_STATUS = detect_gpu_support()
        for framework, available in GPU_STATUS.items():
            status = "GPU" if available else "CPU"
            print(f"  {framework}: {status}")
    return GPU_STATUS


# =============================================================================
# FIX 2: COLAB KEEP-ALIVE
# =============================================================================
def setup_colab_keepalive():
    """Output JavaScript auto-clicker to prevent Colab idle disconnects."""
    if not os.path.exists('/content/drive'):
        return  # Not on Colab
    try:
        from IPython.display import display, HTML
        display(HTML('''
        <script>
        function ClickConnect(){
            console.log("Colab keep-alive: clicking connect button");
            var buttons = document.querySelectorAll("colab-connect-button");
            if (buttons.length > 0) { buttons[0].click(); }
        }
        setInterval(ClickConnect, 60000);  // Every 60 seconds
        </script>
        <p style="color: green;">Colab keep-alive active (60s interval)</p>
        '''))
        print("[KEEPALIVE] Colab auto-clicker installed (60s interval)")
    except ImportError:
        print("[KEEPALIVE] IPython not available, skipping keep-alive setup")


# =============================================================================
# FIX 3: SAFE SAVE — GOOGLE DRIVE RECONNECTION DECORATOR
# =============================================================================
def remount_drive():
    """Attempt to remount Google Drive on Colab."""
    if not os.path.exists('/content'):
        return False
    try:
        from google.colab import drive
        print("[DRIVE] Attempting force remount...")
        drive.mount('/content/drive', force_remount=True)
        print("[DRIVE] Remount successful")
        return True
    except Exception as e:
        print(f"[DRIVE] Remount failed: {e}")
        return False


def safe_save(func):
    """Decorator that retries file saves with Drive remount on failure."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (IOError, OSError) as e:
                print(f"[SAFE_SAVE] Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    remount_drive()
                    time.sleep(2)
                else:
                    print(f"[SAFE_SAVE] All {max_retries} attempts failed for {func.__name__}")
                    raise
    return wrapper


# =============================================================================
# FIX 4: HEARTBEAT CHECKPOINT THREAD
# =============================================================================
class HeartbeatCheckpointer:
    """Background thread that saves checkpoint every N seconds."""

    def __init__(self, interval=HEARTBEAT_INTERVAL):
        self.interval = interval
        self._checkpoint_ref = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self, checkpoint):
        """Start heartbeat with a reference to the checkpoint dict."""
        self._checkpoint_ref = checkpoint
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[HEARTBEAT] Started (every {self.interval}s)")

    def update_checkpoint(self, checkpoint):
        """Update the checkpoint reference (called after add_to_checkpoint)."""
        with self._lock:
            self._checkpoint_ref = checkpoint

    def stop(self):
        """Stop the heartbeat thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("[HEARTBEAT] Stopped")

    def _run(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(self.interval)
            if self._stop_event.is_set():
                break
            with self._lock:
                if self._checkpoint_ref:
                    try:
                        _heartbeat_save(self._checkpoint_ref)
                    except Exception as e:
                        print(f"[HEARTBEAT] Save failed: {e}")


@safe_save
def _heartbeat_save(checkpoint):
    """Save checkpoint from heartbeat thread (with safe_save retry)."""
    checkpoint['last_heartbeat'] = datetime.now().isoformat()
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f, indent=2)


# =============================================================================
# FIX 6: PROGRESS LOGGING
# =============================================================================
@safe_save
def log_progress(message: str):
    """Append timestamped progress line to training_progress.log."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {message}\n"
    with open(PROGRESS_LOG_PATH, 'a') as f:
        f.write(line)


# =============================================================================
# FIX 1: OPTUNA TRIAL-LEVEL CHECKPOINTING
# =============================================================================
def get_optuna_checkpoint_path(model_key: str) -> Path:
    """Get path for Optuna partial checkpoint for a given model."""
    return OPTUNA_CHECKPOINT_DIR / f"{model_key}_optuna.json"


@safe_save
def save_optuna_checkpoint(model_key: str, study):
    """Save best trial params from an Optuna study to a partial checkpoint."""
    if len(study.trials) == 0:
        return
    best = study.best_trial
    data = {
        'model_key': model_key,
        'n_trials_completed': len(study.trials),
        'best_trial_number': best.number,
        'best_value': best.value,
        'best_params': best.params,
        'timestamp': datetime.now().isoformat(),
    }
    path = get_optuna_checkpoint_path(model_key)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_optuna_checkpoint(model_key: str) -> Optional[Dict]:
    """Load partial Optuna checkpoint if it exists and has enough trials."""
    path = get_optuna_checkpoint_path(model_key)
    if not path.exists():
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if data.get('n_trials_completed', 0) >= MIN_TRIALS_FOR_RESUME:
            return data
        return None
    except Exception:
        return None


def delete_optuna_checkpoint(model_key: str):
    """Delete partial Optuna checkpoint after successful model save."""
    path = get_optuna_checkpoint_path(model_key)
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass


class OptunaTrialCheckpointCallback:
    """Callback that saves best trial after every Optuna trial."""

    def __init__(self, model_key: str, log_every_n: int = 5):
        self.model_key = model_key
        self.log_every_n = log_every_n
        self.best_value = float('-inf')

    def __call__(self, study, trial):
        # Save checkpoint after every trial (overwriting previous)
        if trial.value is not None and trial.value > self.best_value:
            self.best_value = trial.value
            save_optuna_checkpoint(self.model_key, study)

        # Log progress every N trials
        if trial.number > 0 and trial.number % self.log_every_n == 0:
            log_progress(
                f"OPTUNA {self.model_key}: trial {trial.number}, "
                f"best_pf={study.best_value:.4f}"
            )


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
# TABNET BALANCED SAMPLING - UNDERSAMPLE FLAT CLASS
# =============================================================================
TABNET_MAX_SAMPLES = 100_000  # Max total samples for TabNet


def sample_for_tabnet(X_train, y_train, max_samples=TABNET_MAX_SAMPLES, random_state=RANDOM_SEED):
    """
    BALANCED sampling for TabNet - undersamples majority class (FLAT).

    The problem: Class imbalance (e.g., 10% LONG, 80% FLAT, 10% SHORT)
    causes TabNet to predict FLAT for everything → 80% accuracy but 0% trades.

    Solution: Sample equal amounts from each class to force learning of
    LONG/SHORT patterns.

    Args:
        X_train: Training features
        y_train: Training labels
        max_samples: Maximum total samples (will be split equally among classes)
        random_state: Random seed for reproducibility

    Returns:
        X_sampled, y_sampled: Balanced data with equal class representation
    """
    from collections import Counter

    np.random.seed(random_state)

    # Count classes
    class_counts = Counter(y_train)
    n_classes = len(class_counts)
    print(f"  Original class distribution: {dict(sorted(class_counts.items()))}")

    # Find minority class count
    min_class_count = min(class_counts.values())

    # Calculate samples per class:
    # - Don't exceed minority class count (can't oversample without replacement)
    # - Don't exceed max_samples / n_classes
    samples_per_class = min(min_class_count, max_samples // n_classes)

    # Sample equal amounts from each class
    indices = []
    for class_label in sorted(class_counts.keys()):
        class_indices = np.where(y_train == class_label)[0]
        sampled = np.random.choice(class_indices, size=samples_per_class, replace=False)
        indices.extend(sampled)

    # Shuffle to mix classes
    indices = np.array(indices)
    np.random.shuffle(indices)

    X_sampled = X_train[indices]
    y_sampled = y_train[indices]

    new_counts = Counter(y_sampled)
    print(f"  Balanced class distribution: {dict(sorted(new_counts.items()))}")
    print(f"  TabNet balanced sampling: {len(X_train):,} -> {len(X_sampled):,} rows")

    return X_sampled, y_sampled


# =============================================================================
# NEURAL NETWORK BALANCED SAMPLING - FOR ALL 12 NN MODELS
# =============================================================================
NN_MAX_SAMPLES = 200_000  # Max samples for neural network training


def sample_for_neural_network(X_train, y_train, max_samples=NN_MAX_SAMPLES, random_state=RANDOM_SEED):
    """
    BALANCED sampling for neural networks - undersamples majority class.

    Applied to ALL 12 neural network models:
    - MLP, LSTM, GRU, Transformer
    - CNN1D, TCN, WaveNet
    - AttentionNet, ResidualMLP, EnsembleNN
    - NBeats, TFT

    Uses same balanced approach as TabNet sampling.
    """
    from collections import Counter

    if len(X_train) <= max_samples:
        # Still apply class balancing even for smaller datasets
        class_counts = Counter(y_train)
        min_class_count = min(class_counts.values())
        # If already small and balanced enough, return as-is
        if min_class_count > max_samples // 6:
            return X_train, y_train

    np.random.seed(random_state)

    # Count classes
    class_counts = Counter(y_train)
    n_classes = len(class_counts)

    # Find minority class count
    min_class_count = min(class_counts.values())

    # Calculate samples per class
    samples_per_class = min(min_class_count, max_samples // n_classes)

    # Sample equal amounts from each class
    indices = []
    for class_label in sorted(class_counts.keys()):
        class_indices = np.where(y_train == class_label)[0]
        sampled = np.random.choice(class_indices, size=samples_per_class, replace=False)
        indices.extend(sampled)

    # Shuffle to mix classes
    indices = np.array(indices)
    np.random.shuffle(indices)

    X_sampled = X_train[indices]
    y_sampled = y_train[indices]

    print(f"  NN balanced sampling: {len(X_train):,} -> {len(X_sampled):,} rows")

    return X_sampled, y_sampled


def sample_validation(X_val, y_val, returns_val, max_samples=100_000, random_state=RANDOM_SEED):
    """Sample validation set if too large."""
    if len(X_val) <= max_samples:
        return X_val, y_val, returns_val

    np.random.seed(random_state)
    indices = np.random.choice(len(X_val), size=max_samples, replace=False)

    return X_val[indices], y_val[indices], returns_val[indices]


# =============================================================================
# CHECKPOINT MANAGEMENT - LIST FORMAT
# =============================================================================
def load_checkpoint() -> Dict:
    """Load checkpoint or create new one."""
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, 'r') as f:
                checkpoint = json.load(f)
            # Ensure completed is a list
            if not isinstance(checkpoint.get('completed'), list):
                checkpoint['completed'] = []
            return checkpoint
        except Exception as e:
            print(f"WARNING: Could not load checkpoint: {e}")
    return {"completed": [], "started": datetime.now().isoformat()}


@safe_save
def save_checkpoint(checkpoint: Dict) -> None:
    """Save checkpoint to file (with safe_save retry on Drive errors)."""
    checkpoint['last_updated'] = datetime.now().isoformat()
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def add_to_checkpoint(checkpoint: Dict, model_key: str) -> Dict:
    """Add completed model to checkpoint using LIST APPEND."""
    if model_key not in checkpoint['completed']:
        checkpoint['completed'].append(model_key)  # LIST APPEND
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
    exclude_patterns = ['target_', 'return', 'Open', 'High', 'Low', 'Close',
                        'Volume', 'timestamp', 'date', 'pair', 'symbol', 'tf',
                        'bar_time', 'Bid_', 'Ask_', 'Spread_', 'Unnamed']

    feature_cols = []
    for col in df.columns:
        if any(p in col for p in exclude_patterns):
            continue
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
    filename = FEATURE_FILE_PATTERN.format(pair=pair, tf=tf)
    filepath = FEATURES_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Feature file not found: {filepath}")

    df = pd.read_parquet(filepath)
    print(f"  Loaded {pair}_{tf}: {len(df):,} rows, {len(df.columns)} columns")

    # VALIDATE EXPLICIT COLUMNS EXIST
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found!")
    if RETURNS_COL not in df.columns:
        raise ValueError(f"Returns column '{RETURNS_COL}' not found!")

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

    X = df[feature_cols].values.astype('float32')
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # EXPLICIT target column
    y = df[TARGET_COL].values.copy()
    y = np.nan_to_num(y, nan=0.0)
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
    def __init__(self, patience: int = 30):
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
# NEURAL NETWORK ARCHITECTURES (13 classes)
# =============================================================================

class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for classification."""
    def __init__(self, input_size, num_classes=3, hidden_sizes=None, dropout=0.3):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LSTMClassifier(nn.Module):
    """LSTM-based classifier."""
    def __init__(self, input_size, num_classes=3, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        return self.fc(out)


class GRUClassifier(nn.Module):
    """GRU-based classifier."""
    def __init__(self, input_size, num_classes=3, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        return self.fc(out)


class TransformerClassifier(nn.Module):
    """Transformer-based classifier."""
    def __init__(self, input_size, num_classes=3, d_model=128, nhead=4,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


class CNN1DClassifier(nn.Module):
    """1D CNN classifier."""
    def __init__(self, input_size, num_classes=3, channels=None, kernel_size=3, dropout=0.3):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]
        layers = []
        in_channels = 1
        for out_channels in channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)
        return self.fc(x)


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.dropout(torch.relu(self.bn1(self.conv1(x))))
        out = self.dropout(torch.relu(self.bn2(self.conv2(out))))
        if self.downsample:
            residual = self.downsample(residual)
        return torch.relu(out + residual)


class TCNClassifier(nn.Module):
    """Temporal Convolutional Network classifier."""
    def __init__(self, input_size, num_classes=3, channels=None, kernel_size=3, dropout=0.3):
        super().__init__()
        if channels is None:
            channels = [64, 128]
        layers = []
        in_channels = 1
        for i, out_channels in enumerate(channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.tcn(x)
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)
        return self.fc(x)


class WaveNetBlock(nn.Module):
    """WaveNet-style dilated causal convolution block."""
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.dilated_conv = nn.Conv1d(channels, channels * 2, kernel_size,
                                      padding=padding, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        residual = x
        out = self.dilated_conv(x)
        out = out[:, :, :x.size(2)]
        tanh_out = torch.tanh(out[:, :out.size(1)//2, :])
        sigmoid_out = torch.sigmoid(out[:, out.size(1)//2:, :])
        out = tanh_out * sigmoid_out
        out = self.conv_1x1(out)
        return out + residual


class WaveNetClassifier(nn.Module):
    """WaveNet-style classifier."""
    def __init__(self, input_size, num_classes=3, channels=64, num_layers=4,
                 kernel_size=2, dropout=0.3):
        super().__init__()
        self.input_conv = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList([
            WaveNetBlock(channels, kernel_size, 2**i)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)
        return self.fc(x)


class AttentionNetClassifier(nn.Module):
    """Self-attention based classifier."""
    def __init__(self, input_size, num_classes=3, hidden_size=128, num_heads=4, dropout=0.3):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = x.squeeze(1)
        return self.fc(x)


class ResidualBlock(nn.Module):
    """Residual block for MLP."""
    def __init__(self, size, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )

    def forward(self, x):
        return torch.relu(x + self.block(x))


class ResidualMLPClassifier(nn.Module):
    """Residual MLP classifier."""
    def __init__(self, input_size, num_classes=3, hidden_size=256, num_blocks=3, dropout=0.3):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_size, dropout) for _ in range(num_blocks)
        ])
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.input_projection(x))
        x = self.blocks(x)
        return self.fc(x)


class EnsembleNNClassifier(nn.Module):
    """Ensemble of different NN architectures."""
    def __init__(self, input_size, num_classes=3, hidden_size=64, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        self.cnn_input = nn.Conv1d(1, hidden_size, 3, padding=1)
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        self.cnn_fc = nn.Linear(hidden_size, num_classes)
        self.combine = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x):
        mlp_out = self.mlp(x)
        cnn_x = x.unsqueeze(1)
        cnn_out = torch.relu(self.cnn_input(cnn_x))
        cnn_out = self.cnn_pool(cnn_out).squeeze(-1)
        cnn_out = self.cnn_fc(cnn_out)
        combined = torch.cat([mlp_out, cnn_out], dim=1)
        return self.combine(combined)


class NBeatsBlock(nn.Module):
    """N-BEATS block."""
    def __init__(self, input_size, theta_size, hidden_size=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.theta_b = nn.Linear(hidden_size, theta_size)
        self.theta_f = nn.Linear(hidden_size, theta_size)
        self.backcast = nn.Linear(theta_size, input_size)
        self.forecast = nn.Linear(theta_size, theta_size)

    def forward(self, x):
        h = self.fc(x)
        return self.backcast(self.theta_b(h)), self.forecast(self.theta_f(h))


class NBeatsClassifier(nn.Module):
    """N-BEATS classifier."""
    def __init__(self, input_size, num_classes=3, hidden_size=256, num_blocks=4, dropout=0.3):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, hidden_size//4, hidden_size)
            for _ in range(num_blocks)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size//4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        residuals = x
        forecast = None
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = block_forecast if forecast is None else forecast + block_forecast
        return self.classifier(forecast)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for TFT."""
    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.1):
        super().__init__()
        output_size = output_size or input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None

    def forward(self, x):
        skip = self.skip(x) if self.skip else x
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        out = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        return self.layernorm(skip + gate * out)


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer classifier."""
    def __init__(self, input_size, num_classes=3, hidden_size=128, num_heads=4,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.input_grn = GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.output_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h = self.input_grn(x)
        h = h.unsqueeze(1)
        lstm_out, _ = self.lstm(h)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        h = self.attention_norm(lstm_out + attn_out)
        h = self.output_grn(h.squeeze(1))
        return self.classifier(h)


# =============================================================================
# PYTORCH TRAINING UTILITIES
# =============================================================================

# Batch size for validation inference to prevent OOM
VAL_INFERENCE_BATCH_SIZE = 2048


def batched_inference(model, X_tensor, batch_size=VAL_INFERENCE_BATCH_SIZE):
    """
    Batched inference to prevent OOM on large validation sets.

    Conv1D models create massive intermediate tensors:
    - Input: 100,000 x 275 features
    - After conv: 100,000 x 275 x num_filters
    - This can exceed 80GB GPU memory

    Solution: Process in batches and concatenate results.
    """
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            if batch.device != next(model.parameters()).device:
                batch = batch.to(next(model.parameters()).device)
            out = model(batch)
            outputs.append(out.cpu())  # Move to CPU immediately to free GPU
            del out
    torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0)


def train_pytorch_model(model, X_train, y_train, X_val, y_val,
                        epochs=100, batch_size=256, learning_rate=1e-3, patience=20):
    """Train a PyTorch model with early stopping, class weights, and OOM-safe validation."""
    model = model.to(DEVICE)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Keep validation data on CPU, move in batches during inference
    X_val_tensor = torch.FloatTensor(X_val)  # CPU
    y_val_tensor = torch.LongTensor(y_val)   # CPU

    # CRITICAL: Compute class weights for balanced loss
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights_tensor = torch.FloatTensor(weights).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  # Weighted loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # BATCHED VALIDATION - prevents OOM for Conv1D models
        model.eval()
        val_outputs = batched_inference(model, X_val_tensor, VAL_INFERENCE_BATCH_SIZE)
        val_loss = criterion(val_outputs.to(DEVICE), y_val_tensor.to(DEVICE)).item()
        del val_outputs  # Free memory immediately
        torch.cuda.empty_cache()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_pytorch(model, X):
    """Get predictions from PyTorch model using batched inference to prevent OOM."""
    X_tensor = torch.FloatTensor(X)  # Keep on CPU
    outputs = batched_inference(model, X_tensor, VAL_INFERENCE_BATCH_SIZE)
    _, predictions = torch.max(outputs, 1)
    return predictions.numpy()


# =============================================================================
# OPTUNA OBJECTIVE FUNCTIONS
# =============================================================================

def create_lgb_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf):
    """LightGBM objective with class_weight='balanced' and GPU if available."""

    # Log class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    print(f"  LGB Class weights: {dict(zip(classes, weights))}")

    # Check GPU availability
    gpu_status = get_gpu_status()
    use_gpu = gpu_status.get('lightgbm', False)

    def objective(trial):
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'class_weight': 'balanced',  # CRITICAL: Handle class imbalance
            'random_state': RANDOM_SEED,
            'verbosity': -1,
            'n_jobs': -1,
        }
        # Add GPU params only if available
        if use_gpu:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
            params.pop('n_jobs', None)  # Remove n_jobs when using GPU
        try:
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            pf, stats = calculate_profit_factor_with_stats(preds, returns_val)

            if trial.number % 10 == 0:
                print(f"    Trial {trial.number}: PF={pf:.2f}, "
                      f"TradeRatio={stats['trade_ratio']:.1%}"
                      f"{' [PENALIZED]' if stats['penalized'] else ''}")

            return pf
        except:
            return 0.5
    return objective


def create_lgb_v2_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf):
    """LightGBM v2 objective with more regularization, class_weight='balanced', GPU if available."""

    # Check GPU availability
    gpu_status = get_gpu_status()
    use_gpu = gpu_status.get('lightgbm', False)

    def objective(trial):
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 100.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 100.0, log=True),
            'class_weight': 'balanced',  # CRITICAL: Handle class imbalance
            'random_state': RANDOM_SEED,
            'verbosity': -1,
            'n_jobs': -1,
        }
        # Add GPU params only if available
        if use_gpu:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
            params.pop('n_jobs', None)
        try:
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            pf, stats = calculate_profit_factor_with_stats(preds, returns_val)
            return pf
        except:
            return 0.5
    return objective


def create_xgb_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf):
    """XGBoost objective with sample weights and GPU acceleration."""

    # Compute sample weights for class imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    sample_weights = np.array([class_weight_dict[y] for y in y_train])
    print(f"  XGB Class weights: {class_weight_dict}")

    def objective(trial):
        params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'tree_method': 'hist',        # XGBoost 2.0+ uses 'hist' not 'gpu_hist'
            'device': 'cuda',             # This enables GPU
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': RANDOM_SEED,
            'verbosity': 0,
        }
        try:
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)  # Use sample weights
            preds = model.predict(X_val)
            pf, stats = calculate_profit_factor_with_stats(preds, returns_val)

            if trial.number % 10 == 0:
                print(f"    Trial {trial.number}: PF={pf:.2f}, "
                      f"TradeRatio={stats['trade_ratio']:.1%}"
                      f"{' [PENALIZED]' if stats['penalized'] else ''}")

            return pf
        except:
            return 0.5
    return objective


def create_xgb_v2_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf):
    """XGBoost v2 objective with more regularization, sample weights, and GPU."""

    # Compute sample weights for class imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    sample_weights = np.array([class_weight_dict[y] for y in y_train])

    def objective(trial):
        params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'tree_method': 'hist',        # XGBoost 2.0+ uses 'hist' not 'gpu_hist'
            'device': 'cuda',             # This enables GPU
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 100.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 100.0, log=True),
            'random_state': RANDOM_SEED,
            'verbosity': 0,
        }
        try:
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)  # Use sample weights
            preds = model.predict(X_val)
            pf, stats = calculate_profit_factor_with_stats(preds, returns_val)
            return pf
        except:
            return 0.5
    return objective


def create_cat_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf):
    """CatBoost objective with auto_class_weights='Balanced' and GPU acceleration."""

    # Log class distribution
    classes, counts = np.unique(y_train, return_counts=True)
    print(f"  CAT Class distribution: {dict(zip(classes, counts))}")

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'auto_class_weights': 'Balanced',  # CRITICAL: Handle class imbalance
            'task_type': 'GPU',           # GPU acceleration
            'devices': '0',
            'random_seed': RANDOM_SEED,
            'verbose': False,
        }
        try:
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            pf, stats = calculate_profit_factor_with_stats(preds, returns_val)

            if trial.number % 10 == 0:
                print(f"    Trial {trial.number}: PF={pf:.2f}, "
                      f"TradeRatio={stats['trade_ratio']:.1%}"
                      f"{' [PENALIZED]' if stats['penalized'] else ''}")

            return pf
        except:
            return 0.5
    return objective


def create_cat_v2_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf):
    """CatBoost v2 objective with more regularization, auto_class_weights, and GPU."""
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'depth': trial.suggest_int('depth', 3, 6),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 100.0, log=True),
            'auto_class_weights': 'Balanced',  # CRITICAL: Handle class imbalance
            'task_type': 'GPU',           # GPU acceleration
            'devices': '0',
            'random_seed': RANDOM_SEED,
            'verbose': False,
        }
        try:
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            pf, stats = calculate_profit_factor_with_stats(preds, returns_val)
            return pf
        except:
            return 0.5
    return objective


def create_tabnet_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf):
    """TabNet objective with GPU acceleration, stratified sampling, and optimized settings."""

    if not TABNET_AVAILABLE:
        print("  TabNet not available - skipping")
        return lambda trial: 0.5

    # Apply stratified sampling if dataset too large (maintains class balance)
    X_train_tab, y_train_tab = sample_for_tabnet(X_train, y_train, TABNET_MAX_SAMPLES)

    # Also sample validation set if too large (for faster evaluation)
    if len(X_val) > 100_000:
        val_frac = 100_000 / len(X_val)
        X_val_tab, _, y_val_tab, _, returns_val_tab, _ = train_test_split(
            X_val, y_val, returns_val,
            train_size=val_frac,
            stratify=y_val,
            random_state=RANDOM_SEED
        )
        print(f"  TabNet val sampling: {len(X_val):,} -> {len(X_val_tab):,}")
    else:
        X_val_tab, y_val_tab, returns_val_tab = X_val, y_val, returns_val

    def objective(trial):
        # Optimized hyperparameter ranges
        n_d = trial.suggest_int('n_d', 8, 32)  # Reduced from 64
        params = {
            'n_d': n_d,
            'n_a': n_d,  # Usually same as n_d
            'n_steps': trial.suggest_int('n_steps', 3, 7),  # Reduced from 10
            'gamma': trial.suggest_float('gamma', 1.0, 2.0),
            'n_independent': trial.suggest_int('n_independent', 1, 3),
            'n_shared': trial.suggest_int('n_shared', 1, 3),
            'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True),
            'mask_type': trial.suggest_categorical('mask_type', ['sparsemax', 'entmax']),
        }

        try:
            model = TabNetModel(
                n_d=params['n_d'],
                n_a=params['n_a'],
                n_steps=params['n_steps'],
                gamma=params['gamma'],
                n_independent=params['n_independent'],
                n_shared=params['n_shared'],
                lambda_sparse=params['lambda_sparse'],
                mask_type=params['mask_type'],
                device_name='cuda',      # Force GPU
                verbose=0,               # Quiet during trials
                seed=RANDOM_SEED,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                scheduler_params={"step_size": 10, "gamma": 0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
            )

            # Optimized training settings for GPU
            model.fit(
                X_train_tab, y_train_tab,
                eval_set=[(X_val_tab, y_val_tab)],
                eval_metric=['accuracy'],
                max_epochs=30,           # Reduced for faster trials
                patience=7,              # Early stop faster
                batch_size=4096,         # Large batch for GPU efficiency
                virtual_batch_size=256,  # Increased from 128
                num_workers=0,           # Required for Colab
                drop_last=False,
            )

            preds = model.predict(X_val_tab)
            pf, stats = calculate_profit_factor_with_stats(preds, returns_val_tab)

            if trial.number % 5 == 0:
                print(f"    Trial {trial.number}: PF={pf:.2f}, "
                      f"TradeRatio={stats['trade_ratio']:.1%}"
                      f"{' [PENALIZED]' if stats['penalized'] else ''}")

            return pf

        except Exception as e:
            print(f"    TabNet trial {trial.number} failed: {e}")
            return 0.5

    return objective


def create_dl_objective(model_class, X_train, y_train, X_val, y_val, returns_val, pair, tf, n_features):
    """Deep learning objective with balanced sampling and model-specific instantiation."""

    # Apply balanced sampling for large datasets (prevents 15+ min stuck on M1)
    X_train_nn, y_train_nn = sample_for_neural_network(X_train, y_train, NN_MAX_SAMPLES)

    # Sample validation if very large
    if len(X_val) > 100_000:
        X_val_nn, y_val_nn, returns_val_nn = sample_validation(X_val, y_val, returns_val, 100_000)
    else:
        X_val_nn, y_val_nn, returns_val_nn = X_val, y_val, returns_val

    # Scale after sampling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_nn)
    X_val_scaled = scaler.transform(X_val_nn)

    # Log class distribution (after sampling)
    classes, counts = np.unique(y_train_nn, return_counts=True)
    print(f"  DL Class distribution (balanced): {dict(zip(classes, counts))}")

    # Get model class name for proper instantiation
    model_name = model_class.__name__

    def objective(trial):
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

        try:
            # Model-specific instantiation based on class name
            if model_name == 'MLPClassifier':
                model = model_class(n_features, num_classes=3,
                                   hidden_sizes=[hidden_size, hidden_size // 2, hidden_size // 4],
                                   dropout=dropout)
            elif model_name == 'TransformerClassifier':
                model = model_class(n_features, num_classes=3,
                                   d_model=hidden_size, nhead=4, num_layers=2,
                                   dropout=dropout)
            elif model_name == 'CNN1DClassifier':
                # Reduced channels to prevent OOM - Conv1D creates large intermediate tensors
                model = model_class(n_features, num_classes=3,
                                   channels=[32, 64, 64],  # Reduced from [hidden_size//2, hidden_size, hidden_size*2]
                                   kernel_size=3, dropout=dropout)
            elif model_name == 'TCNClassifier':
                # Reduced channels to prevent OOM
                model = model_class(n_features, num_classes=3,
                                   channels=[32, 64],  # Reduced from [hidden_size, hidden_size*2]
                                   kernel_size=3, dropout=dropout)
            elif model_name == 'WaveNetClassifier':
                # Reduced channels to prevent OOM
                model = model_class(n_features, num_classes=3,
                                   channels=64, num_layers=3,  # Reduced from hidden_size, 4 layers
                                   kernel_size=2, dropout=dropout)
            elif model_name == 'AttentionNetClassifier':
                model = model_class(n_features, num_classes=3,
                                   hidden_size=hidden_size, num_heads=4,
                                   dropout=dropout)
            elif model_name == 'NBeatsClassifier':
                model = model_class(n_features, num_classes=3,
                                   hidden_size=hidden_size, num_blocks=4,
                                   dropout=dropout)
            elif model_name == 'TemporalFusionTransformer':
                model = model_class(n_features, num_classes=3,
                                   hidden_size=hidden_size, num_heads=4, num_layers=2,
                                   dropout=dropout)
            elif model_name == 'EnsembleNNClassifier':
                # EnsembleNN has Conv1D - use smaller hidden_size to prevent OOM
                model = model_class(n_features, num_classes=3,
                                   hidden_size=64, dropout=dropout)  # Fixed at 64
            else:
                # Default for LSTM, GRU, ResidualMLP - they accept hidden_size
                model = model_class(n_features, num_classes=3,
                                   hidden_size=hidden_size, dropout=dropout)

            model = train_pytorch_model(
                model, X_train_scaled, y_train_nn, X_val_scaled, y_val_nn,
                epochs=50, batch_size=batch_size,
                learning_rate=learning_rate, patience=10
            )
            preds = predict_pytorch(model, X_val_scaled)
            pf, stats = calculate_profit_factor_with_stats(preds, returns_val_nn)

            if trial.number % 10 == 0:
                print(f"    Trial {trial.number}: PF={pf:.2f}, "
                      f"TradeRatio={stats['trade_ratio']:.1%}"
                      f"{' [PENALIZED]' if stats['penalized'] else ''}")

            return pf
        except Exception as e:
            print(f"    Trial failed: {e}")
            return 0.5
    return objective


def create_objective(brain, X_train, y_train, X_val, y_val, returns_val, pair, tf, n_features):
    """Dispatcher for all brain types."""
    if brain == 'lgb_optuna':
        return create_lgb_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf)
    elif brain == 'lgb_v2_optuna':
        return create_lgb_v2_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf)
    elif brain == 'xgb_optuna':
        return create_xgb_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf)
    elif brain == 'xgb_v2_optuna':
        return create_xgb_v2_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf)
    elif brain == 'cat_optuna':
        return create_cat_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf)
    elif brain == 'cat_v2_optuna':
        return create_cat_v2_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf)
    elif brain == 'tabnet_optuna':
        return create_tabnet_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf)
    elif brain == 'mlp_optuna':
        return create_dl_objective(MLPClassifier, X_train, y_train, X_val, y_val, returns_val, pair, tf, n_features)
    elif brain == 'lstm_optuna':
        return create_dl_objective(LSTMClassifier, X_train, y_train, X_val, y_val, returns_val, pair, tf, n_features)
    elif brain == 'gru_optuna':
        return create_dl_objective(GRUClassifier, X_train, y_train, X_val, y_val, returns_val, pair, tf, n_features)
    elif brain == 'transformer_optuna':
        return create_dl_objective(TransformerClassifier, X_train, y_train, X_val, y_val, returns_val, pair, tf, n_features)
    elif brain == 'cnn1d_optuna':
        return create_dl_objective(CNN1DClassifier, X_train, y_train, X_val, y_val, returns_val, pair, tf, n_features)
    elif brain == 'tcn_optuna':
        return create_dl_objective(TCNClassifier, X_train, y_train, X_val, y_val, returns_val, pair, tf, n_features)
    elif brain == 'wavenet_optuna':
        return create_dl_objective(WaveNetClassifier, X_train, y_train, X_val, y_val, returns_val, pair, tf, n_features)
    elif brain == 'attention_net_optuna':
        return create_dl_objective(AttentionNetClassifier, X_train, y_train, X_val, y_val, returns_val, pair, tf, n_features)
    elif brain == 'residual_mlp_optuna':
        return create_dl_objective(ResidualMLPClassifier, X_train, y_train, X_val, y_val, returns_val, pair, tf, n_features)
    elif brain == 'ensemble_nn_optuna':
        return create_dl_objective(EnsembleNNClassifier, X_train, y_train, X_val, y_val, returns_val, pair, tf, n_features)
    elif brain == 'nbeats_optuna':
        return create_dl_objective(NBeatsClassifier, X_train, y_train, X_val, y_val, returns_val, pair, tf, n_features)
    elif brain == 'tft_optuna':
        return create_dl_objective(TemporalFusionTransformer, X_train, y_train, X_val, y_val, returns_val, pair, tf, n_features)
    else:
        raise ValueError(f"Unknown brain: {brain}")


# =============================================================================
# TRAIN FINAL MODEL
# =============================================================================
def train_final_model(brain, best_params, X_train, y_train, X_val, y_val, n_features):
    """Train final model with best parameters and class weights."""
    scaler = None

    # Compute sample weights for XGBoost
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    sample_weights = np.array([class_weight_dict[y] for y in y_train])

    if brain in ['lgb_optuna', 'lgb_v2_optuna']:
        # Check GPU availability for LightGBM
        gpu_status = get_gpu_status()
        use_gpu = gpu_status.get('lightgbm', False)

        lgb_params = {
            **best_params,
            'class_weight': 'balanced',
            'random_state': RANDOM_SEED,
            'verbosity': -1,
        }
        if use_gpu:
            lgb_params['device'] = 'gpu'
            lgb_params['gpu_platform_id'] = 0
            lgb_params['gpu_device_id'] = 0
        else:
            lgb_params['n_jobs'] = -1  # Use all CPUs if no GPU

        model = LGBMClassifier(**lgb_params)
        model.fit(X_train, y_train)
        return model, scaler

    elif brain in ['xgb_optuna', 'xgb_v2_optuna']:
        model = XGBClassifier(**best_params, tree_method='hist', device='cuda',
                               random_state=RANDOM_SEED, verbosity=0)
        model.fit(X_train, y_train, sample_weight=sample_weights)  # Use sample weights
        return model, scaler

    elif brain in ['cat_optuna', 'cat_v2_optuna']:
        model = CatBoostClassifier(**best_params, auto_class_weights='Balanced',
                                   task_type='GPU', devices='0',
                                   random_seed=RANDOM_SEED, verbose=False)
        model.fit(X_train, y_train)
        return model, scaler

    elif brain == 'tabnet_optuna':
        if not TABNET_AVAILABLE:
            raise ValueError("TabNet not available")

        # Apply stratified sampling for large datasets (maintains class balance)
        X_train_tab, y_train_tab = sample_for_tabnet(X_train, y_train, TABNET_MAX_SAMPLES)

        model = TabNetModel(
            n_d=best_params.get('n_d', 16),
            n_a=best_params.get('n_a', best_params.get('n_d', 16)),
            n_steps=best_params.get('n_steps', 5),
            gamma=best_params.get('gamma', 1.5),
            n_independent=best_params.get('n_independent', 2),
            n_shared=best_params.get('n_shared', 2),
            lambda_sparse=best_params.get('lambda_sparse', 1e-4),
            mask_type=best_params.get('mask_type', 'sparsemax'),
            device_name='cuda',       # GPU acceleration
            verbose=1,                # Show progress for final training
            seed=RANDOM_SEED,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
        )

        # Optimized training settings for GPU
        model.fit(
            X_train_tab, y_train_tab,
            eval_set=[(X_val, y_val)],
            eval_metric=['accuracy'],
            max_epochs=100,          # More epochs for final model
            patience=15,             # But still early stop
            batch_size=4096,         # Large batch for GPU efficiency
            virtual_batch_size=256,
            num_workers=0,           # Required for Colab
            drop_last=False,
        )
        return model, scaler

    else:
        # Deep Learning models - apply balanced sampling for large datasets
        X_train_nn, y_train_nn = sample_for_neural_network(X_train, y_train, NN_MAX_SAMPLES)

        # Scale after sampling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_nn)
        X_val_scaled = scaler.transform(X_val)

        # Model-specific instantiation - each model has different __init__ params
        hidden_size = best_params.get('hidden_size', 128)
        dropout = best_params.get('dropout', 0.3)

        if brain == 'mlp_optuna':
            # MLPClassifier expects hidden_sizes (list), not hidden_size (int)
            model = MLPClassifier(
                n_features, num_classes=3,
                hidden_sizes=[hidden_size, hidden_size // 2, hidden_size // 4],
                dropout=dropout
            )
        elif brain == 'lstm_optuna':
            model = LSTMClassifier(
                n_features, num_classes=3,
                hidden_size=hidden_size, num_layers=2, dropout=dropout
            )
        elif brain == 'gru_optuna':
            model = GRUClassifier(
                n_features, num_classes=3,
                hidden_size=hidden_size, num_layers=2, dropout=dropout
            )
        elif brain == 'transformer_optuna':
            # TransformerClassifier expects d_model, not hidden_size
            model = TransformerClassifier(
                n_features, num_classes=3,
                d_model=hidden_size, nhead=4, num_layers=2, dropout=dropout
            )
        elif brain == 'cnn1d_optuna':
            # Reduced channels to prevent OOM - Conv1D creates large intermediate tensors
            model = CNN1DClassifier(
                n_features, num_classes=3,
                channels=[32, 64, 64],  # Reduced from [hidden_size//2, hidden_size, hidden_size*2]
                kernel_size=3, dropout=dropout
            )
        elif brain == 'tcn_optuna':
            # Reduced channels to prevent OOM
            model = TCNClassifier(
                n_features, num_classes=3,
                channels=[32, 64],  # Reduced from [hidden_size, hidden_size*2]
                kernel_size=3, dropout=dropout
            )
        elif brain == 'wavenet_optuna':
            # Reduced channels to prevent OOM
            model = WaveNetClassifier(
                n_features, num_classes=3,
                channels=64, num_layers=3, kernel_size=2, dropout=dropout  # Reduced
            )
        elif brain == 'attention_net_optuna':
            model = AttentionNetClassifier(
                n_features, num_classes=3,
                hidden_size=hidden_size, num_heads=4, dropout=dropout
            )
        elif brain == 'residual_mlp_optuna':
            model = ResidualMLPClassifier(
                n_features, num_classes=3,
                hidden_size=hidden_size, num_blocks=3, dropout=dropout
            )
        elif brain == 'ensemble_nn_optuna':
            # EnsembleNN has Conv1D - use smaller hidden_size to prevent OOM
            model = EnsembleNNClassifier(
                n_features, num_classes=3,
                hidden_size=64, dropout=dropout  # Fixed at 64 to prevent OOM
            )
        elif brain == 'nbeats_optuna':
            model = NBeatsClassifier(
                n_features, num_classes=3,
                hidden_size=hidden_size, num_blocks=4, dropout=dropout
            )
        elif brain == 'tft_optuna':
            model = TemporalFusionTransformer(
                n_features, num_classes=3,
                hidden_size=hidden_size, num_heads=4, num_layers=2, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown neural network brain: {brain}")

        model = train_pytorch_model(
            model, X_train_scaled, y_train_nn, X_val_scaled, y_val,
            epochs=100,
            batch_size=best_params.get('batch_size', 128),
            learning_rate=best_params.get('learning_rate', 1e-3),
            patience=20
        )

        return model, scaler


# =============================================================================
# SAVE MODEL
# =============================================================================
@safe_save
def save_trained_model(brain, pair, tf, model, scaler, best_params, best_pf):
    """Save trained model to file (with safe_save retry on Drive errors)."""
    model_key = f"{pair}_{tf}_{brain}"
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)

    # Save params
    params_file = PARAMS_DIR / f"{model_key}_params.json"
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
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    # Save model
    if brain in ['lgb_optuna', 'lgb_v2_optuna', 'xgb_optuna', 'xgb_v2_optuna',
                 'cat_optuna', 'cat_v2_optuna', 'tabnet_optuna']:
        model_file = MODELS_DIR / f"{model_key}.joblib"
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'brain': brain,
            'pair': pair,
            'tf': tf,
            'best_pf': best_pf,
            'target_col': TARGET_COL,
            'returns_col': RETURNS_COL,
        }, model_file)
    else:
        model_file = MODELS_DIR / f"{model_key}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'brain': brain,
            'pair': pair,
            'tf': tf,
            'best_pf': best_pf,
            'n_features': model.input_projection.in_features if hasattr(model, 'input_projection') else None,
            'target_col': TARGET_COL,
            'returns_col': RETURNS_COL,
        }, model_file)

    file_size = model_file.stat().st_size
    print(f"  Saved: {model_file.name} ({file_size/1024:.1f} KB)")

    return model_file


# =============================================================================
# STUDY DIAGNOSTICS
# =============================================================================
def print_study_diagnostics(study, brain):
    """Print Optuna study diagnostics."""
    print(f"\n  OPTUNA DIAGNOSTICS ({brain}):")
    print(f"    Best trial: {study.best_trial.number} / {len(study.trials)}")
    print(f"    Best value (PF): {study.best_value:.4f}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def build_training_queue(completed: list) -> list:
    """
    FIX 5: Build training queue sorted by estimated training time (fastest first).

    Orders by timeframe speed (D1 fastest → M1 slowest), then pair, then brain.
    This maximizes checkpointed progress before potential Colab disconnects.
    """
    tf_order = {tf: i for i, tf in enumerate(TF_SPEED_ORDER)}

    queue = []
    for pair in ALL_PAIRS:
        for tf in ALL_TIMEFRAMES:
            for brain in GPU_BRAINS:
                model_key = f"{pair}_{tf}_{brain}"
                if model_key in completed:
                    continue
                if brain == 'tabnet_optuna' and not TABNET_AVAILABLE:
                    continue
                queue.append({
                    'pair': pair,
                    'tf': tf,
                    'brain': brain,
                    'model_key': model_key,
                    'tf_rank': tf_order.get(tf, 99),
                })

    # Sort: fastest timeframes first, then alphabetical pair, then brain
    queue.sort(key=lambda x: (x['tf_rank'], x['pair'], x['brain']))

    return queue


def main():
    """Main training loop with all 6 checkpoint optimizations."""
    print("=" * 70)
    print("CHAOS V1.0 - GPU TRAINING SCRIPT (19 BRAINS)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # FIX 2: Colab keep-alive
    setup_colab_keepalive()

    print(f"\nConfiguration:")
    print(f"  BASE_DIR: {BASE_DIR}")
    print(f"  TARGET_COL: {TARGET_COL}")
    print(f"  RETURNS_COL: {RETURNS_COL}")
    print(f"  FILE_PATTERN: {FEATURE_FILE_PATTERN}")
    print(f"  DEVICE: {DEVICE}")
    print("=" * 70)

    # Initialize GPU status at startup
    get_gpu_status()
    print("=" * 70)

    total = len(ALL_PAIRS) * len(ALL_TIMEFRAMES) * len(GPU_BRAINS)
    checkpoint = load_checkpoint()
    completed = checkpoint.get('completed', [])

    # Track failed models
    failed_models = []

    # FIX 5: Smart ordering - build sorted queue
    queue = build_training_queue(completed)

    print(f"\nPairs: {len(ALL_PAIRS)} - {ALL_PAIRS}")
    print(f"Timeframes: {len(ALL_TIMEFRAMES)} - {ALL_TIMEFRAMES}")
    print(f"Brains: {len(GPU_BRAINS)}")
    print(f"Total models: {total}")
    print(f"Completed: {len(completed)}")
    print(f"Remaining: {len(queue)}")
    if queue:
        # Show first few in queue to confirm ordering
        print(f"  Next up: {', '.join(q['model_key'] for q in queue[:5])}")
    print("=" * 70)

    log_progress(f"SESSION START: {len(completed)} done, {len(queue)} remaining")

    # FIX 4: Start heartbeat checkpointer
    heartbeat = HeartbeatCheckpointer(HEARTBEAT_INTERVAL)
    heartbeat.start(checkpoint)

    start_time = time.time()

    # Cache loaded features to avoid reloading for same pair+tf
    features_cache = {}

    for idx, item in enumerate(queue):
        pair = item['pair']
        tf = item['tf']
        brain = item['brain']
        model_key = item['model_key']
        model_num = idx + 1

        # Skip if somehow completed between queue build and now
        if model_key in checkpoint.get('completed', []):
            continue

        # Load features (cached per pair+tf)
        cache_key = f"{pair}_{tf}"
        if cache_key not in features_cache:
            try:
                features_cache[cache_key] = load_features(pair, tf)
            except FileNotFoundError:
                print(f"\n[SKIP] {pair}_{tf}: File not found")
                features_cache[cache_key] = None
                continue
            except Exception as e:
                print(f"\n[ERROR] {pair}_{tf}: {e}")
                features_cache[cache_key] = None
                continue

        if features_cache[cache_key] is None:
            continue

        X_train, y_train, X_val, y_val, returns_val, n_features = features_cache[cache_key]

        print(f"\n[{model_num}/{len(queue)}] Training {model_key}...")

        # CRITICAL: Clear GPU memory before each model to prevent OOM
        gc.collect()
        torch.cuda.empty_cache()

        # TabNet uses fewer trials (slower model)
        if brain == 'tabnet_optuna':
            n_trials = TABNET_TRIALS_CONFIG.get(tf, 20)
        else:
            n_trials = TRIALS_CONFIG.get(tf, 50)

        # FIX 1: Check for partial Optuna checkpoint
        partial = load_optuna_checkpoint(model_key)
        if partial:
            n_completed_trials = partial['n_trials_completed']
            print(f"  Found partial checkpoint: {n_completed_trials} trials done, "
                  f"best_pf={partial['best_value']:.4f}")

            # If we already have enough trials, skip Optuna and go straight to final model
            if n_completed_trials >= n_trials:
                print(f"  All {n_trials} trials already done — using cached best params")
                best_params = partial['best_params']
                best_value = partial['best_value']
            else:
                # Run remaining trials
                remaining_trials = n_trials - n_completed_trials
                print(f"  Running {remaining_trials} remaining trials...")
                best_params = None  # Will be set by study below
                best_value = None
        else:
            best_params = None
            best_value = None

        print(f"  Trials: {n_trials}, Early stop: {EARLY_STOPPING_PATIENCE}")

        try:
            # Only run Optuna if we don't have cached best params
            if best_params is None:
                objective = create_objective(
                    brain, X_train, y_train, X_val, y_val,
                    returns_val, pair, tf, n_features
                )

                study = optuna.create_study(
                    direction='maximize',
                    sampler=TPESampler(seed=RANDOM_SEED)
                )

                # If we have a partial checkpoint, enqueue the best params as a warm start
                if partial:
                    remaining_trials = n_trials - partial['n_trials_completed']
                    study.enqueue_trial(partial['best_params'])
                else:
                    remaining_trials = n_trials

                # FIX 1: Add trial-level checkpoint callback alongside early stopping
                study.optimize(
                    objective,
                    n_trials=remaining_trials,
                    callbacks=[
                        EarlyStoppingCallback(EARLY_STOPPING_PATIENCE),
                        OptunaTrialCheckpointCallback(model_key),
                    ],
                    show_progress_bar=True,
                )

                print_study_diagnostics(study, brain)
                best_params = study.best_params
                best_value = study.best_value

            print(f"  Training final model with best params...")
            log_progress(f"FINAL_TRAIN {model_key}: best_pf={best_value:.4f}")

            model, scaler = train_final_model(
                brain, best_params, X_train, y_train,
                X_val, y_val, n_features
            )

            save_trained_model(
                brain, pair, tf, model, scaler,
                best_params, best_value
            )

            # FIX 1: Clean up partial Optuna checkpoint after successful save
            delete_optuna_checkpoint(model_key)

            checkpoint = add_to_checkpoint(checkpoint, model_key)
            heartbeat.update_checkpoint(checkpoint)
            print(f"  Checkpoint: {len(checkpoint['completed'])} models")

            # FIX 6: Log completion
            elapsed = time.time() - start_time
            done = len(checkpoint['completed'])
            remaining = len(queue) - model_num
            log_progress(
                f"COMPLETED {model_key}: pf={best_value:.4f}, "
                f"done={done}, remaining={remaining}, "
                f"elapsed={elapsed/60:.1f}min"
            )

            if done > 0 and remaining > 0:
                avg_time = elapsed / model_num  # Use queue position for avg
                eta = remaining * avg_time
                print(f"  ETA: {eta/60:.1f} minutes remaining")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved.")
            heartbeat.stop()
            log_progress("SESSION INTERRUPTED by user")
            return

        except Exception as e:
            print(f"  [ERROR] {model_key}: {e}")
            import traceback
            traceback.print_exc()
            # Track failed model
            failed_models.append({
                'model_key': model_key,
                'brain': brain,
                'pair': pair,
                'tf': tf,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            log_progress(f"FAILED {model_key}: {str(e)[:100]}")
            continue

    # Stop heartbeat
    heartbeat.stop()

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Models completed: {len(checkpoint['completed'])}")
    print(f"Models failed: {len(failed_models)}")

    # Report failed models
    if failed_models:
        print("\nFAILED MODELS:")
        for fm in failed_models:
            print(f"  - {fm['model_key']}: {fm['error'][:50]}...")

        # Save failed models log
        failed_log_path = MODELS_DIR / "failed_models.json"
        with open(failed_log_path, 'w') as f:
            json.dump(failed_models, f, indent=2)
        print(f"\nFailed models log saved to: {failed_log_path}")

    log_progress(
        f"SESSION END: {len(checkpoint['completed'])} done, "
        f"{len(failed_models)} failed, {elapsed/60:.1f}min total"
    )
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
# GPU VERIFICATION TEST - RUN BEFORE TRAINING
# =============================================================================
def verify_gpu_models():
    """
    Verify GPU support for all frameworks before training.
    Returns True if all tests pass, False otherwise.
    """
    print("=" * 70)
    print("GPU FRAMEWORK VERIFICATION TEST")
    print("=" * 70)

    all_passed = True
    results = {}

    # 1. PyTorch GPU
    print("\n[1/4] PyTorch CUDA...")
    if torch.cuda.is_available():
        try:
            device = torch.device('cuda')
            x = torch.randn(100, 10).to(device)
            y = torch.randint(0, 3, (100,)).to(device)
            model = nn.Linear(10, 3).to(device)
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            print(f"  PASSED - Device: {torch.cuda.get_device_name(0)}")
            print(f"           Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            results['pytorch'] = 'GPU'
        except Exception as e:
            print(f"  FAILED - {e}")
            results['pytorch'] = 'FAILED'
            all_passed = False
    else:
        print("  NOT AVAILABLE - No CUDA device")
        results['pytorch'] = 'CPU'

    # 2. LightGBM GPU
    print("\n[2/4] LightGBM GPU...")
    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(device='gpu', gpu_platform_id=0, gpu_device_id=0,
                               n_estimators=5, num_leaves=31, verbosity=-1)
        X_test = np.random.randn(100, 10).astype('float32')
        y_test = np.random.randint(0, 3, 100)
        model.fit(X_test, y_test)
        preds = model.predict(X_test)
        print(f"  PASSED - GPU training successful")
        results['lightgbm'] = 'GPU'
    except Exception as e:
        print(f"  FAILED - {e}")
        print("  FALLBACK - Will use CPU for LightGBM")
        results['lightgbm'] = 'CPU'

    # 3. XGBoost GPU (2.0+ syntax)
    print("\n[3/4] XGBoost GPU (tree_method='hist', device='cuda')...")
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(tree_method='hist', device='cuda',
                              n_estimators=5, max_depth=3, verbosity=0)
        X_test = np.random.randn(100, 10).astype('float32')
        y_test = np.random.randint(0, 3, 100)
        model.fit(X_test, y_test)
        preds = model.predict(X_test)
        print(f"  PASSED - GPU training successful")
        results['xgboost'] = 'GPU'
    except Exception as e:
        print(f"  FAILED - {e}")
        results['xgboost'] = 'FAILED'
        all_passed = False

    # 4. CatBoost GPU
    print("\n[4/4] CatBoost GPU...")
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(task_type='GPU', devices='0',
                                   iterations=5, depth=3, verbose=0)
        X_test = np.random.randn(100, 10).astype('float32')
        y_test = np.random.randint(0, 3, 100)
        model.fit(X_test, y_test)
        preds = model.predict(X_test)
        print(f"  PASSED - GPU training successful")
        results['catboost'] = 'GPU'
    except Exception as e:
        print(f"  FAILED - {e}")
        results['catboost'] = 'FAILED'
        all_passed = False

    # 5. TabNet (if available)
    print("\n[5/5] TabNet GPU...")
    if TABNET_AVAILABLE:
        try:
            model = TabNetModel(device_name='cuda', verbose=0, seed=42)
            X_test = np.random.randn(100, 10).astype('float32')
            y_test = np.random.randint(0, 3, 100)
            model.fit(X_test, y_test, max_epochs=1, patience=1, batch_size=32)
            preds = model.predict(X_test)
            print(f"  PASSED - GPU training successful")
            results['tabnet'] = 'GPU'
        except Exception as e:
            print(f"  FAILED - {e}")
            results['tabnet'] = 'FAILED'
    else:
        print("  SKIPPED - TabNet not installed")
        results['tabnet'] = 'NOT_INSTALLED'

    # Summary
    print("\n" + "=" * 70)
    print("GPU VERIFICATION SUMMARY")
    print("=" * 70)
    for framework, status in results.items():
        icon = "OK" if status == 'GPU' else ("CPU" if status == 'CPU' else "X")
        print(f"  [{icon}] {framework}: {status}")

    gpu_count = sum(1 for s in results.values() if s == 'GPU')
    print(f"\n  GPU frameworks: {gpu_count}/{len(results)}")

    if all_passed:
        print("\n  ALL CRITICAL TESTS PASSED - Ready for training!")
    else:
        print("\n  WARNING: Some tests failed - Check configuration")

    print("=" * 70)
    return all_passed


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # 1. Verify PF calculation first
    verify_pf_calculation()
    print()

    # 2. Self-test config
    print("Running config self-test...")

    assert TARGET_COL == 'target_3class_8', f"Wrong TARGET_COL: {TARGET_COL}"
    assert RETURNS_COL == 'target_return_8', f"Wrong RETURNS_COL: {RETURNS_COL}"
    assert len(ALL_PAIRS) == 9, f"Wrong number of pairs: {len(ALL_PAIRS)}"
    assert len(ALL_TIMEFRAMES) == 9, f"Wrong number of timeframes: {len(ALL_TIMEFRAMES)}"
    assert len(GPU_BRAINS) == 19, f"Wrong number of GPU brains: {len(GPU_BRAINS)}"
    assert 'nbeats_optuna' in GPU_BRAINS, "nbeats_optuna missing from GPU_BRAINS"
    assert 'tft_optuna' in GPU_BRAINS, "tft_optuna missing from GPU_BRAINS"
    assert 'W1' in ALL_TIMEFRAMES, "W1 missing from ALL_TIMEFRAMES"
    assert 'MN1' in ALL_TIMEFRAMES, "MN1 missing from ALL_TIMEFRAMES"

    print("All config self-tests passed!")
    print()

    # 3. Verify GPU support for all frameworks
    gpu_ok = verify_gpu_models()
    print()

    if not gpu_ok:
        print("WARNING: Some GPU tests failed. Training will continue but may use CPU fallback.")
        print("Press Ctrl+C within 5 seconds to abort, or wait to continue...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nAborted by user.")
            sys.exit(1)
        print("Continuing with training...\n")

    # 4. Run main training
    main()
