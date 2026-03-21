"""
CHAOS V1.0 - COLAB NOTEBOOK CELL UPDATES
========================================
Copy these cells to CHAOS_FULL_PIPELINE_PHASE2_CLEAN.ipynb

CONFIGURATION NOW MATCHES LOCAL:
  - TARGET_COL = 'target_3class_8'
  - RETURNS_COL = 'target_return_8'
  - FILE_PATTERN = '{pair}_{tf}_phase2.parquet'
"""

# =============================================================================
# CELL 2: IMPORTS (add datetime)
# =============================================================================
CELL_2_IMPORTS = '''
# CELL 2: IMPORTS
import os
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime  # CRITICAL: Must be imported
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Check for GPU
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
'''

# =============================================================================
# CELL 3: CONFIGURATION (EXPLICIT, NO DYNAMIC)
# =============================================================================
CELL_3_CONFIG = '''
# CELL 3: CONFIGURATION - EXPLICIT, NO DYNAMIC SELECTION
# ========================================================
# GOOGLE DRIVE SYNCED WITH LOCAL WINDOWS

# PATHS
BASE_DIR = Path("/content/drive/MyDrive/chaos_v1.0")
FEATURES_PATH = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "optuna_checkpoint.json"
PARAMS_DIR = MODELS_DIR / "optuna_params"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_DIR.mkdir(parents=True, exist_ok=True)

# EXPLICIT COLUMN NAMES - VERIFIED FROM DATA
TARGET_COL = 'target_3class_8'      # 3-class labels for 8-bar horizon
RETURNS_COL = 'target_return_8'     # Returns for 8-bar horizon

# FEATURE FILE PATTERN
FEATURE_FILE_PATTERN = "{pair}_{tf}_phase2.parquet"

# RANDOM SEED
RANDOM_SEED = 42

# Grid configuration
ALL_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
             'USDCHF', 'NZDUSD', 'EURJPY', 'GBPJPY']

ALL_TIMEFRAMES = ['M15', 'M30', 'H1', 'H4', 'D1']  # W1, MN1 not available

# Brain names
ALL_BRAINS = ['lgb_optuna', 'xgb_optuna', 'cat_optuna',
              'lgb_v2_optuna', 'xgb_v2_optuna', 'cat_v2_optuna',
              'tabnet_optuna', 'mlp_optuna', 'lstm_optuna',
              'gru_optuna', 'transformer_optuna', 'cnn_lstm_optuna',
              'ensemble_nn_optuna', 'wavenet_optuna', 'tcn_optuna',
              'nbeats_optuna', 'tft_optuna']

# RF/ET trained locally
LOCAL_BRAINS = ['rf_optuna', 'et_optuna']

# Trials by timeframe
TRIALS_BY_TIMEFRAME = {
    'M15': 500, 'M30': 500, 'H1': 500,
    'H4': 400, 'D1': 300, 'W1': 200, 'MN1': 100
}

# Early stopping
EARLY_STOPPING_PATIENCE = 100

print("Configuration loaded:")
print(f"  TARGET_COL = '{TARGET_COL}'")
print(f"  RETURNS_COL = '{RETURNS_COL}'")
print(f"  FILE_PATTERN = '{FEATURE_FILE_PATTERN}'")
'''

# =============================================================================
# CELL 5: prepare_data() - EXPLICIT COLUMNS
# =============================================================================
CELL_5_PREPARE_DATA = '''
# CELL 5: prepare_data() - EXPLICIT COLUMNS, NO DYNAMIC SELECTION
def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature columns - exclude all targets, returns, metadata."""
    exclude_patterns = ['target_', 'return', 'Open', 'High', 'Low', 'Close',
                        'Volume', 'timestamp', 'date', 'pair', 'symbol', 'tf',
                        'Bid_', 'Ask_', 'Spread_']

    feature_cols = []
    for col in df.columns:
        if any(p in col for p in exclude_patterns):
            continue
        if df[col].dtype not in ['int64', 'int32', 'int8', 'float64', 'float32', 'Float64']:
            continue
        feature_cols.append(col)

    return feature_cols


def prepare_data(df: pd.DataFrame) -> Tuple:
    """Prepare data with EXPLICIT column names - no dynamic selection."""

    # VALIDATE EXPLICIT COLUMNS EXIST
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found! "
                        f"Available: {[c for c in df.columns if 'target' in c.lower()]}")
    if RETURNS_COL not in df.columns:
        raise ValueError(f"Returns column '{RETURNS_COL}' not found! "
                        f"Available: {[c for c in df.columns if 'return' in c.lower()]}")

    # Get feature columns
    feature_cols = get_feature_columns(df)

    # Prepare X
    X = df[feature_cols].values.astype('float32')
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # EXPLICIT target
    y = df[TARGET_COL].values.copy()
    y = np.nan_to_num(y, nan=0.0)
    if y.min() < 0:  # Convert -1,0,1 to 0,1,2
        y = y + 1
    y = y.astype('int64')

    # EXPLICIT returns
    returns = df[RETURNS_COL].values.copy()
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    # 80/20 chronological split
    split_idx = int(len(X) * 0.8)

    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    returns_val = returns[split_idx:]

    print(f"  Features: {len(feature_cols)}, Train: {len(X_train):,}, Val: {len(X_val):,}")
    print(f"  Target: {TARGET_COL}, Returns: {RETURNS_COL}")

    return X_train, y_train, X_val, y_val, returns_val, len(feature_cols)
'''

# =============================================================================
# CELL 6: CHECKPOINT MANAGEMENT - LIST FORMAT
# =============================================================================
CELL_6_CHECKPOINT = '''
# CELL 6: CHECKPOINT MANAGEMENT - LIST FORMAT (append), NOT DICT
def load_checkpoint() -> dict:
    """Load checkpoint or create new one."""
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, 'r') as f:
                cp = json.load(f)
            # Ensure 'completed' is a list
            if 'completed' not in cp or not isinstance(cp['completed'], list):
                cp['completed'] = []
            return cp
        except json.JSONDecodeError:
            print("WARNING: Checkpoint corrupted, starting fresh")
            return {"completed": []}
    return {"completed": []}


def save_checkpoint(checkpoint: dict):
    """Save checkpoint with timestamp."""
    checkpoint['timestamp'] = datetime.now().isoformat()
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def add_to_checkpoint(checkpoint: dict, model_key: str) -> dict:
    """Add model to checkpoint - LIST FORMAT (append), NOT DICT."""
    if model_key not in checkpoint['completed']:
        checkpoint['completed'].append(model_key)  # LIST APPEND, NOT DICT
    save_checkpoint(checkpoint)
    return checkpoint
'''

# =============================================================================
# VERIFICATION CELL - RUN THIS FIRST
# =============================================================================
VERIFICATION_CELL = '''
# VERIFICATION CELL - RUN THIS BEFORE TRAINING
print("="*70)
print("CHAOS V1.0 - COLAB VERIFICATION")
print("="*70)

# Load test file
test_file = FEATURES_PATH / "EURUSD_H1_phase2.parquet"
assert test_file.exists(), f"Test file not found: {test_file}"

df = pd.read_parquet(test_file)
print(f"Loaded: {len(df):,} rows × {len(df.columns)} columns")

# Verify columns
assert TARGET_COL in df.columns, f"MISSING: {TARGET_COL}"
assert RETURNS_COL in df.columns, f"MISSING: {RETURNS_COL}"

# Show values
target_vals = sorted(df[TARGET_COL].unique())
print(f"Target values: {target_vals}")

returns = df[RETURNS_COL]
print(f"Returns: min={returns.min():.4f}, max={returns.max():.4f}")

print("\\n✓ All verified!")
print("="*70)
'''

print("COLAB CELL UPDATES READY")
print("Copy these cells to the notebook:")
print("  - CELL_2_IMPORTS")
print("  - CELL_3_CONFIG")
print("  - CELL_5_PREPARE_DATA")
print("  - CELL_6_CHECKPOINT")
print("  - VERIFICATION_CELL (run first)")
