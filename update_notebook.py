"""
Update Colab notebook with correct configuration.
PRESERVES ALL EXISTING CODE - only updates specific sections.
"""
import json
from datetime import datetime
from pathlib import Path

nb_path = Path(r"G:\My Drive\chaos_v1.0\CHAOS_FULL_PIPELINE_PHASE2_CLEAN.ipynb")

# Create backup
backup_path = nb_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.ipynb')

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(backup_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print(f"Backup created: {backup_path.name}")

# Count before
before_lines = sum(len(c['source']) if isinstance(c['source'], list) else 1
                   for c in nb['cells'] if c['cell_type'] == 'code')
print(f"Lines before: {before_lines}")

# ===========================================================================
# UPDATE CELL 2: Add datetime import
# ===========================================================================
cell2_src = ''.join(nb['cells'][2]['source']) if isinstance(nb['cells'][2]['source'], list) else nb['cells'][2]['source']

if 'from datetime import datetime' not in cell2_src:
    # Add after 'from pathlib import Path' or at end of imports
    if 'from pathlib import Path' in cell2_src:
        cell2_src = cell2_src.replace(
            'from pathlib import Path',
            'from pathlib import Path\nfrom datetime import datetime'
        )
    else:
        cell2_src = cell2_src.rstrip() + '\nfrom datetime import datetime\n'
    nb['cells'][2]['source'] = [cell2_src]
    print("Cell 2: Added datetime import")

# ===========================================================================
# UPDATE CELL 3: Add TARGET_COL, RETURNS_COL, FEATURE_FILE_PATTERN
# ===========================================================================
cell3_src = ''.join(nb['cells'][3]['source']) if isinstance(nb['cells'][3]['source'], list) else nb['cells'][3]['source']

# Add configuration if not present
config_block = """
# EXPLICIT COLUMN CONFIGURATION - NO DYNAMIC SELECTION
TARGET_COL = 'target_3class_8'
RETURNS_COL = 'target_return_8'
FEATURE_FILE_PATTERN = "{pair}_{tf}_features.parquet"
RANDOM_SEED = 42
"""

if 'TARGET_COL' not in cell3_src:
    # Add after DEVICE line
    if 'DEVICE = ' in cell3_src:
        parts = cell3_src.split('DEVICE = ')
        # Find end of DEVICE line
        device_line_end = parts[1].find('\n') + 1
        cell3_src = parts[0] + 'DEVICE = ' + parts[1][:device_line_end] + config_block + parts[1][device_line_end:]
    else:
        cell3_src = cell3_src.rstrip() + '\n' + config_block
    nb['cells'][3]['source'] = [cell3_src]
    print("Cell 3: Added TARGET_COL, RETURNS_COL, FEATURE_FILE_PATTERN")

# ===========================================================================
# UPDATE CELL 5: Replace prepare_data with explicit version
# ===========================================================================
new_cell5 = '''# =============================================================================
# CELL 5: FEATURE LOADING - EXPLICIT COLUMNS
# =============================================================================

def load_features(pair: str, tf: str) -> pd.DataFrame:
    """Load feature file for a pair/timeframe combination."""
    fpath = FEATURES_PATH / f'{pair}_{tf}_features.parquet'
    if not fpath.exists():
        raise FileNotFoundError(f"Features not found: {fpath}")
    return pd.read_parquet(fpath)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get safe feature columns, excluding targets and metadata."""
    EXCLUDE_PATTERNS = ['target_', 'return', 'forward', 'future', 'mfe', 'mae']
    EXCLUDE_EXACT = ['Open', 'High', 'Low', 'Close', 'Volume', 'pair', 'timeframe',
                     'bar_time', 'timestamp', 'date', 'symbol', 'tf']

    feature_cols = []
    for col in df.columns:
        if any(p in col for p in EXCLUDE_PATTERNS):
            continue
        if col in EXCLUDE_EXACT:
            continue
        if df[col].dtype not in ['int64', 'int32', 'int8', 'float64', 'float32', 'float16', 'Float64']:
            continue
        feature_cols.append(col)

    return feature_cols


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Prepare training data with EXPLICIT column names.
    NO DYNAMIC SELECTION. Uses TARGET_COL and RETURNS_COL.
    """
    # VALIDATE EXPLICIT COLUMNS
    if TARGET_COL not in df.columns:
        raise ValueError(f"CRITICAL: {TARGET_COL} not found! Available: {[c for c in df.columns if 'target' in c.lower()]}")
    if RETURNS_COL not in df.columns:
        raise ValueError(f"CRITICAL: {RETURNS_COL} not found! Available: {[c for c in df.columns if 'return' in c.lower()]}")

    # Get feature columns
    feature_cols = get_feature_columns(df)

    # Features
    X = df[feature_cols].values.astype('float32')
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # EXPLICIT target - target_3class_8
    y = df[TARGET_COL].values.copy()
    if y.min() == -1:  # Convert -1,0,1 to 0,1,2
        y = y + 1
    y = y.astype('int64')

    # EXPLICIT returns - target_return_8
    returns = df[RETURNS_COL].values.copy()
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    # 80/20 chronological split
    split_idx = int(len(X) * 0.8)

    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    returns_val = returns[split_idx:]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Target: {TARGET_COL}, Returns: {RETURNS_COL}")
    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}")

    return X_train, y_train, X_val, y_val, returns_val, len(feature_cols)


print("Feature loading functions defined (EXPLICIT COLUMNS)")
'''

nb['cells'][5]['source'] = [new_cell5]
print("Cell 5: Replaced prepare_data with explicit column version")

# ===========================================================================
# UPDATE CELL 6: Fix checkpoint to use LIST not DICT
# ===========================================================================
new_cell6 = '''# =============================================================================
# CELL 6: CHECKPOINT MANAGEMENT - LIST FORMAT (NOT DICT)
# =============================================================================

CHECKPOINT_PATH = MODELS_PATH / 'optuna_checkpoint.json'

def load_checkpoint() -> dict:
    """Load checkpoint - returns dict with 'completed' as LIST."""
    try:
        with open(CHECKPOINT_PATH, 'r') as f:
            checkpoint = json.load(f)
        # Ensure completed is a list
        if not isinstance(checkpoint.get('completed'), list):
            checkpoint['completed'] = []
        return checkpoint
    except FileNotFoundError:
        return {'completed': []}
    except json.JSONDecodeError:
        print("WARNING: Checkpoint file corrupted, starting fresh")
        return {'completed': []}


def save_checkpoint(checkpoint: dict):
    """Save checkpoint with verification."""
    # Ensure completed is list
    if not isinstance(checkpoint.get('completed'), list):
        checkpoint['completed'] = []

    # Create backup first
    if CHECKPOINT_PATH.exists():
        backup_path = CHECKPOINT_PATH.with_suffix('.json.bak')
        import shutil
        shutil.copy(CHECKPOINT_PATH, backup_path)

    # Save new checkpoint
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    # Verify save
    with open(CHECKPOINT_PATH, 'r') as f:
        verify = json.load(f)

    saved_count = len(verify.get('completed', []))
    expected_count = len(checkpoint.get('completed', []))
    if saved_count != expected_count:
        raise ValueError(f"Checkpoint verification failed! Saved {saved_count}, expected {expected_count}")

    return True


def is_model_completed(pair: str, tf: str, brain: str, checkpoint: dict) -> bool:
    """Check if model already trained."""
    model_key = f"{pair}_{tf}_{brain}"
    return model_key in checkpoint.get('completed', [])


def add_to_checkpoint(checkpoint: dict, model_key: str) -> dict:
    """Add model to checkpoint using LIST APPEND (not dict assignment)."""
    if model_key not in checkpoint['completed']:
        checkpoint['completed'].append(model_key)  # LIST APPEND
    save_checkpoint(checkpoint)
    return checkpoint


def mark_model_completed(pair: str, tf: str, brain: str, best_pf: float,
                         best_params: dict, checkpoint: dict) -> dict:
    """Mark model as completed in checkpoint."""
    model_key = f"{pair}_{tf}_{brain}"

    # Add to completed LIST
    if model_key not in checkpoint['completed']:
        checkpoint['completed'].append(model_key)  # LIST APPEND

    # Store results separately for quick lookup
    if 'results' not in checkpoint:
        checkpoint['results'] = {}

    checkpoint['results'][model_key] = {
        'best_pf': float(best_pf),
        'best_params': best_params,
        'timestamp': datetime.now().isoformat(),
    }

    save_checkpoint(checkpoint)
    return checkpoint


print("Checkpoint Management loaded (LIST FORMAT)")
'''

nb['cells'][6]['source'] = [new_cell6]
print("Cell 6: Fixed checkpoint to use LIST format")

# ===========================================================================
# SAVE UPDATED NOTEBOOK
# ===========================================================================
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

# Count after
with open(nb_path, 'r', encoding='utf-8') as f:
    nb_verify = json.load(f)

after_lines = sum(len(c['source']) if isinstance(c['source'], list) else 1
                  for c in nb_verify['cells'] if c['cell_type'] == 'code')

print(f"\nLines after: {after_lines}")
print(f"Change: {after_lines - before_lines:+d} lines")

# Verify brain types still present
all_source = ''
for cell in nb_verify['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        all_source += src + '\n'

brains = []
for b in ['lgb', 'xgb', 'cat', 'tabnet', 'mlp', 'lstm', 'gru', 'transformer',
          'cnn', 'tcn', 'wavenet', 'nbeats', 'tft', 'ensemble', 'rf', 'et']:
    if b in all_source.lower():
        brains.append(b)

print(f"\nBrain types after: {brains}")
print(f"Count: {len(brains)}")

# Verify key items
print("\nVERIFICATION:")
print(f"  TARGET_COL present: {'TARGET_COL' in all_source}")
print(f"  RETURNS_COL present: {'RETURNS_COL' in all_source}")
print(f"  datetime import present: {'from datetime import datetime' in all_source}")
print(f"  append( used: {'append(' in all_source}")

print("\n" + "=" * 60)
print("NOTEBOOK UPDATED SUCCESSFULLY")
print("=" * 60)
