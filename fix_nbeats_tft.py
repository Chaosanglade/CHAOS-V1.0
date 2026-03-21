"""
RE-ADD NBEATS AND TFT - COMPLETE FIX
=====================================
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

# =============================================================================
# FIND AND UPDATE CELLS
# =============================================================================

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue

    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    modified = False

    # =========================================================================
    # 1. FIX GPU_BRAINS LIST
    # =========================================================================
    if "GPU_BRAINS = [" in src and "'lgb_optuna'" in src and "'nbeats_optuna'" not in src:
        old_list = """GPU_BRAINS = [
    # Gradient Boosting
    'lgb_optuna', 'xgb_optuna', 'cat_optuna',
    # Gradient Boosting V2 (more regularization)
    'lgb_v2_optuna', 'xgb_v2_optuna', 'cat_v2_optuna',
    # Neural Networks
    'tabnet_optuna', 'mlp_optuna',
    # Deep Learning
    'lstm_optuna', 'gru_optuna', 'transformer_optuna',
    'cnn1d_optuna', 'tcn_optuna', 'wavenet_optuna',
    'attention_net_optuna', 'residual_mlp_optuna',
    'ensemble_nn_optuna',
]"""

        new_list = """GPU_BRAINS = [
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
    'nbeats_optuna', 'tft_optuna',
]"""

        if old_list in src:
            src = src.replace(old_list, new_list)
            modified = True
            print(f"Cell {i}: Fixed GPU_BRAINS list (added nbeats_optuna, tft_optuna)")
        else:
            # Try alternate format
            if "'ensemble_nn_optuna',\n]" in src:
                src = src.replace(
                    "'ensemble_nn_optuna',\n]",
                    "'ensemble_nn_optuna',\n    # Time Series Specialized (2)\n    'nbeats_optuna', 'tft_optuna',\n]"
                )
                modified = True
                print(f"Cell {i}: Fixed GPU_BRAINS list (alternate format)")

    # Also update ALL_BRAINS comment
    if "ALL_BRAINS = GPU_BRAINS + CPU_BRAINS  # 19 total" in src:
        src = src.replace(
            "ALL_BRAINS = GPU_BRAINS + CPU_BRAINS  # 19 total",
            "ALL_BRAINS = GPU_BRAINS + CPU_BRAINS  # 21 total"
        )
        modified = True
        print(f"Cell {i}: Updated ALL_BRAINS comment (19 -> 21)")

    # =========================================================================
    # 2. ADD NBEATS AND TFT CLASSES (after EnsembleNNClassifier)
    # =========================================================================
    if "class EnsembleNNClassifier" in src and "class NBeatsBlock" not in src:
        nbeats_tft_code = '''


# =============================================================================
# N-BEATS: Neural Basis Expansion Analysis
# =============================================================================

class NBeatsBlock(nn.Module):
    """N-BEATS block with backcast and forecast."""
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
            nn.ReLU(),
        )
        self.theta_b = nn.Linear(hidden_size, theta_size)
        self.theta_f = nn.Linear(hidden_size, theta_size)
        self.backcast = nn.Linear(theta_size, input_size)
        self.forecast = nn.Linear(theta_size, theta_size)

    def forward(self, x):
        h = self.fc(x)
        return self.backcast(self.theta_b(h)), self.forecast(self.theta_f(h))


class NBeatsClassifier(nn.Module):
    """N-BEATS classifier for time series."""
    def __init__(self, n_features, n_classes=3, hidden_size=256, num_blocks=4,
                 dropout=0.3, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(n_features, hidden_size // 4, hidden_size)
            for _ in range(num_blocks)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        residuals = x
        forecast = None
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = block_forecast if forecast is None else forecast + block_forecast
        return self.classifier(forecast)


# =============================================================================
# TFT: Temporal Fusion Transformer
# =============================================================================

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
    """Temporal Fusion Transformer for classification."""
    def __init__(self, n_features, n_classes=3, hidden_size=128, num_heads=4,
                 num_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        # Ensure hidden_size divisible by num_heads
        if hidden_size % num_heads != 0:
            hidden_size = ((hidden_size // num_heads) + 1) * num_heads

        self.input_grn = GatedResidualNetwork(n_features, hidden_size, hidden_size, dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads,
                                               dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.output_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        h = self.input_grn(x)
        h = h.unsqueeze(1)
        lstm_out, _ = self.lstm(h)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        h = self.attention_norm(lstm_out + attn_out)
        h = self.output_grn(h.squeeze(1))
        return self.classifier(h)


print("N-BEATS and TFT Architectures loaded")
'''
        src = src.rstrip() + nbeats_tft_code
        modified = True
        print(f"Cell {i}: Added NBeatsClassifier and TemporalFusionTransformer")

    # =========================================================================
    # 3. ADD OBJECTIVES
    # =========================================================================
    if "def create_ensemble_nn_objective" in src and "def create_nbeats_objective" not in src:
        objectives_code = '''

def create_nbeats_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf):
    """N-BEATS objective."""
    return create_dl_objective(NBeatsClassifier, X_train, y_train, X_val, y_val, returns_val, pair, tf)

def create_tft_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf):
    """TFT objective."""
    return create_dl_objective(TemporalFusionTransformer, X_train, y_train, X_val, y_val, returns_val, pair, tf)


print("N-BEATS and TFT Objectives loaded")
'''
        src = src.rstrip() + objectives_code
        modified = True
        print(f"Cell {i}: Added create_nbeats_objective and create_tft_objective")

    # =========================================================================
    # 4. UPDATE DISPATCHER
    # =========================================================================
    if "def create_objective" in src and "dispatcher = {" in src:
        if "'nbeats_optuna'" not in src:
            # Add to dispatcher
            old_pattern = "'ensemble_nn_optuna': create_ensemble_nn_objective,"
            new_pattern = """'ensemble_nn_optuna': create_ensemble_nn_objective,

        # Time Series Specialized (2)
        'nbeats_optuna': create_nbeats_objective,
        'tft_optuna': create_tft_objective,"""

            if old_pattern in src:
                src = src.replace(old_pattern, new_pattern)
                modified = True
                print(f"Cell {i}: Updated dispatcher with nbeats_optuna and tft_optuna")

    # =========================================================================
    # 5. UPDATE train_final_model
    # =========================================================================
    if "def train_final_model" in src or "brain == 'ensemble_nn_optuna'" in src:
        if "brain == 'nbeats_optuna'" not in src:
            old_handler = """elif brain == 'ensemble_nn_optuna':
        model = EnsembleNNClassifier(n_features=n_features, **best_params).to(DEVICE)
    else:
        raise ValueError(f"Unknown PyTorch brain: {brain}")"""

            new_handler = """elif brain == 'ensemble_nn_optuna':
        model = EnsembleNNClassifier(n_features=n_features, **best_params).to(DEVICE)
    elif brain == 'nbeats_optuna':
        model = NBeatsClassifier(n_features=n_features, **best_params).to(DEVICE)
    elif brain == 'tft_optuna':
        model = TemporalFusionTransformer(n_features=n_features, **best_params).to(DEVICE)
    else:
        raise ValueError(f"Unknown PyTorch brain: {brain}")"""

            if old_handler in src:
                src = src.replace(old_handler, new_handler)
                modified = True
                print(f"Cell {i}: Updated train_final_model with NBEATS and TFT handlers")

    # Save if modified
    if modified:
        nb['cells'][i]['source'] = [src]

# =============================================================================
# SAVE NOTEBOOK
# =============================================================================
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("\n" + "=" * 70)
print("NOTEBOOK SAVED")
print("=" * 70)

# =============================================================================
# VERIFICATION
# =============================================================================
with open(nb_path, 'r', encoding='utf-8') as f:
    nb_verify = json.load(f)

all_source = ''
for cell in nb_verify['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        all_source += src + '\n'

print("\nVERIFICATION:")
print("-" * 70)

checks = [
    ("'nbeats_optuna' in GPU_BRAINS", "'nbeats_optuna'" in all_source),
    ("'tft_optuna' in GPU_BRAINS", "'tft_optuna'" in all_source),
    ("class NBeatsBlock", "class NBeatsBlock" in all_source),
    ("class NBeatsClassifier", "class NBeatsClassifier" in all_source),
    ("class GatedResidualNetwork", "class GatedResidualNetwork" in all_source),
    ("class TemporalFusionTransformer", "class TemporalFusionTransformer" in all_source),
    ("def create_nbeats_objective", "def create_nbeats_objective" in all_source),
    ("def create_tft_objective", "def create_tft_objective" in all_source),
    ("nbeats_optuna in dispatcher", "'nbeats_optuna': create_nbeats_objective" in all_source),
    ("tft_optuna in dispatcher", "'tft_optuna': create_tft_objective" in all_source),
    ("nbeats_optuna in train_final_model", "brain == 'nbeats_optuna'" in all_source),
    ("tft_optuna in train_final_model", "brain == 'tft_optuna'" in all_source),
    ("def load_checkpoint", "def load_checkpoint" in all_source),
    ("TARGET_COL = 'target_3class_8'", "TARGET_COL = 'target_3class_8'" in all_source),
    ("RETURNS_COL = 'target_return_8'", "RETURNS_COL = 'target_return_8'" in all_source),
]

all_pass = True
for name, result in checks:
    status = "[OK]" if result else "[FAIL]"
    print(f"  {status} {name}")
    if not result:
        all_pass = False

print("-" * 70)
if all_pass:
    print("STATUS: ALL CHECKS PASSED")
else:
    print("STATUS: SOME CHECKS FAILED")

# Count brains
gpu_count = 0
for b in ['lgb_optuna', 'xgb_optuna', 'cat_optuna', 'lgb_v2_optuna', 'xgb_v2_optuna',
          'cat_v2_optuna', 'tabnet_optuna', 'mlp_optuna', 'lstm_optuna', 'gru_optuna',
          'transformer_optuna', 'cnn1d_optuna', 'tcn_optuna', 'wavenet_optuna',
          'attention_net_optuna', 'residual_mlp_optuna', 'ensemble_nn_optuna',
          'nbeats_optuna', 'tft_optuna']:
    if f"'{b}'" in all_source:
        gpu_count += 1

print(f"\nGPU_BRAINS count: {gpu_count}")
print(f"Expected: 19")
print(f"Total models: 9 x 7 x 21 = 1,323")
print("=" * 70)
