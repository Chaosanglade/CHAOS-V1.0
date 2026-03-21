"""
ADD NBEATS AND TFT ARCHITECTURES TO COLAB NOTEBOOK
===================================================
Adds 2 new GPU brains: nbeats_optuna, tft_optuna
Total GPU brains: 17 -> 19
Total models: 9 pairs x 7 TFs x 21 brains = 1,323
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
# 1. UPDATE CELL 13: GPU_BRAINS (add nbeats_optuna, tft_optuna)
# =============================================================================
cell13_src = ''.join(nb['cells'][13]['source']) if isinstance(nb['cells'][13]['source'], list) else nb['cells'][13]['source']

# Replace GPU_BRAINS definition
old_gpu_brains = """# GPU brains (17) - trained on Colab
GPU_BRAINS = [
    # Gradient Boosting
    'lgb_optuna', 'xgb_optuna', 'cat_optuna',
    # Gradient Boosting V2 (more regularization)
    'lgb_v2_optuna', 'xgb_v2_optuna', 'cat_v2_optuna',
    # Neural Networks
    'tabnet_optuna', 'mlp_optuna',
    # Deep Learning
    'lstm_optuna', 'gru_optuna', 'transformer_optuna',
    'cnn1d_optuna', 'tcn_optuna', 'wavenet_optuna',
    'attention_net_optuna', 'residual_mlp_optuna', 'ensemble_nn_optuna',
]"""

new_gpu_brains = """# GPU brains (19) - trained on Colab
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
    # Time Series Specialized (2) - NEW
    'nbeats_optuna', 'tft_optuna',
]"""

if 'nbeats_optuna' not in cell13_src:
    cell13_src = cell13_src.replace(old_gpu_brains, new_gpu_brains)
    # Also update ALL_BRAINS comment
    cell13_src = cell13_src.replace(
        "ALL_BRAINS = GPU_BRAINS + CPU_BRAINS  # 19 total",
        "ALL_BRAINS = GPU_BRAINS + CPU_BRAINS  # 21 total"
    )
    nb['cells'][13]['source'] = [cell13_src]
    print("Cell 13: Updated GPU_BRAINS (17 -> 19)")

# =============================================================================
# 2. UPDATE CELL 15: Add NBEATS and TFT architectures
# =============================================================================
cell15_src = ''.join(nb['cells'][15]['source']) if isinstance(nb['cells'][15]['source'], list) else nb['cells'][15]['source']

nbeats_tft_code = '''

# =============================================================================
# N-BEATS: Neural Basis Expansion Analysis for Time Series
# =============================================================================

class NBeatsBlock(nn.Module):
    """Single N-BEATS block with backcast and forecast outputs."""
    def __init__(self, input_size, theta_size, basis_function, layers=4, layer_size=256):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, layer_size)] +
                                     [nn.Linear(layer_size, layer_size) for _ in range(layers - 1)])
        self.basis_parameters = nn.Linear(layer_size, theta_size)
        self.basis_function = basis_function

    def forward(self, x):
        block_input = x
        for layer in self.layers:
            x = torch.relu(layer(x))
        basis_params = self.basis_parameters(x)
        return self.basis_function(basis_params, block_input)


class NBeatsClassifier(nn.Module):
    """
    N-BEATS adapted for classification.
    Uses stacked blocks with backcast/forecast architecture.
    """
    def __init__(self, n_features, hidden_size=256, num_blocks=3, theta_size=32,
                 dropout=0.2, n_classes=3, seq_len=10, **kwargs):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size

        # Input projection
        self.input_proj = nn.Linear(n_features, hidden_size)

        # N-BEATS blocks (simplified for classification)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.blocks.append(block)

        # Residual connections
        self.residual_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_blocks)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * num_blocks, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x):
        # Handle 3D input (batch, seq, features)
        if x.dim() == 3:
            x = x.mean(dim=1)

        # Project input
        x = self.input_proj(x)

        # Process through blocks with residual connections
        block_outputs = []
        residual = x
        for block, res_proj in zip(self.blocks, self.residual_projs):
            block_out = block(residual)
            residual = res_proj(residual) + block_out
            block_outputs.append(residual)

        # Concatenate all block outputs
        combined = torch.cat(block_outputs, dim=1)

        # Classify
        return self.classifier(combined)


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
        self.layer_norm = nn.LayerNorm(output_size)

        # Skip connection projection if dimensions differ
        if input_size != output_size:
            self.skip_proj = nn.Linear(input_size, output_size)
        else:
            self.skip_proj = None

    def forward(self, x):
        # Skip connection
        if self.skip_proj is not None:
            skip = self.skip_proj(x)
        else:
            skip = x

        # Main path
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)

        # Output with gating
        output = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        gated_output = gate * output

        # Residual + layer norm
        return self.layer_norm(skip + gated_output)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for TFT."""
    def __init__(self, input_size, n_vars, hidden_size, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.hidden_size = hidden_size

        # GRN for each variable
        self.grns = nn.ModuleList([
            GatedResidualNetwork(input_size // n_vars, hidden_size, hidden_size, dropout)
            for _ in range(n_vars)
        ])

        # Softmax weights for variable selection
        self.weight_grn = GatedResidualNetwork(input_size, hidden_size, n_vars, dropout)

    def forward(self, x):
        # Assume x is (batch, features) where features = n_vars * var_size
        batch_size = x.size(0)
        var_size = x.size(1) // self.n_vars

        # Process each variable
        var_outputs = []
        for i, grn in enumerate(self.grns):
            var_input = x[:, i*var_size:(i+1)*var_size]
            var_outputs.append(grn(var_input))

        # Stack: (batch, n_vars, hidden)
        var_outputs = torch.stack(var_outputs, dim=1)

        # Get selection weights
        weights = torch.softmax(self.weight_grn(x), dim=-1)  # (batch, n_vars)
        weights = weights.unsqueeze(-1)  # (batch, n_vars, 1)

        # Weighted sum
        selected = (var_outputs * weights).sum(dim=1)  # (batch, hidden)

        return selected


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer adapted for classification.
    Simplified version with key TFT components.
    """
    def __init__(self, n_features, hidden_size=64, num_heads=4, num_layers=2,
                 dropout=0.2, n_classes=3, seq_len=10, **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_features = n_features

        # Ensure hidden_size is divisible by num_heads
        if hidden_size % num_heads != 0:
            hidden_size = ((hidden_size // num_heads) + 1) * num_heads
            self.hidden_size = hidden_size

        # Input projection with GRN
        self.input_grn = GatedResidualNetwork(n_features, hidden_size, hidden_size, dropout)

        # LSTM encoder for temporal patterns
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Multi-head attention for interpretable attention patterns
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_layer_norm = nn.LayerNorm(hidden_size)

        # Post-attention GRN
        self.post_attn_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x):
        # Handle 2D input (batch, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)

        batch_size, seq_len, _ = x.shape

        # Input processing with GRN
        x_flat = x.view(batch_size * seq_len, -1)
        x_grn = self.input_grn(x_flat)
        x_grn = x_grn.view(batch_size, seq_len, -1)

        # LSTM encoding
        lstm_out, _ = self.lstm(x_grn)

        # Self-attention with residual
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_layer_norm(lstm_out + attn_out)

        # Post-attention GRN
        attn_out_flat = attn_out.view(batch_size * seq_len, -1)
        grn_out = self.post_attn_grn(attn_out_flat)
        grn_out = grn_out.view(batch_size, seq_len, -1)

        # Pool over sequence
        pooled = grn_out.mean(dim=1)

        # Classify
        return self.classifier(pooled)


print("N-BEATS and TFT Architectures loaded (2)")
print("  - NBeatsClassifier: Neural Basis Expansion Analysis")
print("  - TemporalFusionTransformer: Interpretable attention-based")
'''

if 'class NBeatsClassifier' not in cell15_src:
    # Find the end of cell (after the last print statement)
    cell15_src = cell15_src.rstrip()
    # Add new architectures
    cell15_src = cell15_src + nbeats_tft_code
    nb['cells'][15]['source'] = [cell15_src]
    print("Cell 15: Added NBeatsClassifier and TemporalFusionTransformer")

# =============================================================================
# 3. UPDATE CELL 18: Add NBEATS and TFT objectives
# =============================================================================
cell18_src = ''.join(nb['cells'][18]['source']) if isinstance(nb['cells'][18]['source'], list) else nb['cells'][18]['source']

nbeats_tft_objectives = '''

def create_nbeats_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf):
    """N-BEATS objective with specialized hyperparameters."""
    return create_dl_objective(NBeatsClassifier, X_train, y_train, X_val, y_val, returns_val, pair, tf)

def create_tft_objective(X_train, y_train, X_val, y_val, returns_val, pair, tf):
    """Temporal Fusion Transformer objective with specialized hyperparameters."""
    return create_dl_objective(TemporalFusionTransformer, X_train, y_train, X_val, y_val, returns_val, pair, tf)


print("N-BEATS and TFT Objectives loaded (2)")
'''

if 'def create_nbeats_objective' not in cell18_src:
    # Find the last print statement and add after it
    cell18_src = cell18_src.rstrip()
    cell18_src = cell18_src + nbeats_tft_objectives
    nb['cells'][18]['source'] = [cell18_src]
    print("Cell 18: Added create_nbeats_objective and create_tft_objective")

# =============================================================================
# 4. UPDATE CELL 20: Update dispatcher
# =============================================================================
cell20_src = ''.join(nb['cells'][20]['source']) if isinstance(nb['cells'][20]['source'], list) else nb['cells'][20]['source']

# Replace dispatcher section
old_dispatcher = """        # Deep Learning (9)
        'lstm_optuna': create_lstm_objective,
        'gru_optuna': create_gru_objective,
        'transformer_optuna': create_transformer_objective,
        'cnn1d_optuna': create_cnn1d_objective,
        'tcn_optuna': create_tcn_objective,
        'wavenet_optuna': create_wavenet_objective,
        'attention_net_optuna': create_attention_net_objective,
        'residual_mlp_optuna': create_residual_mlp_objective,
        'ensemble_nn_optuna': create_ensemble_nn_objective,

        # CPU brains (2)"""

new_dispatcher = """        # Deep Learning (9)
        'lstm_optuna': create_lstm_objective,
        'gru_optuna': create_gru_objective,
        'transformer_optuna': create_transformer_objective,
        'cnn1d_optuna': create_cnn1d_objective,
        'tcn_optuna': create_tcn_objective,
        'wavenet_optuna': create_wavenet_objective,
        'attention_net_optuna': create_attention_net_objective,
        'residual_mlp_optuna': create_residual_mlp_objective,
        'ensemble_nn_optuna': create_ensemble_nn_objective,

        # Time Series Specialized (2) - NEW
        'nbeats_optuna': create_nbeats_objective,
        'tft_optuna': create_tft_objective,

        # CPU brains (2)"""

if 'nbeats_optuna' not in cell20_src:
    cell20_src = cell20_src.replace(old_dispatcher, new_dispatcher)

    # Update the assertion list
    old_assert = """    assert brain in ['lgb_optuna', 'xgb_optuna', 'cat_optuna', 'lgb_v2_optuna',
                     'xgb_v2_optuna', 'cat_v2_optuna', 'tabnet_optuna', 'mlp_optuna',
                     'lstm_optuna', 'gru_optuna', 'transformer_optuna', 'cnn1d_optuna',
                     'tcn_optuna', 'wavenet_optuna', 'attention_net_optuna',
                     'residual_mlp_optuna', 'ensemble_nn_optuna', 'rf_optuna', 'et_optuna']"""

    new_assert = """    assert brain in ['lgb_optuna', 'xgb_optuna', 'cat_optuna', 'lgb_v2_optuna',
                     'xgb_v2_optuna', 'cat_v2_optuna', 'tabnet_optuna', 'mlp_optuna',
                     'lstm_optuna', 'gru_optuna', 'transformer_optuna', 'cnn1d_optuna',
                     'tcn_optuna', 'wavenet_optuna', 'attention_net_optuna',
                     'residual_mlp_optuna', 'ensemble_nn_optuna',
                     'nbeats_optuna', 'tft_optuna', 'rf_optuna', 'et_optuna']"""

    cell20_src = cell20_src.replace(old_assert, new_assert)
    nb['cells'][20]['source'] = [cell20_src]
    print("Cell 20: Updated dispatcher with nbeats_optuna and tft_optuna")

# =============================================================================
# 5. UPDATE CELL 21: Update train_final_model
# =============================================================================
cell21_src = ''.join(nb['cells'][21]['source']) if isinstance(nb['cells'][21]['source'], list) else nb['cells'][21]['source']

# Find the model builder section and add nbeats/tft
old_model_builder = """    elif brain == 'ensemble_nn_optuna':
        model = EnsembleNNClassifier(n_features=n_features, **best_params).to(DEVICE)
    else:
        raise ValueError(f"Unknown PyTorch brain: {brain}")"""

new_model_builder = """    elif brain == 'ensemble_nn_optuna':
        model = EnsembleNNClassifier(n_features=n_features, **best_params).to(DEVICE)
    elif brain == 'nbeats_optuna':
        model = NBeatsClassifier(n_features=n_features, **best_params).to(DEVICE)
    elif brain == 'tft_optuna':
        model = TemporalFusionTransformer(n_features=n_features, **best_params).to(DEVICE)
    else:
        raise ValueError(f"Unknown PyTorch brain: {brain}")"""

if 'nbeats_optuna' not in cell21_src:
    cell21_src = cell21_src.replace(old_model_builder, new_model_builder)
    nb['cells'][21]['source'] = [cell21_src]
    print("Cell 21: Updated train_final_model with NBEATS and TFT")

# =============================================================================
# SAVE UPDATED NOTEBOOK
# =============================================================================
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("\n" + "=" * 60)
print("NOTEBOOK UPDATED SUCCESSFULLY")
print("=" * 60)

# Verify changes
with open(nb_path, 'r', encoding='utf-8') as f:
    nb_verify = json.load(f)

all_source = ''
for cell in nb_verify['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        all_source += src + '\n'

print("\nVERIFICATION:")
print(f"  NBeatsClassifier present: {'class NBeatsClassifier' in all_source}")
print(f"  TemporalFusionTransformer present: {'class TemporalFusionTransformer' in all_source}")
print(f"  GatedResidualNetwork present: {'class GatedResidualNetwork' in all_source}")
print(f"  create_nbeats_objective present: {'create_nbeats_objective' in all_source}")
print(f"  create_tft_objective present: {'create_tft_objective' in all_source}")
print(f"  nbeats_optuna in GPU_BRAINS: {'nbeats_optuna' in all_source}")
print(f"  tft_optuna in GPU_BRAINS: {'tft_optuna' in all_source}")

# Count brains
gpu_brains = []
for b in ['lgb_optuna', 'xgb_optuna', 'cat_optuna', 'lgb_v2_optuna', 'xgb_v2_optuna',
          'cat_v2_optuna', 'tabnet_optuna', 'mlp_optuna', 'lstm_optuna', 'gru_optuna',
          'transformer_optuna', 'cnn1d_optuna', 'tcn_optuna', 'wavenet_optuna',
          'attention_net_optuna', 'residual_mlp_optuna', 'ensemble_nn_optuna',
          'nbeats_optuna', 'tft_optuna']:
    if f"'{b}'" in all_source:
        gpu_brains.append(b)

cpu_brains = []
for b in ['rf_optuna', 'et_optuna']:
    if f"'{b}'" in all_source:
        cpu_brains.append(b)

print(f"\nGPU Brains: {len(gpu_brains)}")
for b in gpu_brains:
    print(f"  - {b}")

print(f"\nCPU Brains: {len(cpu_brains)}")
for b in cpu_brains:
    print(f"  - {b}")

total = len(gpu_brains) + len(cpu_brains)
models = 9 * 7 * total
print(f"\nTOTAL BRAINS: {total}")
print(f"TOTAL MODELS: 9 pairs x 7 TFs x {total} brains = {models}")
print("=" * 60)
