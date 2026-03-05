"""
CHAOS V1.0 ONNX Export Pipeline
================================
Converts trained models to ONNX format for unified inference.

Backend types:
  - OnnxBackend: For all ONNX-converted models (LGB, XGB, CatBoost, PyTorch neural nets)
  - SklearnBackend: For RF/ET models (kept as .joblib, no ONNX conversion)

CRITICAL DESIGN RULE: Every model backend exposes the same interface:
    predict_proba(features[n_samples, N_FEATURES]) -> probs[n_samples, 3]

N_FEATURES is determined per model from the training checkpoint.
The canonical schema (273 universal features) is used at the ZeroMQ boundary.
Models trained on pair-specific supersets use their full feature set internally.

# MT5 NEVER decides model logic. MT5 only:
# 1. Assembles features from price data
# 2. Sends ZeroMQ request
# 3. Receives decision
# 4. Executes with CoreArb DLL + risk constraints
"""
import os
import sys
import json
import logging
import numpy as np
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('onnx_export')

# =============================================================================
# NEURAL NETWORK ARCHITECTURES (replicated from chaos_gpu_training.py)
# These must match the training script EXACTLY for state_dict loading.
# =============================================================================
import torch
import torch.nn as nn


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
        in_ch = 1
        for i, out_ch in enumerate(channels):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout))
            in_ch = out_ch
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
# BRAIN -> MODEL CLASS MAPPING
# =============================================================================
# Maps brain name to (ModelClass, constructor_kwargs_from_checkpoint)
BRAIN_CLASS_MAP = {
    'mlp_optuna': MLPClassifier,
    'lstm_optuna': LSTMClassifier,
    'gru_optuna': GRUClassifier,
    'transformer_optuna': TransformerClassifier,
    'cnn1d_optuna': CNN1DClassifier,
    'tcn_optuna': TCNClassifier,
    'wavenet_optuna': WaveNetClassifier,
    'attention_net_optuna': AttentionNetClassifier,
    'residual_mlp_optuna': ResidualMLPClassifier,
    'ensemble_nn_optuna': EnsembleNNClassifier,
    'nbeats_optuna': NBeatsClassifier,
    'tft_optuna': TemporalFusionTransformer,
}


def _infer_model_params(brain, checkpoint, n_features):
    """
    Infer model constructor parameters from checkpoint state_dict.
    The training script does not save model_config, so we must infer
    from the state_dict shapes.
    """
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    if brain == 'mlp_optuna':
        # Infer hidden sizes from linear layer shapes
        hidden_sizes = []
        for key in sorted(state_dict.keys()):
            if 'network' in key and 'weight' in key and 'bn' not in key.lower():
                hidden_sizes.append(state_dict[key].shape[0])
        # Last one is num_classes (3), remove it
        if hidden_sizes and hidden_sizes[-1] == 3:
            hidden_sizes = hidden_sizes[:-1]
        return {'input_size': n_features, 'num_classes': 3,
                'hidden_sizes': hidden_sizes if hidden_sizes else [256, 128, 64]}

    elif brain == 'transformer_optuna':
        d_model = state_dict['input_projection.weight'].shape[0]
        return {'input_size': n_features, 'num_classes': 3,
                'd_model': d_model, 'nhead': 4, 'num_layers': 2}

    elif brain in ('lstm_optuna', 'gru_optuna'):
        # LSTM/GRU: weight_ih_l0 shape is (4*hidden_size, input_size) for LSTM
        rnn_key = 'lstm' if brain == 'lstm_optuna' else 'gru'
        w_key = f'{rnn_key}.weight_ih_l0'
        if w_key in state_dict:
            multiplier = 4 if brain == 'lstm_optuna' else 3
            hidden_size = state_dict[w_key].shape[0] // multiplier
        else:
            hidden_size = 128
        return {'input_size': n_features, 'num_classes': 3,
                'hidden_size': hidden_size, 'num_layers': 2}

    elif brain == 'cnn1d_optuna':
        # Infer channels from conv weight shapes
        channels = []
        for key in sorted(state_dict.keys()):
            if 'conv' in key and 'weight' in key and key.startswith('conv.'):
                channels.append(state_dict[key].shape[0])
        # Deduplicate (BatchNorm weights also match)
        seen = []
        for c in channels:
            if not seen or seen[-1] != c:
                seen.append(c)
        return {'input_size': n_features, 'num_classes': 3,
                'channels': seen if seen else [32, 64, 64]}

    elif brain == 'tcn_optuna':
        channels = []
        for key in sorted(state_dict.keys()):
            if 'conv1.weight' in key:
                channels.append(state_dict[key].shape[0])
        return {'input_size': n_features, 'num_classes': 3,
                'channels': channels if channels else [32, 64]}

    elif brain == 'wavenet_optuna':
        ch = state_dict['input_conv.weight'].shape[0] if 'input_conv.weight' in state_dict else 64
        n_blocks = sum(1 for k in state_dict.keys() if 'blocks.' in k and '.dilated_conv.weight' in k)
        return {'input_size': n_features, 'num_classes': 3,
                'channels': ch, 'num_layers': n_blocks if n_blocks > 0 else 3, 'kernel_size': 2}

    elif brain == 'attention_net_optuna':
        hidden_size = state_dict['input_projection.weight'].shape[0]
        return {'input_size': n_features, 'num_classes': 3,
                'hidden_size': hidden_size, 'num_heads': 4}

    elif brain == 'residual_mlp_optuna':
        hidden_size = state_dict['input_projection.weight'].shape[0]
        n_blocks = sum(1 for k in state_dict.keys() if '.block.0.weight' in k)
        return {'input_size': n_features, 'num_classes': 3,
                'hidden_size': hidden_size, 'num_blocks': n_blocks if n_blocks > 0 else 3}

    elif brain == 'ensemble_nn_optuna':
        hidden_size = state_dict['mlp.0.weight'].shape[0] if 'mlp.0.weight' in state_dict else 64
        return {'input_size': n_features, 'num_classes': 3, 'hidden_size': hidden_size}

    elif brain == 'nbeats_optuna':
        # Infer from first block
        hidden_size = state_dict.get('blocks.0.fc.0.weight', torch.zeros(256, 1)).shape[0]
        n_blocks = sum(1 for k in state_dict.keys() if '.fc.0.weight' in k and k.startswith('blocks.'))
        return {'input_size': n_features, 'num_classes': 3,
                'hidden_size': hidden_size, 'num_blocks': n_blocks if n_blocks > 0 else 4}

    elif brain == 'tft_optuna':
        hidden_size = state_dict.get('input_grn.fc1.weight', torch.zeros(128, 1)).shape[0]
        return {'input_size': n_features, 'num_classes': 3,
                'hidden_size': hidden_size, 'num_heads': 4, 'num_layers': 2}

    else:
        raise ValueError(f"Unknown brain type: {brain}")


# =============================================================================
# PYTORCH -> ONNX EXPORT
# =============================================================================
def export_pytorch_to_onnx(model_path, output_path, input_size=None):
    """
    Export a PyTorch .pt model to ONNX format.

    Args:
        model_path: path to .pt file
        output_path: path for .onnx output
        input_size: feature count (inferred from checkpoint if None)

    Returns:
        output_path on success
    """
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError(f"Unexpected checkpoint format in {model_path}. Keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}")

    state_dict = checkpoint['model_state_dict']
    brain = checkpoint.get('brain', '')
    n_features = checkpoint.get('n_features', input_size)

    # Infer n_features from state_dict if not in checkpoint
    if n_features is None:
        # Try input_projection.weight (most models)
        if 'input_projection.weight' in state_dict:
            n_features = state_dict['input_projection.weight'].shape[1]
        elif 'network.0.weight' in state_dict:
            n_features = state_dict['network.0.weight'].shape[1]
        elif 'input_grn.fc1.weight' in state_dict:
            n_features = state_dict['input_grn.fc1.weight'].shape[1]
        else:
            raise ValueError(f"Cannot infer n_features from checkpoint. Provide input_size.")

    logger.info(f"Exporting {brain} ({n_features} features) -> {output_path}")

    if brain not in BRAIN_CLASS_MAP:
        raise ValueError(f"Unknown brain '{brain}'. Known: {list(BRAIN_CLASS_MAP.keys())}")

    # Reconstruct model
    model_class = BRAIN_CLASS_MAP[brain]
    params = _infer_model_params(brain, checkpoint, n_features)
    model = model_class(**params)
    model.load_state_dict(state_dict)
    model.eval()

    # Export to ONNX
    dummy_input = torch.randn(1, n_features, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['features'],
        output_names=['class_logits'],
        dynamic_axes={'features': {0: 'batch'}, 'class_logits': {0: 'batch'}},
        opset_version=17,
        do_constant_folding=True
    )

    logger.info(f"  ONNX export complete: {output_path}")
    return output_path, model


# =============================================================================
# LIGHTGBM -> ONNX
# =============================================================================
def export_lgb_to_onnx(model_path, output_path, input_size=273):
    """
    Export LightGBM .joblib model to ONNX.
    Install: pip install onnxmltools skl2onnx onnxconverter-common
    """
    import onnxmltools
    from onnxmltools.convert import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    loaded = joblib.load(model_path)
    if isinstance(loaded, dict):
        model = loaded.get('model', loaded.get('estimator', loaded))
    else:
        model = loaded

    # Infer actual input size from model
    try:
        actual_n_features = model.n_features_
    except AttributeError:
        actual_n_features = input_size

    initial_type = [('features', FloatTensorType([None, actual_n_features]))]
    # onnxmltools tree converters max at opset 15; opset 17 only for PyTorch
    onnx_model = convert_lightgbm(model, initial_types=initial_type, target_opset=15)
    onnxmltools.utils.save_model(onnx_model, output_path)

    logger.info(f"  LGB ONNX export: {output_path} ({actual_n_features} features)")
    return output_path


# =============================================================================
# XGBOOST -> ONNX
# =============================================================================
def export_xgb_to_onnx(model_path, output_path, input_size=273):
    """Export XGBoost .joblib model to ONNX."""
    import onnxmltools
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    loaded = joblib.load(model_path)
    if isinstance(loaded, dict):
        model = loaded.get('model', loaded.get('estimator', loaded))
    else:
        model = loaded

    try:
        actual_n_features = model.n_features_in_
    except AttributeError:
        actual_n_features = input_size

    initial_type = [('features', FloatTensorType([None, actual_n_features]))]
    # onnxmltools tree converters max at opset 15; opset 17 only for PyTorch
    onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=15)
    onnxmltools.utils.save_model(onnx_model, output_path)

    logger.info(f"  XGB ONNX export: {output_path} ({actual_n_features} features)")
    return output_path


# =============================================================================
# CATBOOST -> ONNX
# =============================================================================
def export_catboost_to_onnx(model_path, output_path, input_size=273):
    """CatBoost has native ONNX export."""
    loaded = joblib.load(model_path)
    if isinstance(loaded, dict):
        model = loaded.get('model', loaded.get('estimator', loaded))
    else:
        model = loaded

    model.save_model(output_path, format="onnx",
                     export_parameters={'onnx_domain': 'ai.catboost',
                                        'onnx_model_version': 1})

    logger.info(f"  CatBoost ONNX export: {output_path}")
    return output_path


# =============================================================================
# INFERENCE BACKENDS
# =============================================================================
class SklearnBackend:
    """
    Native sklearn inference for RF/ET models.
    RF/ET models stay as .joblib — no ONNX conversion.
    This avoids skl2onnx fragility with large forests (some RF models are 400+ MB).
    """

    def __init__(self, model_path):
        loaded = joblib.load(model_path)
        if isinstance(loaded, dict):
            self.model = loaded.get('model', loaded.get('estimator', loaded))
            self.scaler = loaded.get('scaler', None)
        else:
            self.model = loaded
            self.scaler = None

    def predict_proba(self, features):
        """
        Args:
            features: np.ndarray shape (n_samples, n_features)
        Returns:
            np.ndarray shape (n_samples, 3) — [P(SHORT), P(FLAT), P(LONG)]
        """
        X = features.astype(np.float64)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def batch_predict_proba(self, features_batch):
        """Batch prediction: (N, n_features) -> (N, 3). Single sklearn call."""
        return self.predict_proba(features_batch)


class OnnxBackend:
    """ONNX Runtime inference for all GPU-trained models."""

    def __init__(self, onnx_path, scaler=None):
        import onnxruntime as ort
        ort.set_default_logger_severity(3)  # ERROR only — suppress batch shape warnings
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.scaler = scaler
        # Detect batch dim support: None or string = dynamic, int 1 = fixed
        batch_dim = self.session.get_inputs()[0].shape[0]
        self._supports_batch = batch_dim is None or isinstance(batch_dim, str)

    def predict_proba(self, features):
        """
        Args:
            features: np.ndarray shape (n_samples, n_features)
        Returns:
            np.ndarray shape (n_samples, 3) — [P(SHORT), P(FLAT), P(LONG)]
        """
        X = features.astype(np.float32)
        if self.scaler is not None:
            X = self.scaler.transform(X).astype(np.float32)
        ort_input = {self.input_name: X}
        results = self.session.run(None, ort_input)

        # Tree model ONNX outputs: [labels, probabilities_zipmap]
        # Neural network ONNX outputs: [logits]
        if len(results) >= 2 and isinstance(results[1], list):
            # ZipMap format from tree models (list of dicts)
            output = np.array([[d.get(i, 0.0) for i in range(3)] for d in results[1]])
        elif len(results) >= 2 and isinstance(results[1], np.ndarray):
            output = results[1]
        else:
            output = results[0]
            # Normalize logits to probabilities if needed
            if isinstance(output, np.ndarray) and output.ndim == 2:
                if not np.allclose(output.sum(axis=1), 1.0, atol=0.01):
                    exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
                    output = exp_output / exp_output.sum(axis=1, keepdims=True)

        return output

    def batch_predict_proba(self, features_batch):
        """Batch prediction: (N, n_features) -> (N, 3).
        Single call if model supports dynamic batch, else loop single-row."""
        if self._supports_batch:
            return self.predict_proba(features_batch)
        # Fixed batch dim=1: loop single-row
        results = []
        for i in range(features_batch.shape[0]):
            row = features_batch[i:i+1]
            results.append(self.predict_proba(row))
        return np.vstack(results)


# =============================================================================
# PARITY VALIDATION
# =============================================================================
def validate_onnx_parity(pytorch_model, onnx_path, test_data, scaler=None, tolerance=1e-4):
    """
    Compare PyTorch and ONNX outputs on identical inputs.

    Args:
        pytorch_model: loaded PyTorch model in eval() mode
        onnx_path: path to exported .onnx file
        test_data: np.ndarray of shape (n_samples, n_features), raw (unscaled)
        scaler: StandardScaler used during training (or None)
        tolerance: max absolute difference allowed per probability

    Returns:
        (bool, max_diff, mean_diff)
    """
    import onnxruntime as ort

    # Scale if needed
    if scaler is not None:
        data_scaled = scaler.transform(test_data).astype(np.float32)
    else:
        data_scaled = test_data.astype(np.float32)

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pt_input = torch.from_numpy(data_scaled).float()
        pt_logits = pytorch_model(pt_input)
        pt_output = torch.softmax(pt_logits, dim=1).numpy()

    # ONNX inference
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_input = {session.get_inputs()[0].name: data_scaled}
    ort_output = session.run(None, ort_input)[0]

    # If ONNX output is logits (not softmax), apply softmax
    if not np.allclose(ort_output.sum(axis=1), 1.0, atol=0.01):
        exp_out = np.exp(ort_output - np.max(ort_output, axis=1, keepdims=True))
        ort_output = exp_out / exp_out.sum(axis=1, keepdims=True)

    max_diff = np.max(np.abs(pt_output - ort_output))
    mean_diff = np.mean(np.abs(pt_output - ort_output))

    passed = max_diff < tolerance
    return passed, max_diff, mean_diff


def validate_lgb_onnx_parity(model_path, onnx_path, test_data, tolerance=1e-4):
    """Compare LightGBM joblib vs ONNX outputs."""
    import onnxruntime as ort

    loaded = joblib.load(model_path)
    if isinstance(loaded, dict):
        model = loaded.get('model', loaded.get('estimator', loaded))
    else:
        model = loaded

    # Original predictions
    original_probs = model.predict_proba(test_data.astype(np.float64))

    # ONNX predictions
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_input = {session.get_inputs()[0].name: test_data.astype(np.float32)}
    results = session.run(None, ort_input)

    # LGB ONNX output may have label + probabilities
    if len(results) >= 2:
        onnx_probs = results[1]
        # onnxmltools LGB output is often a list of dicts — handle that
        if isinstance(onnx_probs, list):
            onnx_probs = np.array([[d.get(0, 0), d.get(1, 0), d.get(2, 0)] for d in onnx_probs])
        elif isinstance(onnx_probs, np.ndarray) and onnx_probs.ndim == 1:
            onnx_probs = onnx_probs.reshape(-1, 3)
    else:
        onnx_probs = results[0]
        if not np.allclose(onnx_probs.sum(axis=1), 1.0, atol=0.01):
            exp_out = np.exp(onnx_probs - np.max(onnx_probs, axis=1, keepdims=True))
            onnx_probs = exp_out / exp_out.sum(axis=1, keepdims=True)

    max_diff = np.max(np.abs(original_probs - onnx_probs))
    mean_diff = np.mean(np.abs(original_probs - onnx_probs))

    passed = max_diff < tolerance
    return passed, max_diff, mean_diff


# =============================================================================
# BATCH EXPORT
# =============================================================================
def export_all_models(models_dir, output_dir, schema_path=None):
    """
    Export all models in models_dir to ONNX (or SklearnBackend for RF/ET).

    Args:
        models_dir: path containing .pt and .joblib files
        output_dir: path for .onnx outputs
        schema_path: path to feature_schema.json (for input_size)
    """
    models_path = Path(models_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    skip_tfs = {'M1', 'W1', 'MN1'}
    rf_et_brains = {'rf_optuna', 'et_optuna'}
    gbm_brains = {'lgb_optuna', 'lgb_v2_optuna'}
    xgb_brains = {'xgb_optuna', 'xgb_v2_optuna'}
    cat_brains = {'cat_optuna', 'cat_v2_optuna'}
    tabnet_brains = {'tabnet_optuna'}

    results = {'success': [], 'failed': [], 'skipped': []}

    # Process .pt files (neural networks)
    for pt_file in sorted(models_path.glob('*.pt')):
        parts = pt_file.stem.split('_')
        if len(parts) < 3:
            continue
        pair, tf = parts[0], parts[1]
        brain = '_'.join(parts[2:])

        if tf in skip_tfs:
            results['skipped'].append(pt_file.name)
            continue

        onnx_file = output_path / f"{pt_file.stem}.onnx"
        try:
            export_pytorch_to_onnx(str(pt_file), str(onnx_file))
            results['success'].append(pt_file.name)
        except Exception as e:
            logger.error(f"Failed: {pt_file.name}: {e}")
            results['failed'].append((pt_file.name, str(e)))

    # Process .joblib files (tree models)
    for jl_file in sorted(models_path.glob('*.joblib')):
        parts = jl_file.stem.split('_')
        if len(parts) < 3:
            continue
        pair, tf = parts[0], parts[1]
        brain = '_'.join(parts[2:])

        if tf in skip_tfs:
            results['skipped'].append(jl_file.name)
            continue

        if brain in rf_et_brains:
            results['skipped'].append(f"{jl_file.name} (RF/ET stays as joblib)")
            continue

        if brain in tabnet_brains:
            results['skipped'].append(f"{jl_file.name} (TabNet stays as joblib)")
            continue

        onnx_file = output_path / f"{jl_file.stem}.onnx"
        try:
            if brain in gbm_brains:
                export_lgb_to_onnx(str(jl_file), str(onnx_file))
            elif brain in xgb_brains:
                export_xgb_to_onnx(str(jl_file), str(onnx_file))
            elif brain in cat_brains:
                export_catboost_to_onnx(str(jl_file), str(onnx_file))
            else:
                results['skipped'].append(f"{jl_file.name} (unknown brain type)")
                continue
            results['success'].append(jl_file.name)
        except Exception as e:
            logger.error(f"Failed: {jl_file.name}: {e}")
            results['failed'].append((jl_file.name, str(e)))

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CHAOS V1.0 ONNX Export Pipeline')
    parser.add_argument('--models', default='G:/My Drive/chaos_v1.0/models')
    parser.add_argument('--output', default='G:/My Drive/chaos_v1.0/models/onnx')
    parser.add_argument('--schema', default='G:/My Drive/chaos_v1.0/schema/feature_schema.json')
    args = parser.parse_args()

    results = export_all_models(args.models, args.output, args.schema)

    print(f"\nExport Results:")
    print(f"  Success: {len(results['success'])}")
    print(f"  Failed:  {len(results['failed'])}")
    print(f"  Skipped: {len(results['skipped'])}")

    if results['failed']:
        print("\nFailed models:")
        for name, err in results['failed']:
            print(f"  {name}: {err}")
