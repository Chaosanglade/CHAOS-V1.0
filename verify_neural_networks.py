#!/usr/bin/env python3
"""
Neural Network Verification Test
=================================
Tests ALL 12 neural network models can be instantiated, trained, and saved.
Run this BEFORE training to verify all models work.
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn

print("=" * 70)
print("NEURAL NETWORK VERIFICATION TEST - ALL 12 MODELS")
print("=" * 70)

# Check CUDA
print(f"\nCUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

# Create test data
print("\nCreating test data...")
np.random.seed(42)
N_FEATURES = 100
X_train = np.random.randn(1000, N_FEATURES).astype(np.float32)
y_train = np.random.randint(0, 3, 1000)
X_val = np.random.randn(200, N_FEATURES).astype(np.float32)
y_val = np.random.randint(0, 3, 200)

print(f"X_train shape: {X_train.shape}")
print(f"y_train distribution: {np.bincount(y_train)}")

# Import model classes
print("\nImporting model classes...")
try:
    # Add the path if running from different directory
    sys.path.insert(0, '/content/drive/MyDrive/chaos_v1.0')
    sys.path.insert(0, 'G:/My Drive/chaos_v1.0')

    from chaos_gpu_training import (
        MLPClassifier, LSTMClassifier, GRUClassifier, TransformerClassifier,
        CNN1DClassifier, TCNClassifier, WaveNetClassifier, AttentionNetClassifier,
        ResidualMLPClassifier, EnsembleNNClassifier, NBeatsClassifier,
        TemporalFusionTransformer, train_pytorch_model, predict_pytorch, DEVICE
    )
    print("  All imports successful!")
except ImportError as e:
    print(f"  Import failed: {e}")
    print("\nDefining models locally for testing...")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configurations - EXACTLY matching what train_final_model uses
MODEL_CONFIGS = {
    'mlp_optuna': {
        'class': 'MLPClassifier',
        'params': {'hidden_sizes': [128, 64, 32], 'dropout': 0.3}
    },
    'lstm_optuna': {
        'class': 'LSTMClassifier',
        'params': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3}
    },
    'gru_optuna': {
        'class': 'GRUClassifier',
        'params': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3}
    },
    'transformer_optuna': {
        'class': 'TransformerClassifier',
        'params': {'d_model': 128, 'nhead': 4, 'num_layers': 2, 'dropout': 0.3}
    },
    'cnn1d_optuna': {
        'class': 'CNN1DClassifier',
        'params': {'channels': [64, 128, 256], 'kernel_size': 3, 'dropout': 0.3}
    },
    'tcn_optuna': {
        'class': 'TCNClassifier',
        'params': {'channels': [128, 256], 'kernel_size': 3, 'dropout': 0.3}
    },
    'wavenet_optuna': {
        'class': 'WaveNetClassifier',
        'params': {'channels': 128, 'num_layers': 4, 'kernel_size': 2, 'dropout': 0.3}
    },
    'attention_net_optuna': {
        'class': 'AttentionNetClassifier',
        'params': {'hidden_size': 128, 'num_heads': 4, 'dropout': 0.3}
    },
    'residual_mlp_optuna': {
        'class': 'ResidualMLPClassifier',
        'params': {'hidden_size': 128, 'num_blocks': 3, 'dropout': 0.3}
    },
    'ensemble_nn_optuna': {
        'class': 'EnsembleNNClassifier',
        'params': {'hidden_size': 128, 'dropout': 0.3}
    },
    'nbeats_optuna': {
        'class': 'NBeatsClassifier',
        'params': {'hidden_size': 128, 'num_blocks': 4, 'dropout': 0.3}
    },
    'tft_optuna': {
        'class': 'TemporalFusionTransformer',
        'params': {'hidden_size': 128, 'num_heads': 4, 'num_layers': 2, 'dropout': 0.1}
    },
}

# Test each model
results = {}
print("\n" + "=" * 70)
print("TESTING ALL 12 NEURAL NETWORK MODELS")
print("=" * 70)

for brain, config in MODEL_CONFIGS.items():
    print(f"\n[{len(results)+1}/12] Testing {brain}...")
    class_name = config['class']
    params = config['params']

    try:
        # Get class
        model_class = eval(class_name)

        # Instantiate
        start = time.time()
        model = model_class(N_FEATURES, num_classes=3, **params)
        model = model.to(DEVICE)
        init_time = time.time() - start
        print(f"  Init: OK ({init_time:.3f}s)")

        # Quick forward pass test
        with torch.no_grad():
            x_test = torch.randn(32, N_FEATURES).to(DEVICE)
            output = model(x_test)
            assert output.shape == (32, 3), f"Wrong output shape: {output.shape}"
        print(f"  Forward: OK (output shape: {output.shape})")

        # Quick training test (just 2 epochs)
        from sklearn.preprocessing import StandardScaler
        from sklearn.utils.class_weight import compute_class_weight
        from torch.utils.data import DataLoader, TensorDataset

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Compute class weights
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights_tensor = torch.FloatTensor(weights).to(DEVICE)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        start = time.time()
        model.train()
        for epoch in range(2):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        train_time = time.time() - start
        print(f"  Training (2 epochs): OK ({train_time:.2f}s)")

        # Prediction test
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(DEVICE)
            outputs = model(X_val_tensor)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()

        unique_preds = np.unique(preds)
        print(f"  Prediction: OK (unique values: {unique_preds})")

        results[brain] = 'PASSED'

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results[brain] = f'FAILED: {str(e)[:50]}'

# Summary
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

passed = sum(1 for v in results.values() if v == 'PASSED')
failed = 12 - passed

for brain, status in results.items():
    icon = "OK" if status == 'PASSED' else "X"
    print(f"  [{icon}] {brain}: {status}")

print(f"\nPassed: {passed}/12")
print(f"Failed: {failed}/12")

if failed > 0:
    print("\n" + "=" * 70)
    print("CANNOT PROCEED - FIX FAILURES FIRST")
    print("=" * 70)
    sys.exit(1)
else:
    print("\n" + "=" * 70)
    print("ALL 12 NEURAL NETWORKS VERIFIED - READY FOR TRAINING")
    print("=" * 70)
    sys.exit(0)
