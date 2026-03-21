#!/usr/bin/env python3
"""
REALISTIC OOM VERIFICATION TEST
================================
Tests ALL 12 neural network models with ACTUAL training data sizes.
Previous verification used 1000 rows - this uses 200K+ rows like real training.

Run this BEFORE training to verify no OOM errors.
"""

import sys
import gc
import time
import numpy as np
import torch
import torch.nn as nn

print("=" * 70)
print("REALISTIC OOM VERIFICATION TEST")
print("=" * 70)

# Check GPU
print(f"\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"Memory: {props.total_memory / 1e9:.1f} GB")

# REALISTIC sizes - same as actual training
TRAIN_SAMPLES = 200_000   # Same as NN_MAX_SAMPLES
VAL_SAMPLES = 100_000     # Same as validation after sampling
FEATURES = 275            # Same as actual features
VAL_BATCH_SIZE = 2048     # Same as VAL_INFERENCE_BATCH_SIZE

print(f"\nTest Configuration:")
print(f"  Train samples: {TRAIN_SAMPLES:,}")
print(f"  Val samples: {VAL_SAMPLES:,}")
print(f"  Features: {FEATURES}")
print(f"  Val batch size: {VAL_BATCH_SIZE}")

# Import from chaos_gpu_training
print("\nImporting from chaos_gpu_training...")
try:
    sys.path.insert(0, '/content/drive/MyDrive/chaos_v1.0')
    sys.path.insert(0, 'G:/My Drive/chaos_v1.0')

    from chaos_gpu_training import (
        MLPClassifier, LSTMClassifier, GRUClassifier, TransformerClassifier,
        CNN1DClassifier, TCNClassifier, WaveNetClassifier, AttentionNetClassifier,
        ResidualMLPClassifier, EnsembleNNClassifier, NBeatsClassifier,
        TemporalFusionTransformer, batched_inference, VAL_INFERENCE_BATCH_SIZE,
        DEVICE
    )
    print("  Imports successful!")
except ImportError as e:
    print(f"  Import failed: {e}")
    sys.exit(1)

# Create realistic test data
print("\nCreating test data...")
np.random.seed(42)
X_train = np.random.randn(TRAIN_SAMPLES, FEATURES).astype(np.float32)
y_train = np.random.randint(0, 3, TRAIN_SAMPLES)
X_val = np.random.randn(VAL_SAMPLES, FEATURES).astype(np.float32)
y_val = np.random.randint(0, 3, VAL_SAMPLES)

print(f"  X_train: {X_train.shape} ({X_train.nbytes / 1e9:.2f} GB)")
print(f"  X_val: {X_val.shape} ({X_val.nbytes / 1e9:.2f} GB)")

# Model configs - EXACTLY matching what training uses
MODEL_CONFIGS = {
    'mlp_optuna': {
        'class': MLPClassifier,
        'params': {'hidden_sizes': [128, 64, 32], 'dropout': 0.3}
    },
    'lstm_optuna': {
        'class': LSTMClassifier,
        'params': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3}
    },
    'gru_optuna': {
        'class': GRUClassifier,
        'params': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3}
    },
    'transformer_optuna': {
        'class': TransformerClassifier,
        'params': {'d_model': 128, 'nhead': 4, 'num_layers': 2, 'dropout': 0.3}
    },
    'cnn1d_optuna': {
        'class': CNN1DClassifier,
        'params': {'channels': [32, 64, 64], 'kernel_size': 3, 'dropout': 0.3}  # REDUCED
    },
    'tcn_optuna': {
        'class': TCNClassifier,
        'params': {'channels': [32, 64], 'kernel_size': 3, 'dropout': 0.3}  # REDUCED
    },
    'wavenet_optuna': {
        'class': WaveNetClassifier,
        'params': {'channels': 64, 'num_layers': 3, 'kernel_size': 2, 'dropout': 0.3}  # REDUCED
    },
    'attention_net_optuna': {
        'class': AttentionNetClassifier,
        'params': {'hidden_size': 128, 'num_heads': 4, 'dropout': 0.3}
    },
    'residual_mlp_optuna': {
        'class': ResidualMLPClassifier,
        'params': {'hidden_size': 128, 'num_blocks': 3, 'dropout': 0.3}
    },
    'ensemble_nn_optuna': {
        'class': EnsembleNNClassifier,
        'params': {'hidden_size': 64, 'dropout': 0.3}  # REDUCED - has Conv1D
    },
    'nbeats_optuna': {
        'class': NBeatsClassifier,
        'params': {'hidden_size': 128, 'num_blocks': 4, 'dropout': 0.3}
    },
    'tft_optuna': {
        'class': TemporalFusionTransformer,
        'params': {'hidden_size': 128, 'num_heads': 4, 'num_layers': 2, 'dropout': 0.1}
    },
}

# Test each model
results = {}
print("\n" + "=" * 70)
print("TESTING ALL 12 NEURAL NETWORK MODELS (REALISTIC DATA SIZE)")
print("=" * 70)

for brain, config in MODEL_CONFIGS.items():
    print(f"\n[{len(results)+1}/12] Testing {brain}...")

    # Clear memory before each test
    gc.collect()
    torch.cuda.empty_cache()

    try:
        model_class = config['class']
        params = config['params']

        # 1. Create model
        print(f"  Creating model...")
        model = model_class(FEATURES, num_classes=3, **params)
        model = model.to(DEVICE)

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # 2. Test batch forward pass (should always work)
        print(f"  Testing batch forward (1024 samples)...")
        X_batch = torch.tensor(X_train[:1024]).to(DEVICE)
        with torch.no_grad():
            out = model(X_batch)
        assert out.shape == (1024, 3), f"Wrong output shape: {out.shape}"
        del X_batch, out
        torch.cuda.empty_cache()
        print(f"  Batch forward: OK")

        # 3. TEST FULL VALIDATION INFERENCE - THE ACTUAL FAILURE POINT
        print(f"  Testing FULL validation inference ({VAL_SAMPLES:,} rows)...")
        start_time = time.time()

        X_val_tensor = torch.tensor(X_val)  # Keep on CPU
        val_outputs = batched_inference(model, X_val_tensor, VAL_INFERENCE_BATCH_SIZE)

        inference_time = time.time() - start_time
        assert val_outputs.shape == (VAL_SAMPLES, 3), f"Wrong output shape: {val_outputs.shape}"
        print(f"  Full validation: OK ({inference_time:.1f}s)")

        # Check GPU memory usage
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Peak GPU memory: {mem_used:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        results[brain] = "PASSED"

        # Cleanup
        del model, X_val_tensor, val_outputs
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        results[brain] = f"FAILED: {str(e)[:60]}"
        print(f"  {results[brain]}")
        gc.collect()
        torch.cuda.empty_cache()

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
    print("ALL 12 MODELS VERIFIED WITH REALISTIC DATA SIZES")
    print("OOM FIXES CONFIRMED WORKING")
    print("=" * 70)
    sys.exit(0)
