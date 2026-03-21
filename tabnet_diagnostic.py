#!/usr/bin/env python3
"""
TabNet Diagnostic Test - Run this on Colab before training
============================================================
This script tests TabNet performance with various data sizes.
Run this FIRST to verify TabNet works correctly on your Colab instance.
"""

import time
import torch
import numpy as np

print("=" * 70)
print("TABNET DIAGNOSTIC TEST")
print("=" * 70)

# Check CUDA
print(f"\n[1/7] CUDA Check...")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
    print(f"  Compute capability: {props.major}.{props.minor}")
else:
    print("  WARNING: No CUDA device found!")
    print("  Make sure you selected GPU runtime in Colab")
    print("  Runtime -> Change runtime type -> GPU")

# Import TabNet
print(f"\n[2/7] Import TabNet...")
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    print("  TabNet imported successfully")
except ImportError as e:
    print(f"  FAILED: {e}")
    print("  Run: pip install pytorch-tabnet")
    exit(1)

# Create small test data
print(f"\n[3/7] Creating test data (10K rows)...")
np.random.seed(42)
X_small = np.random.randn(10000, 100).astype(np.float32)
y_small = np.random.randint(0, 3, 10000)
print(f"  X shape: {X_small.shape}")
print(f"  y distribution: {np.bincount(y_small)}")

# Test TabNet initialization
print(f"\n[4/7] Testing TabNet init...")
start = time.time()
model = TabNetClassifier(
    n_d=8,
    n_a=8,
    n_steps=3,
    gamma=1.5,
    n_independent=1,
    n_shared=1,
    seed=42,
    device_name='cuda',
    verbose=0,
)
init_time = time.time() - start
print(f"  Init time: {init_time:.2f}s")

# Test small data training
print(f"\n[5/7] Training on 10K rows (5 epochs)...")
start = time.time()

try:
    model.fit(
        X_small, y_small,
        max_epochs=5,
        patience=3,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )
    train_time = time.time() - start
    print(f"  Train time (5 epochs): {train_time:.2f}s")
    print(f"  Per epoch: {train_time/5:.2f}s")

    # Test prediction
    preds = model.predict(X_small[:100])
    print(f"  Predictions: {len(preds)} samples, unique values: {np.unique(preds)}")

    print(f"  PASSED")
    small_test_passed = True

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()
    small_test_passed = False

# Test with 500K rows (simulating actual training)
print(f"\n[6/7] Creating large test data (500K rows, 275 features)...")
X_large = np.random.randn(500000, 275).astype(np.float32)
y_large = np.random.randint(0, 3, 500000)
print(f"  X shape: {X_large.shape}")
print(f"  Memory needed: ~{X_large.nbytes / 1e9:.2f} GB")
print(f"  y distribution: {np.bincount(y_large)}")

print(f"\n[7/7] Training 1 epoch on 500K rows...")
model2 = TabNetClassifier(
    n_d=8,
    n_a=8,
    n_steps=3,
    gamma=1.5,
    n_independent=1,
    n_shared=1,
    seed=42,
    device_name='cuda',
    verbose=1,  # Show progress
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
)

start = time.time()

try:
    model2.fit(
        X_large, y_large,
        max_epochs=1,
        patience=1,
        batch_size=4096,       # Large batch for GPU efficiency
        virtual_batch_size=256,
        num_workers=0,
        drop_last=False,
    )
    train_time = time.time() - start
    print(f"\n  1 epoch time: {train_time:.2f}s")

    # Test prediction
    start_pred = time.time()
    preds = model2.predict(X_large[:10000])
    pred_time = time.time() - start_pred
    print(f"  Prediction time (10K): {pred_time:.2f}s")

    large_test_passed = True
    large_epoch_time = train_time

except Exception as e:
    print(f"\n  FAILED: {e}")
    import traceback
    traceback.print_exc()
    large_test_passed = False
    large_epoch_time = 999

# Summary
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

print(f"\n  CUDA Available:      {'YES' if torch.cuda.is_available() else 'NO'}")
print(f"  Small Test (10K):    {'PASSED' if small_test_passed else 'FAILED'}")
print(f"  Large Test (500K):   {'PASSED' if large_test_passed else 'FAILED'}")

if large_test_passed:
    print(f"  1 Epoch Time (500K): {large_epoch_time:.1f}s")

    # Estimate full training time
    # Optuna: 50 trials x 50 epochs each (with early stopping ~20 avg)
    estimated_trial_time = large_epoch_time * 20  # Avg epochs per trial
    estimated_optuna_time = estimated_trial_time * 50 / 60  # 50 trials in minutes

    print(f"\n  ESTIMATED TRAINING TIME:")
    print(f"    Per Optuna trial: ~{estimated_trial_time/60:.1f} min")
    print(f"    Full optimization (50 trials): ~{estimated_optuna_time:.0f} min")

if large_test_passed and large_epoch_time < 60:
    print(f"\n  READY FOR TRAINING")
elif large_test_passed and large_epoch_time < 180:
    print(f"\n  ACCEPTABLE - Training will be slow but workable")
else:
    print(f"\n  NOT READY - TabNet is too slow or failing")
    print(f"  Check GPU allocation and batch size settings")

print("=" * 70)

# Output final verdict
if torch.cuda.is_available() and small_test_passed and large_test_passed and large_epoch_time < 120:
    print("\nVERDICT: TabNet is ready for production training")
    exit(0)
else:
    print("\nVERDICT: TabNet needs investigation before training")
    exit(1)
