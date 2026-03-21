"""
VERIFICATION: ALL 63 FEATURE FILES
"""
from pathlib import Path
import pandas as pd

features_dir = Path(r"G:\My Drive\chaos_v1.0\features")
pairs = ['EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY']
tfs = ['M15','M30','H1','H4','D1','W1','MN1']

print("=" * 70)
print("FILE VERIFICATION - ALL 63 COMBINATIONS")
print("=" * 70)

missing = []
present = []

for pair in pairs:
    for tf in tfs:
        fpath = features_dir / f"{pair}_{tf}_features.parquet"
        if fpath.exists():
            df = pd.read_parquet(fpath)
            has_target = 'target_3class_8' in df.columns
            has_returns = 'target_return_8' in df.columns
            status = "OK" if (has_target and has_returns) else "MISSING COLUMNS"
            present.append({
                'file': f"{pair}_{tf}_features.parquet",
                'rows': len(df),
                'cols': len(df.columns),
                'target_3class_8': has_target,
                'target_return_8': has_returns,
                'status': status
            })
        else:
            missing.append(f"{pair}_{tf}_features.parquet")

print(f"\nPRESENT FILES: {len(present)}/63")
print("-" * 70)
for p in present:
    mark = "OK" if p['status'] == "OK" else "X"
    print(f"  [{mark}] {p['file']}: {p['rows']} rows, {p['cols']} cols")

if missing:
    print(f"\nMISSING FILES: {len(missing)}")
    print("-" * 70)
    for m in missing:
        print(f"  X {m}")
else:
    print(f"\n*** ALL 63 FILES PRESENT ***")

# Column verification
print("\n" + "=" * 70)
print("COLUMN VERIFICATION")
print("=" * 70)

all_have_target = all(p['target_3class_8'] for p in present)
all_have_returns = all(p['target_return_8'] for p in present)

print(f"target_3class_8 in all files: {'YES' if all_have_target else 'NO'}")
print(f"target_return_8 in all files: {'YES' if all_have_returns else 'NO'}")

print("\n" + "=" * 70)
print(f"FINAL RESULT: {len(present)}/63 files present")
if len(present) == 63 and all_have_target and all_have_returns:
    print("STATUS: ALL 63 FILES PRESENT WITH CORRECT COLUMNS")
else:
    print("STATUS: INCOMPLETE")
print("=" * 70)
