"""
Debug script to check feature statistics and find problematic features.
"""
import csv
from pathlib import Path
import numpy as np

def analyze_features(csv_path: Path):
    """Analyze feature statistics from dataset."""
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        # Collect all values per feature
        feature_data = {name: [] for name in fieldnames if name != 'label'}
        
        for row in reader:
            for name in feature_data.keys():
                try:
                    val = float(row[name])
                    feature_data[name].append(val)
                except (ValueError, KeyError):
                    pass
    
    # Print statistics
    print("\n=== FEATURE STATISTICS ===\n")
    
    for feature, values in sorted(feature_data.items()):
        if not values:
            print(f"{feature:30s} | NO DATA")
            continue
            
        arr = np.array(values)
        
        has_nan = np.isnan(arr).any()
        has_inf = np.isinf(arr).any()
        
        status = "⚠️  INVALID" if has_nan or has_inf else "✓"
        
        print(f"{feature:30s} | {status:10s} | "
              f"min={np.nanmin(arr):10.4f} | "
              f"max={np.nanmax(arr):10.4f} | "
              f"mean={np.nanmean(arr):10.4f} | "
              f"std={np.nanstd(arr):10.4f}")
        
        if has_nan:
            print(f"  └─ NaN count: {np.isnan(arr).sum()}")
        if has_inf:
            print(f"  └─ Inf count: {np.isinf(arr).sum()}")

if __name__ == "__main__":
    analyze_features(Path("data/round_dataset.csv"))
