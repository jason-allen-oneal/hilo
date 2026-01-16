# lib/model/train.py

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from lib.model.features import FEATURES


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Run the dataset builder first to generate it."
        )

    X_rows: List[List[float]] = []
    y_rows: List[int] = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        required = set(FEATURES + ["label"])
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

        for i, row in enumerate(reader, start=1):
            feats = [float(row[k]) for k in FEATURES]
            label = int(row["label"])

            if label not in (0, 1):
                raise ValueError(f"Bad label in row #{i}: {label} (expected 0 or 1)")

            if not np.all(np.isfinite(feats)):
                continue

            X_rows.append(feats)
            y_rows.append(label)

    if len(X_rows) < 200:
        raise RuntimeError(
            f"Not enough rows to train: {len(X_rows)}. "
            "You need more historical data."
        )

    return np.asarray(X_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int64)


def build_logistic_model(C: float, class_weight=None) -> Pipeline:
    # LogisticRegression defaults to L2 penalty already.
    # We explicitly set only stable, non-deprecated params.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                C=C,
                class_weight=class_weight,
                random_state=42,
            )),
        ]
    )


def build_xgboost_model(class_weight=None) -> Pipeline:
    # XGBoost with optimized hyperparameters for 15-minute BTC direction prediction
    # XGBoost handles feature scaling internally, but we keep StandardScaler for consistency
    # Note: scale_pos_weight should be set dynamically during training based on actual class distribution
    # For now, we don't set it and let XGBoost handle it internally
    
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=42,
            )),
        ]
    )


def build_model(C: float = 1.0, class_weight=None, model_type: str = "xgboost") -> Pipeline:
    """
    Build a model pipeline based on model_type.
    
    Args:
        C: Inverse regularization strength (used only for logistic regression)
        class_weight: Class weight strategy ('balanced' or None)
        model_type: Type of model ('logistic' or 'xgboost')
    
    Returns:
        sklearn Pipeline with scaler and classifier
    """
    if model_type == "logistic":
        return build_logistic_model(C, class_weight)
    elif model_type == "xgboost":
        return build_xgboost_model(class_weight)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'logistic' or 'xgboost'.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train round direction model from CSV dataset.")
    ap.add_argument("--data", type=Path, default=Path("data/round_dataset.csv"), help="Path to dataset CSV")
    ap.add_argument("--out", type=Path, default=Path("lib/model/model.joblib"), help="Output joblib path")
    ap.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength (higher = weaker reg)")
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none", help="Class weight strategy")
    ap.add_argument("--model-type", choices=["logistic", "xgboost"], default="xgboost", help="Model type to train")
    args = ap.parse_args()

    X, y = load_dataset(args.data)

    n = len(y)
    up = int(y.sum())
    down = n - up
    up_rate = up / n

    print(f"[INFO] Loaded dataset: {args.data}")
    print(f"[INFO] Rows: {n} | UP=1: {up} | DOWN=0: {down} | UP rate: {up_rate:.3f}")

    class_weight = None if args.class_weight == "none" else "balanced"

    model = build_model(args.C, class_weight=class_weight, model_type=args.model_type)
    model.fit(X, y)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"[OK] Model trained and saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
