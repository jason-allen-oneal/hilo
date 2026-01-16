# lib/model/train_eval.py
"""
End-to-end pipeline to build the dataset, tune C, train, calibrate, and evaluate.
Run:
  python -m lib.model.train_eval
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from lib.model.dataset import build_round_dataset
from lib.model.train import build_model, load_dataset


def parse_c_grid(raw: str) -> List[float]:
    vals: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("C grid cannot be empty")
    return vals


def choose_time_series_splits(n_samples: int) -> int:
    # Keep splits small to respect chronology and avoid tiny test windows.
    return max(2, min(5, n_samples - 1))


def crossval_score(
    X: np.ndarray,
    y: np.ndarray,
    Cs: Iterable[float],
    class_weight,
    model_type: str = "xgboost",
) -> Tuple[float, float]:
    """
    Returns (best_C, best_auc).
    Selection: highest mean AUC, tie-broken by lowest log loss.
    """
    n_splits = choose_time_series_splits(len(y))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    best: Tuple[float, float, float] | None = None  # (auc, -logloss, C)

    for C in Cs:
        aucs: List[float] = []
        losses: List[float] = []

        for train_idx, val_idx in splitter.split(X):
            model = build_model(C, class_weight=class_weight, model_type=model_type)
            model.fit(X[train_idx], y[train_idx])

            probs = model.predict_proba(X[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], probs))
            losses.append(log_loss(y[val_idx], probs, labels=[0, 1]))

        mean_auc = float(np.mean(aucs))
        mean_ll = float(np.mean(losses))

        score = (mean_auc, -mean_ll, C)
        if best is None or score > best:
            best = score

    assert best is not None
    _, _, best_C = best
    return best_C, best[0]


def reliability_bins(probs: np.ndarray, y: np.ndarray, bins: int = 5) -> List[Tuple[float, float, int]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    results: List[Tuple[float, float, int]] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        mean_pred = float(probs[mask].mean())
        mean_actual = float(y[mask].mean())
        results.append((mean_pred, mean_actual, int(mask.sum())))
    return results


def evaluate_model(model, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float, float, List[Tuple[float, float, int]]]:
    probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)
    ll = log_loss(y_val, probs, labels=[0, 1])
    brier = brier_score_loss(y_val, probs)
    bins = reliability_bins(probs, y_val)
    return auc, ll, brier, bins


def main() -> int:
    ap = argparse.ArgumentParser(description="Build dataset, tune C, train, calibrate, and evaluate.")
    ap.add_argument("--candles", type=Path, default=Path("data/historical_1m.csv"), help="Input 1m candle CSV")
    ap.add_argument("--dataset", type=Path, default=Path("data/round_dataset.csv"), help="Intermediate dataset CSV")
    ap.add_argument("--model-out", type=Path, default=Path("lib/model/model.joblib"), help="Output joblib path")
    ap.add_argument("--buffer-size", type=int, default=60, help="Rolling candle buffer size")
    ap.add_argument("--decision-window-minutes", type=int, default=2, help="Minutes to decide direction after open")
    ap.add_argument("--test-fraction", type=float, default=0.2, help="Hold-out fraction from the end for evaluation")
    ap.add_argument("--C-grid", type=str, default="0.1,0.5,1.0,2.0,5.0", help="Comma-separated C values to try")
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none", help="Class weight strategy")
    ap.add_argument("--calibrate", action="store_true", help="Calibrate probabilities with sigmoid scaling")
    ap.add_argument("--model-type", choices=["logistic", "xgboost"], default="xgboost", help="Model type to train")
    args = ap.parse_args()

    print("[STEP] Building dataset...")
    dataset_path = build_round_dataset(
        input_csv=args.candles,
        output_csv=args.dataset,
        buffer_size=args.buffer_size,
        decision_window_minutes=args.decision_window_minutes,
    )

    X, y = load_dataset(dataset_path)
    n = len(y)
    up = int(y.sum())
    down = n - up
    up_rate = up / n

    print(f"[INFO] Loaded dataset: {dataset_path}")
    print(f"[INFO] Rows: {n} | UP=1: {up} | DOWN=0: {down} | UP rate: {up_rate:.3f}")

    if not (0.05 <= args.test_fraction <= 0.4):
        raise ValueError("test-fraction should be between 0.05 and 0.4 for sensible evaluation windows")

    val_size = max(1, int(n * args.test_fraction))
    train_size = n - val_size
    if train_size < 50:
        raise RuntimeError(f"Not enough training samples after split: train={train_size}, val={val_size}")

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    Cs = parse_c_grid(args.C_grid)
    class_weight = None if args.class_weight == "none" else "balanced"

    print("[STEP] Tuning C with time-series CV...")
    best_C, best_auc = crossval_score(X_train, y_train, Cs=Cs, class_weight=class_weight, model_type=args.model_type)
    print(f"[OK] Best C={best_C} (mean CV AUC={best_auc:.4f})")

    if args.calibrate:
        splits = choose_time_series_splits(len(y_train))
        calibrator = CalibratedClassifierCV(
            estimator=build_model(best_C, class_weight=class_weight, model_type=args.model_type),
            cv=TimeSeriesSplit(n_splits=splits),
            method="sigmoid",
        )
        model = calibrator.fit(X_train, y_train)
    else:
        model = build_model(best_C, class_weight=class_weight, model_type=args.model_type).fit(X_train, y_train)

    print("[STEP] Evaluating on hold-out tail...")
    auc, ll, brier, bins = evaluate_model(model, X_val, y_val)
    print(f"[METRIC] Hold-out AUC:   {auc:.4f}")
    print(f"[METRIC] Hold-out LogLoss:{ll:.4f}")
    print(f"[METRIC] Hold-out Brier:  {brier:.4f}")
    print("[METRIC] Calibration bins (pred -> actual, count):")
    for mean_pred, mean_actual, count in bins:
        print(f"  {mean_pred: .3f} -> {mean_actual: .3f} (n={count})")

    print("[STEP] Training final model on full dataset...")
    if args.calibrate:
        splits = choose_time_series_splits(len(y))
        final_model = CalibratedClassifierCV(
            estimator=build_model(best_C, class_weight=class_weight, model_type=args.model_type),
            cv=TimeSeriesSplit(n_splits=splits),
            method="sigmoid",
        ).fit(X, y)
    else:
        final_model = build_model(best_C, class_weight=class_weight, model_type=args.model_type).fit(X, y)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.model_out)
    print(f"[OK] Saved model to {args.model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
