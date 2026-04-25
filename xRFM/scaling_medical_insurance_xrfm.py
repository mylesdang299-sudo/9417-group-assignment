"""
Scaling experiment on medical_insurance:
train xRFM on subsets of size {1k, 2k, 5k, 10k, 20k, 50k} of the training data
and record test RMSE + training time.

Usage:
    python scaling_medical_insurance_xrfm.py
    python scaling_medical_insurance_xrfm.py --sizes 1000 2000 5000 10000 20000 50000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_xrfm_utils import (
    _default_rfm_params,
    build_categorical_info_from_feature_names,
    load_split,
    metrics_regression,
)


def run_subset(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, n, seed):
    import torch
    from xrfm import xRFM

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Subsample training set (keep val/test fixed)
    rng = np.random.default_rng(seed)
    n_full = len(X_train)
    n_use = min(n, n_full)
    idx = rng.choice(n_full, size=n_use, replace=False)
    X_sub = X_train.iloc[idx].reset_index(drop=True)
    y_sub = y_train[idx]

    X_train_t = torch.tensor(X_sub.to_numpy(dtype=np.float32), device=device)
    X_val_t = torch.tensor(X_val.to_numpy(dtype=np.float32), device=device)
    X_test_t = torch.tensor(X_test.to_numpy(dtype=np.float32), device=device)
    y_train_t = torch.tensor(y_sub.astype(np.float32), device=device).view(-1, 1)
    y_val_t = torch.tensor(y_val.astype(np.float32), device=device).view(-1, 1)

    categorical_info = build_categorical_info_from_feature_names(feature_names)
    model = xRFM(
        rfm_params=_default_rfm_params(),
        device=device,
        tuning_metric="mse",
        split_method="random_pca",
        max_leaf_size=5_000,
        categorical_info=categorical_info,
    )

    t0 = time.perf_counter()
    model.fit(X_train_t, y_train_t, X_val_t, y_val_t)
    train_seconds = time.perf_counter() - t0

    y_pred = model.predict(X_test_t) if hasattr(model, "predict") else model.predict_proba(X_test_t)
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.cpu()
    if hasattr(y_pred, "numpy"):
        y_pred = y_pred.numpy()
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.float64)

    rmse, _, _ = metrics_regression(y_test.astype(np.float64), y_pred)
    return n_use, rmse, train_seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(
            Path("COMP9417_GroupProject_Datasets(1)")
            / "COMP9417_GroupProject_Datasets"
            / "outputs"
            / "medical_insurance"
        ),
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 2000, 5000, 10000, 20000, 50000],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default="xrfm_scaling_medical_insurance.csv")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = load_split(dataset_dir)
    feature_names = list(X_train.columns)
    print(f"loaded: n_train={len(X_train)}, n_val={len(X_val)}, n_test={len(X_test)}, d={len(feature_names)}")

    rows = []
    print(f"\n{'n':>10} {'test_RMSE':>12} {'train_time_s':>14}")
    print("-" * 40)
    for n in args.sizes:
        n_used, rmse, ts = run_subset(
            X_train, y_train, X_val, y_val, X_test, y_test, feature_names, n, seed=args.seed
        )
        print(f"{n_used:>10d} {rmse:>12.4f} {ts:>14.4f}")
        rows.append({"n": n_used, "test_rmse": rmse, "train_time_s": ts})

    out = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out.to_csv(out_path, index=False)
    print(f"\nsaved -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
