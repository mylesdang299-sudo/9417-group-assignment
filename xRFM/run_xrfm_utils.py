import time
from pathlib import Path

import numpy as np
import pandas as pd

def load_split(dataset_dir: Path):
    X_train = pd.read_csv(dataset_dir / "X_train_processed.csv")
    X_val = pd.read_csv(dataset_dir / "X_val_processed.csv")
    X_test = pd.read_csv(dataset_dir / "X_test_processed.csv")
    y_train = pd.read_csv(dataset_dir / "y_train.csv").iloc[:, 0].to_numpy()
    y_val = pd.read_csv(dataset_dir / "y_val.csv").iloc[:, 0].to_numpy()
    y_test = pd.read_csv(dataset_dir / "y_test.csv").iloc[:, 0].to_numpy()

    if list(X_train.columns) != list(X_val.columns) or list(X_train.columns) != list(X_test.columns):
        raise ValueError("Feature columns mismatch between train/val/test processed splits.")

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_categorical_info_from_feature_names(feature_names):
    """
    Reconstruct xRFM `categorical_info` from sklearn ColumnTransformer feature names.

    Expected pattern from preprocessing notebook:
    - numerical: "num__<col>"
    - one-hot:  "cat__<col>_<category>"
    """
    import torch

    num_idxs = []
    cat_groups = {}  # base_col -> [feature_idx,...]

    for i, name in enumerate(feature_names):
        if name.startswith("num__"):
            num_idxs.append(i)
            continue
        if name.startswith("cat__"):
            rest = name[len("cat__") :]
            base = rest.split("_", 1)[0]
            cat_groups.setdefault(base, []).append(i)
            continue

    numerical_indices = torch.tensor(num_idxs, dtype=torch.long)
    categorical_indices = []
    categorical_vectors = []

    for base_col in sorted(cat_groups.keys()):
        idxs = torch.tensor(cat_groups[base_col], dtype=torch.long)
        categorical_indices.append(idxs)
        categorical_vectors.append(torch.eye(len(idxs), dtype=torch.float32))

    return dict(
        numerical_indices=numerical_indices,
        categorical_indices=categorical_indices,
        categorical_vectors=categorical_vectors,
    )


def metrics_binary(y_true, y_prob, threshold=0.5):
    from sklearn.metrics import accuracy_score, roc_auc_score

    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    return acc, auc


def metrics_regression(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def _default_rfm_params():
    return {
        "model": {
            "kernel": "l2_high_dim",
            "bandwidth": 10.0,
            "exponent": 1.0,
            "diag": True,
            "bandwidth_mode": "constant",
        },
        "fit": {
            "reg": 1e-3,
            "iters": 1,
            "verbose": False,
            "early_stop_rfm": True,
            "M_batch_size": 256,
            "total_points_to_sample": 2048,
        },
    }

def _to_numpy_1d(x):
    """
    xRFM outputs may be either torch.Tensor or numpy.ndarray depending on version/config.
    Convert both safely into a 1D numpy array.
    """
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    x = np.asarray(x)
    return x.reshape(-1)


def run_xrfm_classification(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_names,
    seed=42,
    tuning_metric="auc",
):
    import torch
    from xrfm import xRFM

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train.to_numpy(dtype=np.float32), device=device)
    X_val_t = torch.tensor(X_val.to_numpy(dtype=np.float32), device=device)
    X_test_t = torch.tensor(X_test.to_numpy(dtype=np.float32), device=device)

    y_train_t = torch.tensor(y_train.astype(np.float32), device=device).view(-1, 1)
    y_val_t = torch.tensor(y_val.astype(np.float32), device=device).view(-1, 1)

    categorical_info = build_categorical_info_from_feature_names(feature_names)

    model = xRFM(
        rfm_params=_default_rfm_params(),
        device=device,
        tuning_metric=tuning_metric,
        split_method="random_pca",
        max_leaf_size=5_000,
        categorical_info=categorical_info,
    )

    t0 = time.perf_counter()
    model.fit(X_train_t, y_train_t, X_val_t, y_val_t)
    train_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_prob = model.predict_proba(X_test_t)
    infer_seconds = time.perf_counter() - t1

    y_prob_np = np.asarray(y_prob.detach().float().cpu().numpy()) if hasattr(y_prob, "detach") else np.asarray(y_prob)
    if y_prob_np.ndim == 2 and y_prob_np.shape[1] == 2:
        y_prob_np = y_prob_np[:, 1]
    y_prob_np = y_prob_np.reshape(-1)

    acc, auc = metrics_binary(y_test, y_prob_np)
    return {
        "model": "xRFM",
        "task_type": "classification",
        "accuracy": float(acc),
        "auroc": float(auc),
        "train_seconds": float(train_seconds),
        "infer_seconds": float(infer_seconds),
        "device": str(device),
    }


def run_xrfm_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_names,
    seed=42,
    tuning_metric="mse",
):
    import torch
    from xrfm import xRFM

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train.to_numpy(dtype=np.float32), device=device)
    X_val_t = torch.tensor(X_val.to_numpy(dtype=np.float32), device=device)
    X_test_t = torch.tensor(X_test.to_numpy(dtype=np.float32), device=device)

    y_train_t = torch.tensor(y_train.astype(np.float32), device=device).view(-1, 1)
    y_val_t = torch.tensor(y_val.astype(np.float32), device=device).view(-1, 1)

    categorical_info = build_categorical_info_from_feature_names(feature_names)

    model = xRFM(
        rfm_params=_default_rfm_params(),
        device=device,
        tuning_metric=tuning_metric,
        split_method="random_pca",
        max_leaf_size=5_000,
        categorical_info=categorical_info,
    )

    t0 = time.perf_counter()
    model.fit(X_train_t, y_train_t, X_val_t, y_val_t)
    train_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    # Some xRFM versions expose predict() for regression; keep a safe fallback.
    if hasattr(model, "predict"):
        y_pred_t = model.predict(X_test_t)
    else:
        y_pred_t = model.predict_proba(X_test_t)
    infer_seconds = time.perf_counter() - t1

    y_pred_np = _to_numpy_1d(y_pred_t)
    rmse, mae, r2 = metrics_regression(y_test.astype(np.float64), y_pred_np.astype(np.float64))

    return {
        "model": "xRFM",
        "task_type": "regression",
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "train_seconds": float(train_seconds),
        "infer_seconds": float(infer_seconds),
        "device": str(device),
    }

