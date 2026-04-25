"""
Interpretability comparison on one dataset (default: medical_insurance).

For the *same* training data we compute four per-feature importance scores:

  1. AGOP diagonal     - extracted from every xRFM leaf, aggregated by leaf size.
  2. PCA loadings      - sum_k  explained_variance_ratio_k * loading_{k,j}^2
                         (how much of the data variance each feature carries).
  3. Mutual information- sklearn.feature_selection.mutual_info_regression.
  4. Permutation imp.  - drop in R^2 on a held-out set when feature j is
                         permuted (implemented here to wrap xRFM.predict).

We then compare the four rankings with:
  * Spearman rank-correlation matrix (written to CSV + heatmap PNG).
  * Top-K overlap (Jaccard) matrix.
  * Per-method top-K bar chart.
  * Raw + rank tables written to CSV.

Run:
    python interpretability_comparison.py
    python interpretability_comparison.py --dataset_dir <path> --top_k 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

from run_xrfm_utils import (
    _default_rfm_params,
    build_categorical_info_from_feature_names,
    load_split,
    metrics_regression,
)


# -----------------------------------------------------------------------------
# Helpers: importance extraction
# -----------------------------------------------------------------------------
def _to_numpy(t):
    if hasattr(t, "detach"):
        t = t.detach()
    if hasattr(t, "cpu"):
        t = t.cpu()
    if hasattr(t, "numpy"):
        t = t.numpy()
    return np.asarray(t)


def extract_leaf_agop_diagonals(model):
    """Return leaf_diags: (n_leaves, d), leaf_sizes: (n_leaves,)."""
    leaf_diags = []
    leaf_sizes = []
    for tree in model.trees:
        leaf_nodes = model._collect_leaf_nodes(tree)
        for node in leaf_nodes:
            agop = getattr(node["model"], "agop_best_model", None)
            if agop is None:
                agop = getattr(node["model"], "M", None)
            if agop is None:
                continue
            A = _to_numpy(agop)
            diag = np.diag(A) if A.ndim == 2 else np.asarray(A).reshape(-1)
            leaf_diags.append(diag.astype(np.float64))
            leaf_sizes.append(int(len(node.get("train_indices", [])) or 0))
    leaf_diags = np.vstack(leaf_diags)
    leaf_sizes = np.asarray(leaf_sizes, dtype=np.float64)
    if leaf_sizes.sum() == 0:
        leaf_sizes = np.ones_like(leaf_sizes)
    return leaf_diags, leaf_sizes


def agop_global_importance(leaf_diags, leaf_sizes):
    """Size-weighted mean of per-leaf AGOP diagonals."""
    w = leaf_sizes / leaf_sizes.sum()
    return (leaf_diags * w[:, None]).sum(axis=0)


def pca_loadings_importance(X, n_components=None):
    """
    Per-feature PCA-loading score:
        score_j = sum_k  explained_variance_ratio_k * loading_{k, j}^2
    This is proportional to how much of the overall data variance feature j
    contributes through the principal directions.
    """
    n_components = n_components or min(X.shape)
    pca = PCA(n_components=n_components, svd_solver="auto")
    pca.fit(X)
    L = pca.components_
    evr = pca.explained_variance_ratio_
    return (evr[:, None] * (L ** 2)).sum(axis=0), pca


def mutual_information_importance(X, y, seed=42):
    return mutual_info_regression(X, y, random_state=seed)


def permutation_importance_xrfm(predict_fn, X, y, n_repeats=5, seed=42, subsample=None):
    """
    Simple permutation importance using R^2 as the score (higher = better).
    `predict_fn` takes numpy X (n, d) and returns numpy y_hat (n,).
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape

    if subsample is not None and subsample < n:
        idx = rng.choice(n, size=subsample, replace=False)
        X = X[idx]
        y = y[idx]
        n = X.shape[0]

    base_pred = predict_fn(X)
    base_rmse, _, base_r2 = metrics_regression(y, base_pred)
    base_score = base_r2

    importances = np.zeros((d, n_repeats), dtype=np.float64)
    for j in range(d):
        for r in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            _, _, r2_p = metrics_regression(y, predict_fn(Xp))
            importances[j, r] = base_score - r2_p
        print(f"  [perm] feature {j + 1}/{d} done", end="\r", flush=True)
    print()
    return importances.mean(axis=1), importances.std(axis=1), base_score


# -----------------------------------------------------------------------------
# Helpers: comparison and reporting
# -----------------------------------------------------------------------------
def _rank_desc(x):
    """Rank from 1 (largest) to n (smallest). Ties use average rank."""
    x = np.asarray(x, dtype=np.float64)
    order = (-x).argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # Average-rank for ties
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros_like(uniq, dtype=np.float64)
        for i, r in zip(inv, ranks):
            sums[i] += r
        avg = sums / counts
        ranks = avg[inv]
    return ranks


def spearman_matrix(scores_dict):
    names = list(scores_dict.keys())
    k = len(names)
    M = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            rho, _ = spearmanr(scores_dict[names[i]], scores_dict[names[j]])
            M[i, j] = M[j, i] = rho
    return pd.DataFrame(M, index=names, columns=names)


def topk_jaccard_matrix(scores_dict, top_k):
    names = list(scores_dict.keys())
    tops = {
        n: set(np.argsort(-np.asarray(scores_dict[n]))[:top_k].tolist())
        for n in names
    }
    k = len(names)
    M = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = tops[names[i]], tops[names[j]]
            M[i, j] = M[j, i] = len(a & b) / max(len(a | b), 1)
    return pd.DataFrame(M, index=names, columns=names)


def aggregate_to_base_columns(scores, feature_names):
    """
    Combine one-hot `cat__<col>_<cat>` scores into a single `<col>` score by
    summing; keep `num__<col>` as `<col>`.
    """
    base_scores = {}
    for name, s in zip(feature_names, scores):
        if name.startswith("num__"):
            base = name[len("num__"):]
        elif name.startswith("cat__"):
            rest = name[len("cat__"):]
            base = rest.split("_", 1)[0]
        else:
            base = name
        base_scores[base] = base_scores.get(base, 0.0) + float(s)
    return base_scores


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def train_xrfm_and_get_predictor(X_train, y_train, X_val, y_val, feature_names, seed=42):
    import torch
    from xrfm import xRFM

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train.to_numpy(dtype=np.float32), device=device)
    X_val_t = torch.tensor(X_val.to_numpy(dtype=np.float32), device=device)
    y_train_t = torch.tensor(y_train.astype(np.float32), device=device).view(-1, 1)
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
    model.fit(X_train_t, y_train_t, X_val_t, y_val_t)

    def predict_fn(X_np):
        X_t = torch.tensor(np.asarray(X_np, dtype=np.float32), device=device)
        out = model.predict(X_t) if hasattr(model, "predict") else model.predict_proba(X_t)
        return _to_numpy(out).reshape(-1).astype(np.float64)

    return model, predict_fn, device


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
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Defaults to <dataset_dir>/interpretability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--perm_subsample", type=int, default=5000)
    parser.add_argument("--perm_repeats", type=int, default=5)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir) if args.out_dir else dataset_dir / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] loading split from {dataset_dir}")
    X_train, y_train, X_val, y_val, X_test, y_test = load_split(dataset_dir)
    feature_names = list(X_train.columns)
    d = len(feature_names)
    print(f"       n_train={len(X_train)}, n_val={len(X_val)}, n_test={len(X_test)}, d={d}")

    print("[2/5] training xRFM")
    model, predict_fn, device = train_xrfm_and_get_predictor(
        X_train, y_train, X_val, y_val, feature_names, seed=args.seed
    )

    print("[3/5] extracting AGOP diagonals from leaves")
    leaf_diags, leaf_sizes = extract_leaf_agop_diagonals(model)
    agop_score = agop_global_importance(leaf_diags, leaf_sizes)
    print(f"       n_leaves={len(leaf_sizes)},  sizes min/median/max = "
          f"{int(leaf_sizes.min())}/{int(np.median(leaf_sizes))}/{int(leaf_sizes.max())}")

    # save per-leaf AGOP diagonals for inspection
    leaf_df = pd.DataFrame(leaf_diags, columns=feature_names)
    leaf_df.insert(0, "leaf_size", leaf_sizes.astype(int))
    leaf_df.to_csv(out_dir / "agop_diagonals_per_leaf.csv", index=False)

    print("[4/5] computing PCA / MI / permutation importance")
    X_train_np = X_train.to_numpy(dtype=np.float64)
    X_test_np = X_test.to_numpy(dtype=np.float64)

    pca_score, _ = pca_loadings_importance(X_train_np)
    mi_score = mutual_information_importance(X_train_np, y_train, seed=args.seed)
    print("       permutation importance (this can take a few minutes)...")
    perm_score, perm_std, base_r2 = permutation_importance_xrfm(
        predict_fn,
        X_test_np,
        y_test.astype(np.float64),
        n_repeats=args.perm_repeats,
        seed=args.seed,
        subsample=args.perm_subsample,
    )
    print(f"       base test R^2 used by permutation importance: {base_r2:.4f}")

    scores = {
        "AGOP_diag": agop_score,
        "PCA_loadings": pca_score,
        "MutualInfo": mi_score,
        "PermImportance": perm_score,
    }

    # assemble raw score table + ranks
    score_df = pd.DataFrame(scores, index=feature_names)
    rank_df = pd.DataFrame(
        {k: _rank_desc(v) for k, v in scores.items()}, index=feature_names
    )
    score_df.to_csv(out_dir / "scores_raw.csv")
    rank_df.to_csv(out_dir / "scores_ranks.csv")

    # aggregate to base columns (combine one-hot groups)
    base_score_df = pd.DataFrame(
        {k: aggregate_to_base_columns(v, feature_names) for k, v in scores.items()}
    )
    base_score_df.to_csv(out_dir / "scores_raw_by_base_column.csv")

    print("[5/5] comparing rankings")
    sp = spearman_matrix(scores)
    jc = topk_jaccard_matrix(scores, args.top_k)
    sp.to_csv(out_dir / "spearman_rank_corr.csv")
    jc.to_csv(out_dir / f"topk_jaccard_k{args.top_k}.csv")

    # ---- plots ---------------------------------------------------------------
    # Spearman heatmap
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(sp.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(sp))); ax.set_xticklabels(sp.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(sp))); ax.set_yticklabels(sp.index)
    for i in range(len(sp)):
        for j in range(len(sp)):
            ax.text(j, i, f"{sp.values[i, j]:.2f}", ha="center", va="center",
                    color="black" if abs(sp.values[i, j]) < 0.6 else "white", fontsize=9)
    ax.set_title("Spearman rank-correlation\n(feature importance rankings)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "spearman_heatmap.png", dpi=160)
    plt.close(fig)

    # Top-K jaccard heatmap
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(jc.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(jc))); ax.set_xticklabels(jc.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(jc))); ax.set_yticklabels(jc.index)
    for i in range(len(jc)):
        for j in range(len(jc)):
            ax.text(j, i, f"{jc.values[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=9)
    ax.set_title(f"Top-{args.top_k} Jaccard overlap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / f"topk_jaccard_heatmap_k{args.top_k}.png", dpi=160)
    plt.close(fig)

    # Per-method top-K bar chart (using base-column aggregated scores for readability)
    names = list(scores.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 0.35 * args.top_k + 2))
    for ax, n in zip(axes, names):
        s = base_score_df[n].sort_values(ascending=False).head(args.top_k)
        # min-max normalize within method for comparable bar lengths
        smin, smax = float(s.min()), float(s.max())
        s_norm = (s - smin) / (smax - smin + 1e-12)
        ax.barh(s.index[::-1], s_norm.values[::-1])
        ax.set_title(n)
        ax.set_xlim(0, 1.05)
        ax.tick_params(axis="y", labelsize=9)
    fig.suptitle(f"Top-{args.top_k} features per method  (min-max normalized, base columns)")
    fig.tight_layout()
    fig.savefig(out_dir / f"top{args.top_k}_per_method.png", dpi=160)
    plt.close(fig)

    # small summary JSON
    def _top(s, k):
        return s.sort_values(ascending=False).head(k).index.tolist()

    summary = {
        "dataset_dir": str(dataset_dir),
        "n_features": d,
        "n_leaves": int(len(leaf_sizes)),
        "leaf_sizes": leaf_sizes.astype(int).tolist(),
        "base_r2_for_permutation": float(base_r2),
        "spearman": sp.round(4).to_dict(),
        "topk_jaccard": jc.round(4).to_dict(),
        f"top{args.top_k}_per_method_base": {
            n: _top(base_score_df[n], args.top_k) for n in names
        },
        f"top{args.top_k}_per_method_processed": {
            n: score_df[n].sort_values(ascending=False).head(args.top_k).index.tolist()
            for n in names
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Spearman rank correlation ===")
    print(sp.round(3).to_string())
    print(f"\n=== Top-{args.top_k} Jaccard overlap ===")
    print(jc.round(3).to_string())
    print(f"\nTop-{args.top_k} features (base columns) by method:")
    for n in names:
        print(f"  [{n}] {', '.join(_top(base_score_df[n], args.top_k))}")
    print(f"\nAll artifacts saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
