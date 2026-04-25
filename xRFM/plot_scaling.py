"""
Plot test performance and training time vs n for xRFM / XGBoost / LightGBM / RF
on medical_insurance.

Data is hardcoded from scaling.csv (the 4-block version shown in the editor),
so the script runs regardless of what encoding / layout the CSV is saved in.

Output:
    scaling_rmse_vs_n.png
    scaling_time_vs_n.png
    scaling_combined.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA = {
    "LGBM": [
        (1000, 3073.5702, 0.0608),
        (2000, 2966.2046, 0.0871),
        (5000, 2931.9855, 0.0982),
        (10000, 2919.3076, 0.1174),
        (20000, 2915.6468, 0.1498),
        (50000, 2909.6604, 0.2252),
    ],
    "RF": [
        (1000, 2962.9815, 0.2231),
        (2000, 2958.3388, 0.2198),
        (5000, 2954.6193, 0.2488),
        (10000, 2943.3331, 0.2574),
        (20000, 2932.5256, 0.2979),
        (50000, 2929.2464, 0.5525),
    ],
    "XGB": [
        (1000, 3102.2678, 1.1485),
        (2000, 2997.9150, 1.1201),
        (5000, 2914.0531, 1.3289),
        (10000, 2876.8866, 1.4327),
        (20000, 2863.5196, 1.3865),
        (50000, 2864.8859, 1.6131),
    ],
    "xRFM": [
        (1000, 3064.0788, 0.2101),
        (2000, 3030.5681, 0.3128),
        (5000, 3045.2392, 0.9527),
        (10000, 2954.6234, 12.7189),
        (20000, 2949.3288, 25.2699),
        (50000, 2946.8976, 33.1275),
    ],
}

STYLE = {
    "xRFM": dict(marker="o", color="#d62728"),
    "XGB":  dict(marker="s", color="#1f77b4"),
    "LGBM": dict(marker="^", color="#2ca02c"),
    "RF":   dict(marker="D", color="#ff7f0e"),
}


def to_frames():
    return {
        name: pd.DataFrame(rows, columns=["n", "rmse", "time_s"])
        for name, rows in DATA.items()
    }


def plot_rmse(models, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, df in models.items():
        ax.plot(df["n"], df["rmse"], label=name, linewidth=2, markersize=7, **STYLE[name])
    ax.set_xscale("log")
    ax.set_xlabel("Training set size n  (log scale)")
    ax.set_ylabel("Test RMSE  (lower is better)")
    ax.set_title("Test performance vs n  —  medical_insurance")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_time(models, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, df in models.items():
        ax.plot(df["n"], df["time_s"], label=name, linewidth=2, markersize=7, **STYLE[name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training set size n  (log scale)")
    ax.set_ylabel("Training time  (seconds, log scale)")
    ax.set_title("Training time vs n  —  medical_insurance")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_combined(models, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for name, df in models.items():
        ax.plot(df["n"], df["rmse"], label=name, linewidth=2, markersize=7, **STYLE[name])
    ax.set_xscale("log")
    ax.set_xlabel("n  (log scale)")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Test performance vs n")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    for name, df in models.items():
        ax.plot(df["n"], df["time_s"], label=name, linewidth=2, markersize=7, **STYLE[name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n  (log scale)")
    ax.set_ylabel("Training time  (s, log scale)")
    ax.set_title("Training time vs n")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.suptitle("Scaling on medical_insurance", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    models = to_frames()

    print("Data used:")
    for k, v in models.items():
        print(f"  [{k}]")
        print(v.to_string(index=False))
        print()

    plot_rmse(models, here / "scaling_rmse_vs_n.png")
    plot_time(models, here / "scaling_time_vs_n.png")
    plot_combined(models, here / "scaling_combined.png")
    print("Saved: scaling_rmse_vs_n.png, scaling_time_vs_n.png, scaling_combined.png")


if __name__ == "__main__":
    main()
