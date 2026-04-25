import argparse
from pathlib import Path

import pandas as pd

from run_xrfm_utils import load_split, run_xrfm_regression


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
        help="Path to outputs/medical_insurance (contains X_train_processed.csv, etc.)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir.resolve()}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_split(dataset_dir)
    feature_names = list(X_train.columns)

    results = [run_xrfm_regression(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, seed=args.seed)]
    out = pd.DataFrame(results)
    print(out.to_string(index=False))

    out_path = dataset_dir / "model_results_medical_insurance.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()

