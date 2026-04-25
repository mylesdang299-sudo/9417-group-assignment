from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parent
    base = root / "COMP9417_GroupProject_Datasets(1)" / "COMP9417_GroupProject_Datasets" / "outputs"

    paths = [
        ("bank_marketing", base / "bank_marketing" / "model_results_bank_marketing.csv"),
        ("customer_churn", base / "customer_churn" / "model_results_customer_churn.csv"),
        ("hr_attrition", base / "hr_attrition" / "model_results_hr_attrition.csv"),
        ("medical_insurance", base / "medical_insurance" / "model_results_medical_insurance.csv"),
        ("house_prices", base / "house_prices" / "model_results_house_prices.csv"),
    ]

    dfs = []
    for dataset, p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing results file: {p}")
        df = pd.read_csv(p)
        df.insert(0, "dataset", dataset)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True, sort=False)

    preferred = [
        "dataset",
        "model",
        "task_type",
        "accuracy",
        "auroc",
        "rmse",
        "mae",
        "r2",
        "train_seconds",
        "infer_seconds",
        "device",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[cols]

    out_path = root / "model_results_all.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

