"""Titanic data cleaning and preprocessing workflow.

Run from the repository root:
    python titanic_preprocessing.py

The script reads data/Titanic-Dataset.csv and recreates every file under
outputs/. It is deliberately written as a clear, beginner-friendly pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Titanic-Dataset.csv"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

BLUE = "#2563EB"
GREEN = "#059669"
ORANGE = "#EA580C"
RED = "#DC2626"


def configure_output() -> None:
    """Create output folders and set a consistent plot style."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save one tightly cropped PNG and close it."""
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def data_quality_report(data: pd.DataFrame) -> pd.DataFrame:
    """Return a compact column-by-column quality report."""
    rows: list[dict[str, object]] = []
    for column in data.columns:
        non_null = data[column].dropna()
        sample = "" if non_null.empty else str(non_null.iloc[0])
        rows.append(
            {
                "column": column,
                "dtype": str(data[column].dtype),
                "rows": len(data),
                "non_null_count": int(data[column].notna().sum()),
                "missing_count": int(data[column].isna().sum()),
                "missing_percent": round(float(data[column].isna().mean() * 100), 2),
                "unique_values": int(data[column].nunique(dropna=True)),
                "example_value": sample,
            }
        )
    return pd.DataFrame(rows)


def extract_title(names: pd.Series) -> pd.Series:
    """Extract and simplify passenger titles from the Name field."""
    titles = names.str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
    titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    common_titles = {"Mr", "Miss", "Mrs", "Master"}
    return titles.where(titles.isin(common_titles), "Rare")


def iqr_bounds(series: pd.Series, multiplier: float) -> tuple[float, float, float, float, float]:
    """Calculate Q1, Q3, IQR, lower bound, and upper bound."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return q1, q3, iqr, lower, upper


def plot_missing_values(data: pd.DataFrame) -> None:
    """Visualize missing values before cleaning."""
    missing = data.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(missing.index, missing.values, color=[RED, ORANGE, BLUE])
    ax.bar_label(bars, labels=[f"{v} ({v / len(data):.1%})" for v in missing.values], padding=4)
    ax.set_title("Missing values before cleaning")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Missing rows")
    ax.set_ylim(0, max(missing.values) * 1.15)
    save_figure(fig, "01_missing_values_before.png")


def plot_boxplots(data: pd.DataFrame, filename: str, title: str) -> None:
    """Plot Age and Fare boxplots on separate scales."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    sns.boxplot(y=data["Age"], color=BLUE, ax=axes[0])
    axes[0].set_title("Age")
    axes[0].set_ylabel("Years")
    sns.boxplot(y=data["Fare"], color=ORANGE, ax=axes[1])
    axes[1].set_title("Fare")
    axes[1].set_ylabel("Fare")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, filename)


def plot_target_balance(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Compare target class counts before and after outlier filtering."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, data, label in zip(axes, [before, after], ["Before filtering", "After filtering"]):
        counts = data["Survived"].value_counts().reindex([0, 1], fill_value=0)
        bars = ax.bar(["Did not survive", "Survived"], counts.values, color=[RED, GREEN])
        ax.bar_label(bars, padding=3)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=8)
    axes[0].set_ylabel("Passengers")
    fig.suptitle("Target balance check", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "04_target_balance_before_after.png")


def plot_scaling_example(readable: pd.DataFrame, processed: pd.DataFrame) -> None:
    """Show how standardization changes scale but not distribution shape."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    sns.histplot(readable["Age"], bins=25, color=BLUE, ax=axes[0, 0])
    axes[0, 0].set_title("Age before scaling")
    sns.histplot(processed["Age"], bins=25, color=GREEN, ax=axes[0, 1])
    axes[0, 1].set_title("Age after standardization")
    sns.histplot(readable["Fare"], bins=25, color=ORANGE, ax=axes[1, 0])
    axes[1, 0].set_title("Fare before scaling")
    sns.histplot(processed["Fare"], bins=25, color=RED, ax=axes[1, 1])
    axes[1, 1].set_title("Fare after standardization")
    fig.suptitle("Feature scaling comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "05_feature_scaling_comparison.png")


def build_features(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Remove duplicates, engineer useful fields, and impute missing values."""
    working = data.drop_duplicates().copy()
    duplicate_rows_removed = len(data) - len(working)

    working["Title"] = extract_title(working["Name"])
    working["FamilySize"] = working["SibSp"] + working["Parch"] + 1
    working["IsAlone"] = (working["FamilySize"] == 1).astype(int)
    working["TicketGroupSize"] = working.groupby("Ticket")["Ticket"].transform("size")
    working["FarePerPerson"] = working["Fare"] / working["TicketGroupSize"]
    working["CabinKnown"] = working["Cabin"].notna().astype(int)
    working["Deck"] = working["Cabin"].str[0].fillna("Unknown")

    age_missing_before = int(working["Age"].isna().sum())
    group_age_median = working.groupby(["Sex", "Pclass", "Title"])["Age"].transform("median")
    working["Age"] = working["Age"].fillna(group_age_median)
    working["Age"] = working["Age"].fillna(working["Age"].median())

    embarked_missing_before = int(working["Embarked"].isna().sum())
    embarked_mode = str(working["Embarked"].mode(dropna=True).iloc[0])
    working["Embarked"] = working["Embarked"].fillna(embarked_mode)

    details = {
        "duplicate_rows_removed": duplicate_rows_removed,
        "age_values_imputed": age_missing_before,
        "age_imputation": "Median within Sex x Pclass x Title groups; overall median fallback",
        "embarked_values_imputed": embarked_missing_before,
        "embarked_imputation": f"Mode ({embarked_mode})",
        "cabin_handling": "Converted to Deck and CabinKnown; missing decks marked Unknown",
    }
    return working, details


def analyze_and_remove_outliers(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """Document 1.5x IQR and remove only extreme 3.0x IQR Age/Fare rows."""
    rows: list[dict[str, object]] = []
    selected_bounds: dict[str, dict[str, float]] = {}

    for multiplier in (1.5, 3.0):
        for feature in ("Age", "Fare"):
            q1, q3, iqr, lower, upper = iqr_bounds(data[feature], multiplier)
            outlier_mask = (data[feature] < lower) | (data[feature] > upper)
            rows.append(
                {
                    "feature": feature,
                    "iqr_multiplier": multiplier,
                    "q1": round(q1, 4),
                    "q3": round(q3, 4),
                    "iqr": round(iqr, 4),
                    "lower_bound": round(lower, 4),
                    "upper_bound": round(upper, 4),
                    "outlier_rows": int(outlier_mask.sum()),
                    "used_for_removal": multiplier == 3.0,
                }
            )
            if multiplier == 3.0:
                selected_bounds[feature] = {"lower": lower, "upper": upper}

    feature_masks = pd.DataFrame(index=data.index)
    for feature, bounds in selected_bounds.items():
        feature_masks[feature] = (data[feature] < bounds["lower"]) | (
            data[feature] > bounds["upper"]
        )
    combined_mask = feature_masks.any(axis=1)
    reasons = feature_masks.apply(
        lambda row: ", ".join(row.index[row.to_numpy(dtype=bool)]), axis=1
    )

    removed = data.loc[combined_mask].copy()
    removed["OutlierReason"] = reasons.loc[combined_mask]
    filtered = data.loc[~combined_mask].copy()
    return filtered, removed, pd.DataFrame(rows), selected_bounds


def make_readable_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Keep useful, interpretable fields and drop raw text identifiers."""
    columns = [
        "PassengerId",
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Title",
        "Deck",
        "CabinKnown",
        "FamilySize",
        "IsAlone",
        "TicketGroupSize",
        "FarePerPerson",
    ]
    return data[columns].reset_index(drop=True)


def encode_and_standardize(
    readable: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode categories and standardize numeric features."""
    identifiers = readable[["PassengerId", "Survived"]].copy()
    features = readable.drop(columns=["PassengerId", "Survived"]).copy()
    features["Pclass"] = features["Pclass"].astype(str)

    categorical_features = ["Pclass", "Sex", "Embarked", "Title", "Deck"]
    encoded = pd.get_dummies(
        features,
        columns=categorical_features,
        prefix=categorical_features,
        dtype=int,
        drop_first=False,
    )

    numeric_features = [
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "TicketGroupSize",
        "FarePerPerson",
    ]
    scaler = StandardScaler()
    encoded[numeric_features] = scaler.fit_transform(encoded[numeric_features])

    scaler_parameters = pd.DataFrame(
        {
            "feature": numeric_features,
            "mean_used_by_scaler": scaler.mean_,
            "scale_used_by_scaler": scaler.scale_,
        }
    )
    processed = pd.concat(
        [identifiers.reset_index(drop=True), encoded.reset_index(drop=True)], axis=1
    )
    return processed, scaler_parameters


def run_pipeline() -> dict[str, object]:
    """Run the complete workflow and return its summary."""
    configure_output()
    raw = pd.read_csv(DATA_PATH)

    data_quality_report(raw).to_csv(OUTPUT_DIR / "data_quality_report_before.csv", index=False)
    missing_before = (
        raw.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_percent=lambda x: (x["missing_count"] / len(raw) * 100).round(2))
    )
    missing_before.to_csv(OUTPUT_DIR / "missing_values_before.csv")
    plot_missing_values(raw)
    plot_boxplots(raw, "02_outliers_before.png", "Age and Fare before outlier filtering")

    featured, imputation_details = build_features(raw)
    filtered, removed, outlier_summary, selected_bounds = analyze_and_remove_outliers(featured)
    readable = make_readable_dataset(filtered)
    processed, scaler_parameters = encode_and_standardize(readable)

    readable.to_csv(OUTPUT_DIR / "titanic_cleaned_features.csv", index=False)
    processed.to_csv(OUTPUT_DIR / "titanic_preprocessed.csv", index=False)
    removed.to_csv(OUTPUT_DIR / "removed_extreme_outliers.csv", index=False)
    outlier_summary.to_csv(OUTPUT_DIR / "outlier_summary.csv", index=False)
    scaler_parameters.to_csv(OUTPUT_DIR / "scaler_parameters.csv", index=False)
    data_quality_report(readable).to_csv(
        OUTPUT_DIR / "data_quality_report_after.csv", index=False
    )

    missing_after = (
        processed.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_percent=lambda x: (x["missing_count"] / len(processed) * 100).round(2))
    )
    missing_after.to_csv(OUTPUT_DIR / "missing_values_after.csv")

    plot_boxplots(
        readable,
        "03_outliers_after.png",
        "Age and Fare after extreme-outlier filtering",
    )
    plot_target_balance(raw, readable)
    plot_scaling_example(readable, processed)

    summary: dict[str, object] = {
        "input_file": DATA_PATH.name,
        "input_rows": int(len(raw)),
        "input_columns": int(raw.shape[1]),
        "duplicate_rows_found": int(raw.duplicated().sum()),
        "rows_after_duplicate_removal": int(len(featured)),
        "outlier_policy": (
            "Document the standard 1.5x IQR result, but remove only extreme "
            "3.0x IQR outliers from Age and Fare to reduce unnecessary data loss."
        ),
        "selected_outlier_bounds": {
            feature: {key: round(value, 4) for key, value in bounds.items()}
            for feature, bounds in selected_bounds.items()
        },
        "extreme_outlier_rows_removed": int(len(removed)),
        "final_rows": int(len(processed)),
        "final_columns": int(processed.shape[1]),
        "missing_values_before": {
            column: int(value) for column, value in raw.isna().sum().items()
        },
        "missing_values_after_total": int(processed.isna().sum().sum()),
        "survival_rate_before": round(float(raw["Survived"].mean()), 4),
        "survival_rate_after": round(float(readable["Survived"].mean()), 4),
        "imputation_details": imputation_details,
        "encoding": "One-hot encoding for Pclass, Sex, Embarked, Title, and Deck",
        "scaling": "StandardScaler for continuous and count features",
        "output_feature_columns": processed.columns.tolist(),
    }
    with (OUTPUT_DIR / "preprocessing_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    if int(processed.isna().sum().sum()) != 0:
        raise ValueError("The final preprocessed dataset still contains missing values.")
    if processed["PassengerId"].duplicated().any():
        raise ValueError("PassengerId should remain unique after preprocessing.")

    return summary


def main() -> None:
    summary = run_pipeline()
    print("Titanic preprocessing completed successfully.")
    print(f"Input rows: {summary['input_rows']}")
    print(f"Rows removed as extreme outliers: {summary['extreme_outlier_rows_removed']}")
    print(f"Final modeling rows: {summary['final_rows']}")
    print(f"Final features (including ID and target): {summary['final_columns']}")
    print(f"Missing values in final data: {summary['missing_values_after_total']}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
