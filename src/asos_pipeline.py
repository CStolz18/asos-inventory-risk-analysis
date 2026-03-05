"""ASOS inventory risk pipeline.

Run from project root:
    .\\.venv\\Scripts\\python.exe src\\asos_pipeline.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REQUIRED_COLUMNS = [
    "url",
    "name",
    "size",
    "category",
    "price",
    "color",
    "sku",
    "description",
    "images",
]


@dataclass(frozen=True)
class ValidationThresholds:
    max_unknown_brand_pct: float = 10.0
    max_duplicate_url_pct: float = 5.0
    max_non_numeric_price_pct: float = 50.0


def ensure_dirs(project_root: Path) -> dict[str, Path]:
    processed = project_root / "data" / "processed"
    figures = project_root / "reports" / "figures"
    reports = project_root / "reports"
    for p in (processed, figures, reports):
        p.mkdir(parents=True, exist_ok=True)
    return {"processed": processed, "figures": figures, "reports": reports}


def load_data(csv_path: Path) -> pd.DataFrame:
    # Rows with malformed CSV structures are skipped by design.
    return pd.read_csv(csv_path, on_bad_lines="skip")


def validate_required_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def extract_brand_simple(description: pd.Series) -> pd.Series:
    brand_raw = (
        description.astype(str)
        .str.extract(r"(?i)\bby\s+([A-Za-z]+)", expand=False)
        .fillna("Unknown")
    )
    return brand_raw.replace(
        {
            "New": "New Look",
            "River": "River Island",
            "Miss": "Miss Selfridge",
            "TopshopWelcome": "Topshop",
        }
    )


def extract_brand_multiword(description: pd.Series) -> pd.Series:
    brand_raw = (
        description.astype(str)
        .str.extract(r"(?i)\bby\s+([A-Za-z][A-Za-z&'. -]{1,40})", expand=False)
        .fillna("Unknown")
        .str.strip()
    )

    # Conservative cleanup to avoid long noisy tails from merged strings.
    brand_raw = brand_raw.str.split(r"\s{2,}", regex=True).str[0]

    return brand_raw.replace(
        {
            "New": "New Look",
            "River": "River Island",
            "Miss": "Miss Selfridge",
            "TopshopWelcome": "Topshop",
        }
    )


def apply_price_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["price"]).copy()
    out = out[out["price"] > 0].copy()
    return out


def calculate_stockout_metrics(size_str: object) -> tuple[int, float]:
    if not isinstance(size_str, str):
        return 0, 0.0

    sizes = [s.strip() for s in size_str.split(",") if s.strip()]
    total_sizes = len(sizes)

    out_of_stock_count = size_str.count("Out of stock")
    stockout_rate = out_of_stock_count / total_sizes if total_sizes > 0 else 0.0
    return out_of_stock_count, stockout_rate


def add_stockout_and_risk(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    metrics = out["size"].apply(calculate_stockout_metrics)
    out["Stockout_Count"] = [m[0] for m in metrics]
    out["Stockout_Rate"] = [m[1] for m in metrics]
    out["Revenue_Risk_Score"] = out["price"] * out["Stockout_Count"]
    return out


def add_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    name_lower = out["name"].astype(str).str.lower()

    conditions = [
        name_lower.str.contains(r"coat|jacket|blazer|parka|trench|puffer|gilet|shacket|bomber", regex=True),
        name_lower.str.contains(r"dress", regex=True),
        name_lower.str.contains(r"t-shirt|top|shirt|blouse|bodysuit|vest", regex=True),
        name_lower.str.contains(r"jean|trouser|pant|skirt|short|legging", regex=True),
        name_lower.str.contains(r"jumper|cardigan|knit|sweater", regex=True),
    ]
    choices = ["Outerwear", "Dresses", "Tops", "Bottoms", "Knitwear"]

    out["product_type"] = np.select(conditions, choices, default="Other")

    out["price_band"] = pd.cut(
        out["price"],
        bins=[0, 20, 40, 60, 100, np.inf],
        labels=["0-20", "20-40", "40-60", "60-100", "100+"],
        include_lowest=True,
    )

    return out


def weighted_risk_by_price_band(df: pd.DataFrame) -> pd.Series:
    weight_map = {"0-20": 1.0, "20-40": 1.1, "40-60": 1.25, "60-100": 1.4, "100+": 1.6}
    weights = df["price_band"].astype(str).map(weight_map).fillna(1.0)
    return df["Revenue_Risk_Score"] * weights


def build_validation_report(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    brand_series: pd.Series,
    thresholds: ValidationThresholds,
) -> pd.DataFrame:
    non_numeric_price_pct = pd.to_numeric(raw_df["price"], errors="coerce").isna().mean() * 100
    duplicate_url_pct = raw_df.duplicated(subset=["url"]).mean() * 100
    unknown_brand_pct = (brand_series == "Unknown").mean() * 100

    rows = [
        ("rows_parsed", float(len(raw_df)), np.nan, "info"),
        ("rows_after_price_cleaning", float(len(processed_df)), np.nan, "info"),
        ("non_numeric_price_pct", round(non_numeric_price_pct, 2), thresholds.max_non_numeric_price_pct, "ok"),
        ("duplicate_url_pct", round(duplicate_url_pct, 2), thresholds.max_duplicate_url_pct, "ok"),
        ("unknown_brand_pct", round(unknown_brand_pct, 2), thresholds.max_unknown_brand_pct, "ok"),
    ]

    report = pd.DataFrame(rows, columns=["metric", "value", "threshold_max", "status"])
    for metric in ["non_numeric_price_pct", "duplicate_url_pct", "unknown_brand_pct"]:
        idx = report.index[report["metric"] == metric][0]
        report.loc[idx, "status"] = "ok" if report.loc[idx, "value"] <= report.loc[idx, "threshold_max"] else "warning"

    return report


def build_top_action_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    brand_actions = (
        df.groupby("Brand", as_index=False)
        .agg(
            avg_price=("price", "mean"),
            avg_stockout_rate=("Stockout_Rate", "mean"),
            item_count=("name", "count"),
            Revenue_Risk_Score=("Revenue_Risk_Score", "sum"),
        )
        .assign(
            avg_price=lambda x: x["avg_price"].round(2),
            avg_stockout_rate=lambda x: x["avg_stockout_rate"].round(3),
            Revenue_Risk_Score=lambda x: x["Revenue_Risk_Score"].round(2),
        )
        .sort_values("Revenue_Risk_Score", ascending=False)
        .head(10)
    )

    sku_actions = (
        df.groupby(["sku", "Brand", "name"], as_index=False)
        .agg(
            avg_price=("price", "mean"),
            avg_stockout_rate=("Stockout_Rate", "mean"),
            item_count=("name", "count"),
            Revenue_Risk_Score=("Revenue_Risk_Score", "sum"),
        )
        .assign(
            avg_price=lambda x: x["avg_price"].round(2),
            avg_stockout_rate=lambda x: x["avg_stockout_rate"].round(3),
            Revenue_Risk_Score=lambda x: x["Revenue_Risk_Score"].round(2),
        )
        .sort_values("Revenue_Risk_Score", ascending=False)
        .head(10)
    )

    return brand_actions, sku_actions


def build_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["product_type", "price_band"], as_index=False, observed=True)
        .agg(
            item_count=("name", "count"),
            avg_stockout_rate=("Stockout_Rate", "mean"),
            Revenue_Risk_Score=("Revenue_Risk_Score", "sum"),
        )
        .assign(
            avg_stockout_rate=lambda x: x["avg_stockout_rate"].round(3),
            Revenue_Risk_Score=lambda x: x["Revenue_Risk_Score"].round(2),
        )
        .sort_values("Revenue_Risk_Score", ascending=False)
    )


def build_scenario_table(df: pd.DataFrame) -> pd.DataFrame:
    scenarios: list[dict[str, float | int | str]] = []

    formulas = {
        "base": df["Revenue_Risk_Score"],
        "weighted_price_band": weighted_risk_by_price_band(df),
    }

    for threshold in (0.3, 0.4, 0.5):
        mask = df["Stockout_Rate"] > threshold
        for formula_name, risk in formulas.items():
            subset = df.loc[mask].copy()
            subset["scenario_risk"] = risk.loc[mask]

            scenarios.append(
                {
                    "stockout_threshold": threshold,
                    "risk_formula": formula_name,
                    "rows_flagged": int(mask.sum()),
                    "brands_flagged": int(subset["Brand"].nunique()),
                    "total_scenario_risk": round(float(subset["scenario_risk"].sum()), 2),
                    "avg_price_flagged": round(float(subset["price"].mean()) if len(subset) else 0.0, 2),
                }
            )

    return pd.DataFrame(scenarios)


def build_robustness_report(df_price_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_simple = extract_brand_simple(df_price_clean["description"]).rename("Brand")
    method_multi = extract_brand_multiword(df_price_clean["description"]).rename("Brand")

    s_df = df_price_clean.copy()
    s_df["Brand"] = method_simple
    s_df = s_df[~s_df["Brand"].isin(["Unknown", "NA"])].copy()

    m_df = df_price_clean.copy()
    m_df["Brand"] = method_multi
    m_df = m_df[~m_df["Brand"].isin(["Unknown", "NA"])].copy()

    simple_top = s_df["Brand"].value_counts().head(10)
    multi_top = m_df["Brand"].value_counts().head(10)

    overlap = set(simple_top.index).intersection(set(multi_top.index))

    summary = pd.DataFrame(
        [
            {"metric": "rows_simple", "value": len(s_df)},
            {"metric": "rows_multiword", "value": len(m_df)},
            {"metric": "top10_overlap_count", "value": len(overlap)},
            {"metric": "top10_overlap_pct", "value": round(len(overlap) / 10 * 100, 2)},
        ]
    )

    simple_tbl = simple_top.reset_index()
    simple_tbl.columns = ["simple_brand", "simple_count"]
    multi_tbl = multi_top.reset_index()
    multi_tbl.columns = ["multiword_brand", "multiword_count"]
    comparison = pd.concat(
        [simple_tbl.reset_index(drop=True), multi_tbl.reset_index(drop=True)], axis=1
    )

    return summary, comparison


def min_max_scale(series: pd.Series) -> pd.Series:
    min_v = series.min()
    max_v = series.max()
    if math.isclose(float(max_v), float(min_v)):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_v) / (max_v - min_v)


def build_sku_priority_table(df: pd.DataFrame) -> pd.DataFrame:
    sku = (
        df.groupby(["sku", "Brand", "name"], as_index=False)
        .agg(
            avg_price=("price", "mean"),
            avg_stockout_rate=("Stockout_Rate", "mean"),
            total_stockout_count=("Stockout_Count", "sum"),
            Revenue_Risk_Score=("Revenue_Risk_Score", "sum"),
            row_count=("name", "count"),
        )
        .copy()
    )

    sku["risk_norm"] = min_max_scale(sku["Revenue_Risk_Score"]) 
    sku["stockout_norm"] = min_max_scale(sku["avg_stockout_rate"]) 
    sku["price_norm"] = min_max_scale(sku["avg_price"]) 

    # Weighted score with explicit component weights.
    w_risk, w_stockout, w_price = 0.5, 0.3, 0.2
    sku["Priority_Score"] = (
        w_risk * sku["risk_norm"]
        + w_stockout * sku["stockout_norm"]
        + w_price * sku["price_norm"]
    )

    out = (
        sku.assign(
            avg_price=lambda x: x["avg_price"].round(2),
            avg_stockout_rate=lambda x: x["avg_stockout_rate"].round(3),
            Revenue_Risk_Score=lambda x: x["Revenue_Risk_Score"].round(2),
            Priority_Score=lambda x: x["Priority_Score"].round(4),
        )
        .sort_values("Priority_Score", ascending=False)
    )

    return out


def build_brand_strategy(df: pd.DataFrame) -> pd.DataFrame:
    strategy = (
        df.groupby("Brand", as_index=False)
        .agg(
            avg_price=("price", "mean"),
            avg_stockout_rate=("Stockout_Rate", "mean"),
            item_count=("name", "count"),
            Revenue_Risk_Score=("Revenue_Risk_Score", "sum"),
        )
    )

    strategy = strategy[strategy["item_count"] > 10].copy()

    strategy["Action"] = np.select(
        [
            (strategy["avg_price"] > 40) & (strategy["avg_stockout_rate"] > 0.4),
            (strategy["avg_price"] <= 40) & (strategy["avg_stockout_rate"] > 0.4),
            (strategy["avg_price"] > 40) & (strategy["avg_stockout_rate"] <= 0.4),
        ],
        [
            "Replenishment priority",
            "Supplier/forecasting review",
            "Maintain policy",
        ],
        default="Monitor",
    )

    return strategy


def save_chart_pack(brand_strategy: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    plot_df = brand_strategy.copy()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=plot_df,
        x="avg_price",
        y="avg_stockout_rate",
        size="Revenue_Risk_Score",
        hue="Revenue_Risk_Score",
        palette="viridis",
        sizes=(50, 500),
        alpha=0.7,
        ax=ax,
    )

    winners = plot_df[
        (plot_df["avg_price"] > 40) & (plot_df["avg_stockout_rate"] > 0.4)
    ]
    for _, row in winners.iterrows():
        ax.text(row["avg_price"] + 1, row["avg_stockout_rate"] + 0.01, str(row["Brand"]))

    ax.set_title("Brand Strategy Analysis")
    ax.set_xlabel("Average Price")
    ax.set_ylabel("Average Stockout Rate")
    ax.axvline(x=40, color="red", linestyle="--")
    ax.axhline(y=0.4, color="red", linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / "brand_strategy_scatter.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    top10 = plot_df.sort_values("Revenue_Risk_Score", ascending=False).head(10)
    sns.barplot(data=top10, x="Revenue_Risk_Score", y="Brand", color="#2f6c8f", ax=ax)
    ax.set_title("Top 10 Brands by Revenue Risk Score")
    ax.set_xlabel("Revenue Risk Score")
    ax.set_ylabel("Brand")
    fig.tight_layout()
    fig.savefig(out_dir / "top10_brand_risk.png", dpi=150)
    plt.close(fig)

    action_counts = plot_df["Action"].value_counts().reset_index()
    action_counts.columns = ["Action", "count"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=action_counts, x="count", y="Action", color="#5ba87a", ax=ax)
    ax.set_title("Brand Action Distribution")
    ax.set_xlabel("Count")
    ax.set_ylabel("Action")
    fig.tight_layout()
    fig.savefig(out_dir / "action_distribution.png", dpi=150)
    plt.close(fig)


def write_dashboard_summary(
    out_path: Path,
    validation_report: pd.DataFrame,
    scenario_table: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    sku_priority: pd.DataFrame,
) -> None:
    top_priority = sku_priority.head(5)[
        ["sku", "Brand", "name", "Priority_Score", "Revenue_Risk_Score", "avg_stockout_rate"]
    ]

    lines = [
        "# ASOS Dashboard Summary",
        "",
        "## Validation status",
        validation_report.to_string(index=False),
        "",
        "## Scenario analysis",
        scenario_table.to_string(index=False),
        "",
        "## Brand extraction robustness",
        robustness_summary.to_string(index=False),
        "",
        "## Top 5 SKU priorities",
        top_priority.to_string(index=False),
        "",
        "Generated by src/asos_pipeline.py",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dirs = ensure_dirs(project_root)

    csv_path = project_root / "products_asos.csv"
    if not csv_path.exists():
        alt = project_root / "data" / "raw" / "products_asos.csv"
        csv_path = alt if alt.exists() else csv_path
    if not csv_path.exists():
        raise FileNotFoundError("products_asos.csv not found in project root or data/raw")

    raw_df = load_data(csv_path)

    missing_columns = validate_required_columns(raw_df)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    price_clean_df = apply_price_cleaning(raw_df)

    brand_series = extract_brand_simple(price_clean_df["description"])
    df_clean = price_clean_df.copy()
    df_clean["Brand"] = brand_series
    df_clean = df_clean[~df_clean["Brand"].isin(["Unknown", "NA"])].copy()

    df_clean = add_stockout_and_risk(df_clean)
    df_clean = add_segmentation(df_clean)

    thresholds = ValidationThresholds()
    validation_report = build_validation_report(raw_df, df_clean, brand_series, thresholds)

    brand_actions, sku_actions = build_top_action_tables(df_clean)
    segment_summary = build_segment_summary(df_clean)
    scenario_table = build_scenario_table(df_clean)
    robustness_summary, robustness_comparison = build_robustness_report(price_clean_df)
    sku_priority = build_sku_priority_table(df_clean)
    brand_strategy = build_brand_strategy(df_clean)

    save_chart_pack(brand_strategy, dirs["figures"])

    # Export dashboard-style tables.
    validation_report.to_csv(dirs["processed"] / "validation_report.csv", index=False)
    brand_actions.to_csv(dirs["processed"] / "brand_actions_top10.csv", index=False)
    sku_actions.to_csv(dirs["processed"] / "sku_actions_top10.csv", index=False)
    segment_summary.to_csv(dirs["processed"] / "segment_summary.csv", index=False)
    scenario_table.to_csv(dirs["processed"] / "scenario_analysis.csv", index=False)
    robustness_summary.to_csv(dirs["processed"] / "robustness_summary.csv", index=False)
    robustness_comparison.to_csv(dirs["processed"] / "robustness_top10_comparison.csv", index=False)
    sku_priority.to_csv(dirs["processed"] / "sku_priority_scores.csv", index=False)
    brand_strategy.to_csv(dirs["processed"] / "brand_strategy.csv", index=False)

    write_dashboard_summary(
        dirs["reports"] / "dashboard_summary.md",
        validation_report,
        scenario_table,
        robustness_summary,
        sku_priority,
    )

    metrics = {
        "rows_parsed": int(len(raw_df)),
        "rows_after_cleaning": int(len(df_clean)),
        "brands_modeled": int(df_clean["Brand"].nunique()),
        "total_revenue_risk_score": float(df_clean["Revenue_Risk_Score"].sum()),
    }
    (dirs["processed"] / "run_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Pipeline completed successfully.")
    print(f"CSV outputs: {dirs['processed']}")
    print(f"Chart pack: {dirs['figures']}")
    print(f"Summary: {dirs['reports'] / 'dashboard_summary.md'}")


if __name__ == "__main__":
    main()
