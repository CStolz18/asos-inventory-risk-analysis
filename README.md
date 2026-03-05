# ASOS Inventory Risk Analysis

## Business Objective
This project identifies stockout pressure and commercial risk in ASOS product data. The objective is to support clearer operational decisions for Buying, Merchandising, Inventory Planning, Supply Chain, and Commercial teams.

## Stakeholder Questions
1. Which brands and SKUs carry the highest availability risk?
2. Where is risk concentrated by product type and price band?
3. Which actions should be prioritized: replenishment, supplier/forecast review, or monitoring?
4. How robust are results to different brand extraction methods and scenario assumptions?

## Method Overview
The project combines notebook storytelling and a reproducible Python pipeline:

- Notebook: `notebooks/01_exploration.ipynb`
- Pipeline: `src/asos_pipeline.py`

Core workflow:
1. Load CSV and inspect quality
2. Validate schema and data thresholds
3. Clean prices and extract brands
4. Compute stockout metrics (`Stockout_Count`, `Stockout_Rate`)
5. Compute `Revenue_Risk_Score = price * Stockout_Count`
6. Run scenario analysis (thresholds: 0.3, 0.4, 0.5; base vs weighted risk formula)
7. Run robustness comparison (simple vs multiword brand extraction)
8. Build SKU-level priority score and action framework
9. Export dashboard-style tables and chart pack

## Outputs
After running the pipeline, outputs are saved to:

- Tables: `data/processed/`
  - `validation_report.csv`
  - `brand_actions_top10.csv`
  - `sku_actions_top10.csv`
  - `segment_summary.csv`
  - `scenario_analysis.csv`
  - `robustness_summary.csv`
  - `robustness_top10_comparison.csv`
  - `sku_priority_scores.csv`
  - `brand_strategy.csv`
  - `run_metrics.json`
- Figures: `reports/figures/`
  - `brand_strategy_scatter.png`
  - `top10_brand_risk.png`
  - `action_distribution.png`
- Summary:
  - `reports/dashboard_summary.md`

## Assumptions
- `Revenue_Risk_Score` is a prioritization metric, not realized lost sales
- "Out of stock" text in `size` reliably indicates unavailable size options
- First-token brand extraction after "by" is the main production rule
- Scenario thresholds (0.3, 0.4, 0.5) are decision support cutoffs, not causal proofs

## Limitations
- Data is a static snapshot; no time-series inventory trend is available
- External demand drivers (campaigns, weather, competitor pricing) are not included
- Brand parsing remains rule-based and can misclassify edge cases
- CSV parsing skips malformed lines by design (`on_bad_lines='skip'`)

## Run
From project root:

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run notebook (storytelling):
   - `jupyter notebook`
3. Run reproducible pipeline:
   - `.\\.venv\\Scripts\\python.exe src\\asos_pipeline.py`
4. Optional unit tests:
   - `.\\.venv\\Scripts\\python.exe -m unittest tests\\test_asos_pipeline.py -v`

## Project Structure
- `notebooks/` narrative analysis and stakeholder explanation
- `src/` reproducible pipeline and reusable logic
- `tests/` optional unit tests for key functions
- `data/processed/` exported tables
- `reports/figures/` exported charts

## Data Availability
The raw dataset (`products_asos.csv`) is not included in this repository due to file size and sharing constraints.

To run the project, place the source CSV file in one of these locations:
- project root: `products_asos.csv`
- or `data/raw/products_asos.csv`

The notebook and pipeline will use this local file path.