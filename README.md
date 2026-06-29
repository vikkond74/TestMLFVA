# ML Exclusion FVA Analyzer

A Streamlit app that judges whether ML exclusions in a demand-planning process
are justified, using Forecast Value Added (FVA) logic: **an exclusion is only
good if the user/consensus forecast beats the ML engine's own (shadow)
forecast against shipped actuals.**

## Input

Long-format CSV or Excel — one row per item per month — at **any level**
(SKU, SPU, CVC = SKU+SPU, country, ...). Required content:

| Content | Example column |
|---|---|
| One or more dimension columns | `SapCode`, `SPU Id` |
| ML exclusion flag (Y/N) | `ML Exclusion` |
| Month / period | `Month` |
| ML forecast (shadow / engine output) | `ML Forecast` |
| User / consensus forecast | `Global Lag 0 PBU Consensus FC` |
| Shipped units (actuals) | `Shipped Units` |

Columns are auto-detected by name and re-mappable in the sidebar. Only
**completed months** are scored (the current and future months are dropped),
and analysis is capped at the most recent 12 completed months. On load, **FCA defaults to the first dimension column (e.g. SapCode) level**; change it in the sidebar. A **Load sample data** button lets new users explore immediately with synthetic data.

## Methodology

- **Variance** = Forecast − Shipped (signed units) — **net** (over/under offset)
- **Bias %** = Variance / Shipped — **net**, positive = over-forecast
- **FCA** = 1 − Σ|error| / ΣShipped, with the absolute errors computed at the
  **FCA calculation level** (e.g. SKU) × month and volume-weighted up — **not
  netted**, so offsetting item/month misses are not forgiven. This is
  mathematically equivalent to a volume-weighted average of item-level FCAs.
- **Verdict** compares FCA(User) vs FCA(ML) against a materiality threshold.

Bias is net by design; accuracy is not — the two answer different questions.

## Features

- **Scorecard** split All / Excluded (Y) / Not excluded (N), with a headline
  verdict on the excluded population.
- **Summed-up analysis** by any column, with Y/N/Both scope, per-month or
  aggregated view, year/period selection, an FCA chart, and CSV export.
- **Item drill-in** at a user-chosen level (Y/N/All scope), with a monthly
  detail chart and per-month metrics.
- **Data-quality checks**, including a shadow-ML detector that flags excluded
  rows where the ML column merely echoes the user forecast.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or deploy on a Streamlit server pointing at `app.py`.

`sample_data.csv` is a fully synthetic example (random brands, SPUs, values)
at CVC (SapCode + SPU) level.
