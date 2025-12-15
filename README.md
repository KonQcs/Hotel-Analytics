# Hotel Analytics — Cancellation Risk & Short-Term Occupancy (2024–2025)

A reproducible baseline project for:
1) **Per-booking cancellation prediction** with **calibrated probabilities** and **threshold optimization** (Best‑F1 & Cost‑optimal).
2) **Short‑term occupancy forecasting** per hotel (7–14 days) using **HistGradientBoostingRegressor** with **time‑series features** (lags / rolling stats / cyclic seasonality).


---

## Repository Contents

```
DATA EXTRACTION 2025/
├─ Forecast_vs_Actual_Charts/          # optional: per‑hotel PNGs if you generate them
├─ output/                             # legacy v1 outputs (optional)
├─ v2Output/                           # main outputs produced by v2
│   ├─ baseline_report.txt
│   ├─ pr_cancellation.png
│   ├─ roc_cancellation.png
│   ├─ threshold_grid_costs.csv
│   ├─ cancellation_scored_sample.csv
│   ├─ occupancy_predictions_sample.csv
│   ├─ hist_lead_time.png
│   └─ hist_price_per_night.png
├─ Forecast_vs_Actual.py               # helper: Forecast vs Actual per hotel (PNGs/PDF)
├─ Forecast_vs_Actual_by_hotel.pdf     # multi‑page Forecast vs Actual (per hotel)
├─ README_hotel_dataset.txt            # data dictionary for the CSVs
├─ hotel_analytics_baseline_v1.py
├─ hotel_analytics_baseline_v2.py
├─ hotel_bookings_synth.csv
├─ hotel_occupancy_daily.csv
└─ occupancy_predictions_sample.csv    # sample output for plotting / checks
```

**What each item is for**
- **hotel_analytics_baseline_v1.py** — simple baseline (Logistic for cancellations without calibration; RandomForest for occupancy).
- **hotel_analytics_baseline_v2.py** — recommended pipeline (Calibrated Logistic for cancellations; HistGradientBoostingRegressor for occupancy). Generates everything in **`v2Output/`**.
- **hotel_bookings_synth.csv** — synthetic bookings dataset (2024–2025).
- **hotel_occupancy_daily.csv** — daily occupancy per hotel.
- **README_hotel_dataset.txt** — column notes and conventions for both CSVs.
- **Forecast_vs_Actual.py** — creates per‑hotel Forecast‑vs‑Actual PNGs and the multipage **Forecast_vs_Actual_by_hotel.pdf**.
- **v2Output/** — metrics & visualizations produced by **v2** (see details below).
- **Forecast_vs_Actual_Charts/** — optional folder where the per‑hotel PNGs can be saved.

---

## Data

**`hotel_bookings_synth.csv`** — per‑booking rows. Key fields:  
`booking_id, hotel_id, created_at, checkin_date, checkout_date, cancel_date, is_cancelled, channel, cancellation_policy, price_per_night, lead_time_days, num_guests, checkin_dow, holiday_flag, competitor_price_index, weather_score, events_nearby_level`.  
Target: **`is_cancelled` (0/1)**.

**`hotel_occupancy_daily.csv`** — daily occupancy per hotel. Fields:  
`hotel_id, date, occupancy_rate, holiday_flag`.  
Target: **`occupancy_rate ∈ [0,1]`** (regression).

For column notes, see **`README_hotel_dataset.txt`**.

---

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -U pip
pip install pandas numpy scikit-learn matplotlib
```

---

## How to Run

### v1 (simple baseline)
Plain Logistic for cancellations (no calibration); RandomForest for occupancy.
```bash
python hotel_analytics_baseline_v1.py   --bookings hotel_bookings_synth.csv   --occupancy hotel_occupancy_daily.csv   --outdir output
```

### v2 (recommended)
Calibrated Logistic (isotonic, `class_weight='balanced'`) + PR/ROC + **threshold grid**;  
HistGradientBoostingRegressor for occupancy with time‑series features.
```bash
python hotel_analytics_baseline_v2.py   --bookings hotel_bookings_synth.csv   --occupancy hotel_occupancy_daily.csv   --outdir v2Output
```

> If your local copy uses hard‑coded paths, adjust the constants or use the CLI flags above.

---

## Outputs (in `v2Output/`)

- **`baseline_report.txt`** — metrics summary (ROC‑AUC, PR‑AUC, Best‑F1 threshold & F1, Cost‑optimal threshold & total cost, MAE & sMAPE) and brief EDA notes.
- **`pr_cancellation.png` / `roc_cancellation.png`** — PR/ROC curves on the 2025 test set.
- **`threshold_grid_costs.csv`** — table with `threshold, FP, FN, total_cost` for the chosen cost scheme (default example: `cFN=100`, `cFP=10`).
- **`cancellation_scored_sample.csv`** — high‑risk bookings (top‑N) with `p_cancel` and operational context (e.g., channel, policy, date).
- **`occupancy_predictions_sample.csv`** — `hotel_id, date, occupancy_rate, pred_occupancy_rate` for Forecast‑vs‑Actual plots.
- **`hist_lead_time.png`, `hist_price_per_night.png`** — EDA histograms.

**Extra plotting**
```bash
python Forecast_vs_Actual.py
# Produces per‑hotel PNGs (optional) and updates Forecast_vs_Actual_by_hotel.pdf
```

---

## Methodology (concise)

**Cancellation Classification**
- Pipeline: numeric → `StandardScaler`, categorical → `OneHotEncoder`.
- Estimator: `LogisticRegression(class_weight='balanced')` wrapped by `CalibratedClassifierCV(method='isotonic', cv=3)` → reliable **P(cancel)**.
- Thresholds:
  - **Best‑F1** from PR sweep (balance precision/recall).
  - **Cost‑optimal** from a grid using `cFN`/`cFP` (minimize operational loss).

**Occupancy Regression**
- Model: `HistGradientBoostingRegressor`.
- Features: time lags (1/7/14/28), rolling stats (mean/std 7/14/28), cyclic seasonality (`month_sin/cos`, `dow_sin/cos`), `holiday_flag`, and `hotel_id` one‑hot.
- Split: time‑based — **train=2024**, **test=2025**.

---

## Evaluation (from a recent run — check `v2Output/baseline_report.txt` for your exact numbers)

- **Classification:** ROC‑AUC ≈ **0.705**, PR‑AUC ≈ **0.374**; **Best‑F1 threshold ≈ 0.25** (F1 ≈ **0.463**); **Cost‑optimal threshold ≈ 0.09** (example scheme `cFN=100`, `cFP=10`, total cost ≈ **54.9k**).
- **Occupancy:** **MAE ≈ 0.0243**, **sMAPE ≈ 0.2159** on the 2025 test horizon.

---

## Reproducibility

- A single **v2** run generates all artifacts deterministically under `v2Output/`.
- Re‑running with the same CSVs reproduces the outputs 1:1 (PR/ROC, threshold grid, scored samples, predictions, histograms).

---

## Troubleshooting

- **Date parsing:** If warnings mention `dayfirst=False`, parse with `dayfirst=True` or use an explicit format (`%d/%m/%Y`).
- **Missing columns:** v2 expects the fields listed in **Data**; otherwise a `KeyError` will occur.
- **Windows & Greek paths:** Prefer ASCII paths or raw strings (e.g., `r"C:\path\to\data"`).
- **Imbalance:** Evaluate with **PR‑AUC** and keep `class_weight='balanced'`.
- **Costs:** Tune `COST_FN` / `COST_FP` to match your business context (e.g., high‑season vs low‑season).
