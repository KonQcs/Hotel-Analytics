#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hotel_analytics_baseline_v2.py

CHANGES vs v1
-------------
- Features (calibration, PR curve, thresholds, extra features).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report, RocCurveDisplay, precision_recall_curve, f1_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Config
SCRIPT_DIR = Path(__file__).resolve().parent
HARD_BASE_DIR = None  # set to a string/Path to override
BASE_DIR = Path(HARD_BASE_DIR) if HARD_BASE_DIR else SCRIPT_DIR
BOOKINGS_PATH  = BASE_DIR / "hotel_bookings_synth.csv"
OCCUPANCY_PATH = BASE_DIR / "hotel_occupancy_daily.csv"
OUTDIR         = BASE_DIR / "v2Output"

COST_FN = 100.0
COST_FP = 10.0

def ensure_outdir(out_path) -> None:
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

def coerce_dates(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce', dayfirst=True, infer_datetime_format=True)
    return df

def require_no_nat(df: pd.DataFrame, cols, name: str):
    problems = {c: int(df[c].isna().sum()) for c in cols if c in df.columns and df[c].isna().any()}
    if problems:
        details = ', '.join([f"{c}: {cnt} NaT" for c, cnt in problems.items()])
        raise ValueError(f"[{name}] Failed to parse some dates -> {details}. Expected dd/mm/yyyy or yyyy-mm-dd.")

def save_hist(series, title, xlabel, outpath, bins=40):
    outpath = Path(outpath)
    plt.figure()
    series.hist(bins=bins)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel('Count'); plt.tight_layout()
    plt.savefig(outpath); plt.close()

def plot_pr_curve(y_true, proba, outpath):
    outpath = Path(outpath)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    plt.figure(); plt.plot(rec, prec); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve - Cancellation Model'); plt.tight_layout()
    plt.savefig(outpath); plt.close()

def main():
    ensure_outdir(OUTDIR)

    bookings = pd.read_csv(BOOKINGS_PATH)
    occupancy = pd.read_csv(OCCUPANCY_PATH)

    bookings = coerce_dates(bookings, ['created_at','checkin_date','checkout_date','cancel_date'])
    occupancy = coerce_dates(occupancy, ['date'])
    require_no_nat(bookings, ['created_at','checkin_date','checkout_date'], 'bookings')
    require_no_nat(occupancy, ['date'], 'occupancy')

    report_lines = []
    report_lines.append('Bookings shape: ' + str(bookings.shape))
    report_lines.append('Occupancy shape: ' + str(occupancy.shape))
    report_lines.append('Bookings null ratio (top 10):'); report_lines.append(str(bookings.isnull().mean().sort_values(ascending=False).head(10)))
    report_lines.append('Occupancy null ratio (top 10):'); report_lines.append(str(occupancy.isnull().mean().sort_values(ascending=False).head(10)))

    cancel_rate = float(bookings['is_cancelled'].mean())
    report_lines.append('Overall cancellation rate: ' + f'{cancel_rate:.2%}')
    if 'channel' in bookings.columns:
        by_channel = bookings.groupby('channel')['is_cancelled'].mean().sort_values(ascending=False)
        report_lines.append('Cancellation rate by channel:'); report_lines.append(str(by_channel))
    if 'price_per_night' in bookings.columns:
        save_hist(bookings['price_per_night'], 'Price per night', '€ per night', OUTDIR / 'hist_price_per_night.png')
    if 'lead_time_days' in bookings.columns:
        save_hist(bookings['lead_time_days'], 'Lead time (days)', 'Days', OUTDIR / 'hist_lead_time.png')

    needed_cols = ['lead_time_days','price_per_night','num_guests','events_nearby_level','weather_score','competitor_price_index','holiday_flag','channel','cancellation_policy','checkin_dow','hotel_id','is_cancelled','checkin_date']
    missing = [c for c in needed_cols if c not in bookings.columns]
    if missing: raise KeyError('Missing columns in bookings: ' + str(missing))

    train_mask = (bookings['checkin_date'].dt.year == 2024)
    test_mask  = (bookings['checkin_date'].dt.year == 2025)
    train_df = bookings[train_mask].copy(); test_df = bookings[test_mask].copy()

    target = 'is_cancelled'
    features_num = ['lead_time_days','price_per_night','num_guests','events_nearby_level','weather_score','competitor_price_index','holiday_flag']
    features_cat = ['channel','cancellation_policy','checkin_dow','hotel_id']

    X_train = train_df[features_num + features_cat]; y_train = train_df[target].astype(int)
    X_test  = test_df[features_num + features_cat];  y_test = test_df[target].astype(int)

    pre = ColumnTransformer([('num', StandardScaler(), features_num), ('cat', OneHotEncoder(handle_unknown='ignore'), features_cat)])
    base_clf = LogisticRegression(max_iter=500, class_weight='balanced')
    clf = Pipeline([('prep', pre), ('cal', CalibratedClassifierCV(base_clf, method='isotonic', cv=3))])
    clf.fit(X_train, y_train); proba = clf.predict_proba(X_test)[:,1]

    pred_05 = (proba >= 0.5).astype(int)
    roc = roc_auc_score(y_test, proba); pr_auc = average_precision_score(y_test, proba)
    cm_05 = confusion_matrix(y_test, pred_05); rep_05 = classification_report(y_test, pred_05)

    prec, rec, thr = precision_recall_curve(y_test, proba)
    f1s = (2*prec*rec) / (prec + rec + 1e-9)
    best_idx = int(np.nanargmax(f1s[:-1])) if len(f1s) > 1 else 0
    best_thr = float(thr[best_idx]) if len(thr) > 0 else 0.5
    pred_best = (proba >= best_thr).astype(int); f1_best = f1_score(y_test, pred_best)

    costs = []; grid = np.linspace(0,1,101)
    for t in grid:
        p = (proba >= t).astype(int)
        fp = int(((p==1)&(y_test==0)).sum()); fn = int(((p==0)&(y_test==1)).sum())
        total_cost = fp*COST_FP + fn*COST_FN; costs.append((t, fp, fn, total_cost))
    best_cost_thr, best_fp, best_fn, best_cost = min(costs, key=lambda x: x[3])

    plot_pr_curve(y_test, proba, OUTDIR / 'pr_cancellation.png')

    grid_df = pd.DataFrame(costs, columns=['threshold','FP','FN','total_cost'])
    grid_df.to_csv(OUTDIR / 'threshold_grid_costs.csv', index=False)

    report_lines.append(''); report_lines.append('== Cancellation Model (LogReg + calibration) ==')
    report_lines.append('ROC-AUC: ' + f'{roc:.3f}'); report_lines.append('PR-AUC : ' + f'{pr_auc:.3f}')
    report_lines.append('Default thr=0.5 confusion matrix:\n' + str(cm_05))
    report_lines.append('Classification report (thr=0.5):\n' + rep_05)
    report_lines.append('Best-F1 threshold: ' + f'{best_thr:.2f}' + '  F1: ' + f'{f1_best:.3f}')
    report_lines.append('Best-Cost threshold: ' + f'{best_cost_thr:.2f}' + '  Total cost: ' + f'{best_cost:.2f}' + '  (cFN=' + str(COST_FN) + ', cFP=' + str(COST_FP) + ')')

    plt.figure(); RocCurveDisplay.from_predictions(y_test, proba); plt.title('ROC Curve - Cancellation Model'); plt.tight_layout(); plt.savefig(OUTDIR / 'roc_cancellation.png'); plt.close()

    scored = test_df[['booking_id','hotel_id','checkin_date','channel','cancellation_policy']].copy()
    scored['p_cancel'] = proba; scored['pred_thr_0.5'] = pred_05; scored['pred_thr_bestF1'] = pred_best
    scored.sort_values('p_cancel', ascending=False).head(1000).to_csv(OUTDIR / 'cancellation_scored_sample.csv', index=False)

    need_occ = ['hotel_id','date','occupancy_rate','holiday_flag']
    missing_occ = [c for c in need_occ if c not in occupancy.columns]
    if missing_occ: raise KeyError('Missing columns in occupancy: ' + str(missing_occ))

    df = occupancy.sort_values(['hotel_id','date']).copy()
    df['year'] = df['date'].dt.year; df['month'] = df['date'].dt.month; df['dow_num'] = df['date'].dt.weekday
    for lag in [1,7,14,28]: df[f'lag_{lag}'] = df.groupby('hotel_id')['occupancy_rate'].shift(lag)
    for w in [7,14,28]: df[f'roll_mean_{w}'] = df.groupby('hotel_id')['occupancy_rate'].shift(1).rolling(w).mean(); df[f'roll_std_{w}'] = df.groupby('hotel_id')['occupancy_rate'].shift(1).rolling(w).std()

    df['month_sin'] = np.sin(2*np.pi*df['month']/12.0); df['month_cos'] = np.cos(2*np.pi*df['month']/12.0)
    df['dow_sin'] = np.sin(2*np.pi*df['dow_num']/7.0);   df['dow_cos'] = np.cos(2*np.pi*df['dow_num']/7.0)

    df = df.dropna().copy(); train_df2 = df[df['year']==2024].copy(); test_df2 = df[df['year']==2025].copy()
    feat = ['month','dow_num','holiday_flag','month_sin','month_cos','dow_sin','dow_cos','lag_1','lag_7','lag_14','lag_28','roll_mean_7','roll_mean_14','roll_mean_28','roll_std_7','roll_std_14','roll_std_28','hotel_id']

    X_train2 = pd.get_dummies(train_df2[feat], columns=['hotel_id'], drop_first=True)
    X_test2  = pd.get_dummies(test_df2[feat],  columns=['hotel_id'], drop_first=True)
    X_train2, X_test2 = X_train2.align(X_test2, join='left', axis=1, fill_value=0)
    y_train2 = train_df2['occupancy_rate'].values; y_test2  = test_df2['occupancy_rate'].values

    reg = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.06, max_iter=600); reg.fit(X_train2, y_train2); pred2 = reg.predict(X_test2)
    mae = mean_absolute_error(y_test2, pred2); smape = float(np.mean(2 * np.abs(pred2 - y_test2) / (np.abs(y_test2) + np.abs(pred2) + 1e-8)))

    report_lines.append(''); report_lines.append('== Occupancy Forecast (HGBRegressor) =='); report_lines.append('MAE: ' + f'{mae:.4f}'); report_lines.append('sMAPE: ' + f'{smape:.4f}')

    out_pred = test_df2[['hotel_id','date','occupancy_rate']].copy(); out_pred['pred_occupancy_rate'] = pred2
    out_pred.sort_values(['hotel_id','date']).head(3000).to_csv(OUTDIR / 'occupancy_predictions_sample.csv', index=False)

    report_path = OUTDIR / 'baseline_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f: f.write('\n'.join(report_lines))

    print('[OK] Report: ' + str(report_path))
    print('[OK] Outputs folder: ' + str(OUTDIR))

if __name__ == '__main__':
    sys.exit(main())
