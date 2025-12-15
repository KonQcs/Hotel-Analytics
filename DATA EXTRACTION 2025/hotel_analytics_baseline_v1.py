#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix, classification_report,
    RocCurveDisplay
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def ensure_outdir(out_path: Path) -> None:
    out_path.mkdir(parents=True, exist_ok=True)

def coerce_dates(df: pd.DataFrame, cols) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True, infer_datetime_format=True)
    return df

def require_no_nat(df: pd.DataFrame, cols, name: str) -> None:
    problems = {c: int(df[c].isna().sum()) for c in cols if c in df.columns and df[c].isna().any()}
    if problems:
        details = ', '.join([f"{c}: {cnt} NaT" for c, cnt in problems.items()])
        raise ValueError(f"[{name}] Failed to parse some dates -> {details}. Expected dd/mm/yyyy or yyyy-mm-dd.")

def save_hist(series, title, xlabel, outpath: Path, bins=40) -> None:
    plt.figure()
    series.hist(bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--bookings', type=Path, default=Path('hotel_bookings_synth.csv'))
    parser.add_argument('--occupancy', type=Path, default=Path('hotel_occupancy_daily.csv'))
    parser.add_argument('--outdir', type=Path, default=Path('output'))
    ns = parser.parse_args(args=args)

    ensure_outdir(ns.outdir)

    bookings = pd.read_csv(ns.bookings)
    occupancy = pd.read_csv(ns.occupancy)

    bookings = coerce_dates(bookings, ['created_at','checkin_date','checkout_date','cancel_date'])
    occupancy = coerce_dates(occupancy, ['date'])
    require_no_nat(bookings, ['created_at','checkin_date','checkout_date'], 'bookings')
    require_no_nat(occupancy, ['date'], 'occupancy')

    report_lines = []
    report_lines.append(f'Bookings shape: {bookings.shape}')
    report_lines.append(f'Occupancy shape: {occupancy.shape}')
    report_lines.append('Bookings null ratio (top 10):')
    report_lines.append(str(bookings.isnull().mean().sort_values(ascending=False).head(10)))
    report_lines.append('Occupancy null ratio (top 10):')
    report_lines.append(str(occupancy.isnull().mean().sort_values(ascending=False).head(10)))

    cancel_rate = float(bookings['is_cancelled'].mean())
    report_lines.append(f'Overall cancellation rate: {cancel_rate:.2%}')
    if 'channel' in bookings.columns:
        by_channel = bookings.groupby('channel')['is_cancelled'].mean().sort_values(ascending=False)
        report_lines.append('Cancellation rate by channel:')
        report_lines.append(str(by_channel))
    if 'price_per_night' in bookings.columns:
        report_lines.append('Price per night stats:')
        report_lines.append(str(bookings['price_per_night'].describe()))
        save_hist(bookings['price_per_night'], 'Price per night', '€ per night', ns.outdir / 'hist_price_per_night.png')
    if 'lead_time_days' in bookings.columns:
        save_hist(bookings['lead_time_days'], 'Lead time (days)', 'Days', ns.outdir / 'hist_lead_time.png')

    needed_cols = ['lead_time_days','price_per_night','num_guests','events_nearby_level','weather_score','competitor_price_index','holiday_flag',
                   'channel','cancellation_policy','checkin_dow','hotel_id','is_cancelled','checkin_date']
    missing = [c for c in needed_cols if c not in bookings.columns]
    if missing:
        raise KeyError(f'Missing columns in bookings: {missing}')

    train_mask = (bookings['checkin_date'].dt.year == 2024)
    test_mask  = (bookings['checkin_date'].dt.year == 2025)
    train_df = bookings[train_mask].copy()
    test_df  = bookings[test_mask].copy()

    target = 'is_cancelled'
    features_num = ['lead_time_days','price_per_night','num_guests','events_nearby_level','weather_score','competitor_price_index','holiday_flag']
    features_cat = ['channel','cancellation_policy','checkin_dow','hotel_id']

    X_train = train_df[features_num + features_cat]
    y_train = train_df[target].astype(int)
    X_test  = test_df[features_num + features_cat]
    y_test  = test_df[target].astype(int)

    pre = ColumnTransformer([
        ('num', StandardScaler(), features_num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), features_cat)
    ])
    clf = Pipeline([('prep', pre), ('logit', LogisticRegression(max_iter=200))])
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:,1]
    pred = (proba >= 0.5).astype(int)

    roc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)
    clsrep = classification_report(y_test, pred)

    report_lines.append('')
    report_lines.append('== Cancellation Model (Logistic Regression) ==')
    report_lines.append(f'ROC-AUC: {roc:.3f}')
    report_lines.append(f'PR-AUC : {pr_auc:.3f}')
    report_lines.append(f'Confusion matrix (thr=0.5):\n{cm}')
    report_lines.append('Classification report:\n' + clsrep)

    plt.figure()
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title('ROC Curve - Cancellation Model')
    plt.tight_layout()
    roc_path = ns.outdir / 'roc_cancellation.png'
    plt.savefig(roc_path)
    plt.close()

    scored = test_df[['booking_id','hotel_id','checkin_date','channel','cancellation_policy']].copy()
    scored['p_cancel'] = proba
    scored['pred_cancel_0.5'] = pred
    scored_path = ns.outdir / 'cancellation_scored_sample.csv'
    scored.sort_values('p_cancel', ascending=False).head(1000).to_csv(scored_path, index=False)

    # Occupancy forecast
    need_occ = ['hotel_id','date','occupancy_rate','holiday_flag']
    missing_occ = [c for c in need_occ if c not in occupancy.columns]
    if missing_occ:
        raise KeyError(f'Missing columns in occupancy: {missing_occ}')

    df = occupancy.sort_values(['hotel_id','date']).copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dow_num'] = df['date'].dt.weekday

    for lag in [1,7,14,28]:
        df[f'lag_{lag}'] = df.groupby('hotel_id')['occupancy_rate'].shift(lag)
    for w in [7,14,28]:
        df[f'roll_mean_{w}'] = df.groupby('hotel_id')['occupancy_rate'].shift(1).rolling(w).mean()

    df = df.dropna().copy()
    train_df2 = df[df['year'] == 2024].copy()
    test_df2  = df[df['year'] == 2025].copy()

    feat = ['month','dow_num','holiday_flag'] + [f'lag_{l}' for l in [1,7,14,28]] + [f'roll_mean_{w}' for w in [7,14,28]] + ['hotel_id']

    X_train2 = pd.get_dummies(train_df2[feat], columns=['hotel_id'], drop_first=True)
    X_test2  = pd.get_dummies(test_df2[feat],  columns=['hotel_id'], drop_first=True)
    X_train2, X_test2 = X_train2.align(X_test2, join='left', axis=1, fill_value=0)

    y_train2 = train_df2['occupancy_rate'].values
    y_test2  = test_df2['occupancy_rate'].values

    reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    reg.fit(X_train2, y_train2)
    pred2 = reg.predict(X_test2)

    mae = mean_absolute_error(y_test2, pred2)
    smape = float(np.mean(2 * np.abs(pred2 - y_test2) / (np.abs(y_test2) + np.abs(pred2) + 1e-8)))

    report_lines.append('')
    report_lines.append('== Occupancy Forecast (RandomForestRegressor) ==')
    report_lines.append(f'MAE: {mae:.4f}')
    report_lines.append(f'sMAPE: {smape:.4f}')

    out_pred = test_df2[['hotel_id','date','occupancy_rate']].copy()
    out_pred['pred_occupancy_rate'] = pred2
    occ_pred_path = ns.outdir / 'occupancy_predictions_sample.csv'
    out_pred.sort_values(['hotel_id','date']).head(3000).to_csv(occ_pred_path, index=False)

    report_path = ns.outdir / 'baseline_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # Print using plain concatenation to avoid nested f-strings
    print('[OK] Report: ' + str(report_path))
    print('[OK] Cancellation scored sample: ' + str(scored_path))
    print('[OK] Plots: ' + str(ns.outdir / 'hist_price_per_night.png') + ', ' + str(ns.outdir / 'hist_lead_time.png') + ', ' + str(roc_path))
    print('[OK] Occupancy predictions sample: ' + str(occ_pred_path))

if __name__ == '__main__':
    sys.exit(main())
