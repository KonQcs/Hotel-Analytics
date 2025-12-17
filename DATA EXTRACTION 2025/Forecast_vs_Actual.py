# forecast_vs_actual_from_csv.py
# Δημιουργεί 1 PNG ανά hotel + 1 πολυσέλιδο PDF (χωρίς subplots)

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE = Path(__file__).resolve().parent
CSV = BASE / "occupancy_predictions_sample.csv"

OUT_DIR = BASE / "Forecast_vs_Actual_Charts"
PDF = BASE / "Forecast_vs_Actual_by_hotel.pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV, parse_dates=["date"])

required = {"hotel_id","date","occupancy_rate","pred_occupancy_rate"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Λείπουν στήλες: {missing}")

df = df.sort_values(["hotel_id","date"])
hotels = df["hotel_id"].dropna().unique()

with PdfPages(PDF) as pdf:
    for hid in hotels:
        sub = df[df["hotel_id"] == hid]
        if sub.empty:
            continue
        plt.figure()
        plt.plot(sub["date"], sub["occupancy_rate"], label="Actual")
        plt.plot(sub["date"], sub["pred_occupancy_rate"], label="Forecast")
        plt.title(f"Hotel {hid} — Forecast vs Actual")
        plt.xlabel("Date"); plt.ylabel("Occupancy rate")
        plt.ylim(0, 1) # ποσοστό 0–1
        plt.legend(); plt.tight_layout()

        png = OUT_DIR / f"forecast_vs_actual_hotel_{hid}.png"
        plt.savefig(png) # PNG ανά hotel
        pdf.savefig() # σελίδα στο PDF
        plt.close()

print(f"[OK] PNG charts: {OUT_DIR}")
print(f"[OK] Multi-page PDF: {PDF}")

