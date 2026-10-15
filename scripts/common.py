"""common.py
Shared settings and the banding rules used by every analysis step, so the one-way analysis,
the GLM and the rating plan all group the rating variables the same way.
"""
from pathlib import Path

import pandas as pd

CLEAN = Path("data") / "clean"
TABLES = Path("output") / "tables"
CHARTS = Path("output") / "charts"
LARGE_LOSS_LOADING = 1.2182   # total / capped losses, from 02_clean.py

RATING_VARIABLES = {
    # variable label: column holding the band
    "Driver age": "drv_age_band",
    "Bonus malus": "bm_band",
    "Vehicle age": "veh_age_band",
    "Vehicle power": "veh_power_band",
    "Area": "Area",
    "Density": "density_band",
    "Fuel": "VehGas",
    "Vehicle brand": "VehBrand",
    "Region": "Region",
}


BANDS = {
    # band column: (source column, upper edges, labels); each band includes its upper edge
    "drv_age_band": ("DrivAge", [17, 20, 25, 30, 40, 50, 60, 70, 200],
                     ["18-20", "21-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71+"]),
    "bm_band": ("BonusMalus", [0, 50, 60, 80, 99, 125, 1000],
                ["50", "51-60", "61-80", "81-99", "100-125", "126+"]),
    "veh_age_band": ("VehAge", [-1, 0, 2, 5, 10, 15, 200],
                     ["0", "1-2", "3-5", "6-10", "11-15", "16+"]),
    "veh_power_band": ("VehPower", [0, 4, 5, 6, 7, 8, 10, 100],
                       ["4", "5", "6", "7", "8", "9-10", "11+"]),
    "density_band": ("Density", [0, 50, 200, 1000, 5000, 100000],
                     ["1-50", "51-200", "201-1,000", "1,001-5,000", "5,001+"]),
}


def add_bands(df):
    df = df.copy()
    for band_col, (source, edges, labels) in BANDS.items():
        df[band_col] = pd.cut(df[source], bins=edges, labels=labels, right=True).astype(str)
    return df


def load_policies():
    return add_bands(pd.read_parquet(CLEAN / "policies.parquet"))
