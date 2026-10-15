"""01_get_data.py
Downloads the French motor third party liability data (freMTPL2) from OpenML,
saves untouched copies to data/raw/, and prints load checks.

freMTPL2freq (OpenML 41214): one row per policy with exposure, claim count and rating variables.
freMTPL2sev  (OpenML 41215): one row per claim with the claim amount.

Run from the project folder:  python scripts/01_get_data.py
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

RAW = Path("data") / "raw"
OUT = Path("output")


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    cache = RAW / "openml_cache"
    print("Downloading freMTPL2freq (about 678,000 policies)...")
    freq = fetch_openml(data_id=41214, as_frame=True, data_home=cache).frame
    print("Downloading freMTPL2sev (claim amounts)...")
    sev = fetch_openml(data_id=41215, as_frame=True, data_home=cache).frame
    freq.to_parquet(RAW / "freMTPL2freq.parquet", index=False)
    sev.to_parquet(RAW / "freMTPL2sev.parquet", index=False)
    return freq, sev


def load_checks(freq, sev):
    """Checks on the raw files. Nothing is changed here; cleaning happens in 02_clean.py."""
    freq_ids = freq["IDpol"].astype("int64")
    sev_ids = sev["IDpol"].astype("int64")

    print("\n--- Columns and types: freq ---")
    print(freq.dtypes.to_string())
    print("\n--- Columns and types: sev ---")
    print(sev.dtypes.to_string())
    print("\n--- First rows: freq ---")
    print(freq.head().to_string())

    claims_per_policy = sev_ids.value_counts()
    freq_claims = pd.Series(freq["ClaimNb"].astype("int64").values, index=freq_ids.values)
    has_amount = freq_claims.index.isin(claims_per_policy.index)

    checks = pd.DataFrame(
        [
            ("Policies (freq rows)", len(freq)),
            ("Unique policy IDs in freq", freq_ids.nunique()),
            ("Claim rows (sev rows)", len(sev)),
            ("Policies with at least one claim amount", sev_ids.nunique()),
            ("Total claim count in freq (ClaimNb)", int(freq_claims.sum())),
            ("Total claim amount in sev", round(float(sev["ClaimAmount"].sum()), 2)),
            ("Total exposure (policy years)", round(float(freq["Exposure"].sum()), 1)),
            ("Policies with exposure above 1 year", int((freq["Exposure"] > 1).sum())),
            ("Policies with ClaimNb > 0 but no claim amount", int(((freq_claims > 0) & ~has_amount).sum())),
            ("Claim rows whose policy is not in freq", int((~sev_ids.isin(freq_ids)).sum())),
            ("Claim amounts of 0 or less", int((sev["ClaimAmount"] <= 0).sum())),
            ("Missing values in freq", int(freq.isna().sum().sum())),
            ("Missing values in sev", int(sev.isna().sum().sum())),
        ],
        columns=["check", "value"],
        dtype=object,
    )
    print("\n--- Load checks ---")
    print(checks.to_string(index=False))

    print("\n--- Policies by claim count ---")
    print(freq_claims.value_counts().sort_index().to_string())

    print("\n--- Claim amount percentiles ---")
    print(sev["ClaimAmount"].quantile([0.5, 0.9, 0.99, 0.999, 1.0]).round(0).to_string())

    OUT.mkdir(exist_ok=True)
    checks.to_csv(OUT / "validation_load.csv", index=False)
    print(f"\nSaved {OUT / 'validation_load.csv'}")


if __name__ == "__main__":
    freq, sev = download()
    load_checks(freq, sev)
