"""02_clean.py
Cleans the raw freMTPL2 files into one policy table and one claim table, caps large claims,
splits policies into train and test samples, and prints reconciliation checks.
Every rule here is explained in notes/decisions.md.

Run from the project folder:  python scripts/02_clean.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data") / "raw"
CLEAN = Path("data") / "clean"
OUT = Path("output")

BAD_ID_BLOCK_MAX = 1_000_000   # policy IDs up to here have unreliable claim counts (see decisions.md)
LARGE_CLAIM_CAP = 100_000      # euros per claim; the excess is spread back as a flat loading
TEST_SHARE = 0.20
SEED = 42


def main():
    CLEAN.mkdir(parents=True, exist_ok=True)
    freq = pd.read_parquet(RAW / "freMTPL2freq.parquet")
    sev = pd.read_parquet(RAW / "freMTPL2sev.parquet")
    checks = []

    def log(check, value):
        checks.append((check, value))

    # Types and text cleanup
    freq["IDpol"] = freq["IDpol"].astype("int64")
    sev["IDpol"] = sev["IDpol"].astype("int64")
    for col in ["Area", "VehBrand", "VehGas", "Region"]:
        freq[col] = freq[col].astype(str).str.strip("'")

    log("Raw policies", len(freq))
    log("Raw claim amount rows", len(sev))
    raw_amount = sev["ClaimAmount"].sum()
    log("Raw claim amount total", round(raw_amount, 2))
    log("Raw exposure", round(freq["Exposure"].sum(), 1))

    # 1. Claim amounts whose policy is not in the policy file
    orphan = ~sev["IDpol"].isin(freq["IDpol"])
    log("Dropped: claim rows with no matching policy", int(orphan.sum()))
    orphan_amount = sev.loc[orphan, "ClaimAmount"].sum()
    log("Dropped: claim amount with no matching policy", round(orphan_amount, 2))
    sev = sev[~orphan]

    # 2. Policy ID block with unreliable claim counts
    in_block = freq["IDpol"] <= BAD_ID_BLOCK_MAX
    sev_in_block = sev["IDpol"] <= BAD_ID_BLOCK_MAX
    log("Dropped: policies in ID block <= 1,000,000", int(in_block.sum()))
    log("Dropped: exposure in ID block", round(freq.loc[in_block, "Exposure"].clip(upper=1).sum(), 1))
    log("Dropped: claim rows in ID block", int(sev_in_block.sum()))
    block_amount = sev.loc[sev_in_block, "ClaimAmount"].sum()
    log("Dropped: claim amount in ID block", round(block_amount, 2))
    freq = freq[~in_block].copy()
    sev = sev[~sev_in_block].copy()

    # 3. Exposure above 1 year
    log("Policies with exposure capped at 1", int((freq["Exposure"] > 1).sum()))
    freq["Exposure"] = freq["Exposure"].clip(upper=1)

    # 4. Claim table with capped amounts
    sev["claim_amount_capped"] = sev["ClaimAmount"].clip(upper=LARGE_CLAIM_CAP)
    sev["is_large"] = sev["ClaimAmount"] > LARGE_CLAIM_CAP
    log("Claims above the 100,000 cap", int(sev["is_large"].sum()))
    excess_load = sev["ClaimAmount"].sum() / sev["claim_amount_capped"].sum()
    log("Large loss loading (total / capped losses)", round(excess_load, 4))

    # 5. Claim counts that disagree with the claim amounts outside the block
    per_policy = sev.groupby("IDpol").agg(
        claims_with_amount=("ClaimAmount", "size"),
        claim_amount=("ClaimAmount", "sum"),
        claim_amount_capped=("claim_amount_capped", "sum"),
    )
    freq = freq.join(per_policy, on="IDpol")
    for col in ["claims_with_amount", "claim_amount", "claim_amount_capped"]:
        freq[col] = freq[col].fillna(0)
    mismatch = freq["ClaimNb"] != freq["claims_with_amount"]
    log("Policies whose claim count was reset to claims with amounts", int(mismatch.sum()))
    freq["claim_count"] = freq["claims_with_amount"].astype("int64")

    # 6. Train and test split, by policy
    rng = np.random.default_rng(SEED)
    freq["sample"] = np.where(rng.random(len(freq)) < TEST_SHARE, "test", "train")

    policies = freq[[
        "IDpol", "sample", "Exposure", "claim_count", "claim_amount", "claim_amount_capped",
        "Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region",
    ]].rename(columns={"IDpol": "policy_id", "Exposure": "exposure"})
    claims = sev.rename(columns={"IDpol": "policy_id", "ClaimAmount": "claim_amount"})

    # Reconciliation: every raw claim dollar is either kept or in a reported drop
    kept_amount = claims["claim_amount"].sum()
    log("Final policies", len(policies))
    log("Final exposure", round(policies["exposure"].sum(), 1))
    log("Final claims", int(policies["claim_count"].sum()))
    log("Final claim amount", round(kept_amount, 2))
    log("Claim amount check: raw minus kept minus dropped", round(raw_amount - kept_amount - orphan_amount - block_amount, 2))
    log("Policies with claims but no amount (should be 0)",
        int(((policies["claim_count"] > 0) & (policies["claim_amount"] <= 0)).sum()))
    for s in ["train", "test"]:
        p = policies[policies["sample"] == s]
        log(f"{s.title()} policies", len(p))
        log(f"{s.title()} claim frequency (claims per policy year)", round(p["claim_count"].sum() / p["exposure"].sum(), 4))

    policies.to_parquet(CLEAN / "policies.parquet", index=False)
    claims.to_parquet(CLEAN / "claims.parquet", index=False)

    checks = pd.DataFrame(checks, columns=["check", "value"], dtype=object)
    print(checks.to_string(index=False))
    checks.to_csv(OUT / "validation_clean.csv", index=False)
    print(f"\nSaved {CLEAN / 'policies.parquet'}, {CLEAN / 'claims.parquet'} and {OUT / 'validation_clean.csv'}")


if __name__ == "__main__":
    main()
