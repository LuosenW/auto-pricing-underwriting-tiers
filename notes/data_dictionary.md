# Data dictionary

Source: freMTPL2freq and freMTPL2sev, French motor third party liability insurance (OpenML 41214 and 41215; also in the R package CASdatasets). Amounts are in euros. Cleaned tables are built by `scripts/02_clean.py`.

## policies (data/clean/policies.parquet), one row per policy

| Column | Meaning | Values in cleaned data |
| --- | --- | --- |
| policy_id | Policy ID (IDpol in the raw file) | IDs above 1,000,000 |
| sample | Model sample | train (80%), test (20%) |
| exposure | Policy years the policy was in force | 0.003 to 1 (capped at 1) |
| claim_count | Claims with a claim amount | 0 to 16 |
| claim_amount | Total claim amounts, uncapped | euros |
| claim_amount_capped | Total claim amounts with each claim capped at 100,000 | euros |
| Area | Area code from the population density of where the driver lives, A (rural) to F (urban) | A to F |
| VehPower | Vehicle power, ordered category | 4 to 15 |
| VehAge | Vehicle age in years | 0 to 100 |
| DrivAge | Driver age in years | 18 to 100 |
| BonusMalus | French bonus malus level; below 100 is a discount for a good claim history, above 100 a surcharge | 50 to 230 |
| VehBrand | Vehicle brand, coded | 11 brands (B1 to B6, B10 to B14) |
| VehGas | Fuel type | Diesel, Regular |
| Density | Inhabitants per square km in the driver's city | 1 to 27,000 |
| Region | Region in France where the policy is written, coded | 22 regions |

## claims (data/clean/claims.parquet), one row per claim

| Column | Meaning |
| --- | --- |
| policy_id | Policy the claim belongs to |
| claim_amount | Claim amount, uncapped |
| claim_amount_capped | Claim amount capped at 100,000 |
| is_large | Claim above 100,000 |

## Key measures

| Measure | Formula |
| --- | --- |
| Claim frequency | claim_count / exposure |
| Claim severity | claim_amount_capped / claim_count |
| Pure premium (loss cost) | claim_amount_capped / exposure = frequency x severity |
| Large loss loading | total losses / capped losses = 1.2182 |
