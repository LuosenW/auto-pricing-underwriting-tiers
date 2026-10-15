"""09_export_tableau.py
Reshapes the results into tidy CSV files for the Tableau Public dashboard (dashboard/tableau_data/).
Each file has a level_order column so Tableau keeps bands in their natural order (18-20 before 21-25)
instead of sorting them alphabetically.

Run from the project folder after scripts 03 to 08:  python scripts/09_export_tableau.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from common import BANDS, RATING_VARIABLES, TABLES

OUT = Path("dashboard") / "tableau_data"
LABEL_TO_COL = dict(RATING_VARIABLES)


def add_order(df, label_col="variable", level_col="level", sort_col=None):
    """level_order = band order for banded variables; for codes (brand, region), order by sort_col."""
    parts = []
    for label, g in df.groupby(label_col, sort=False):
        col = LABEL_TO_COL.get(label, label)
        if col in BANDS:
            order = {lv: i for i, lv in enumerate(BANDS[col][2])}
            g = g.assign(level_order=g[level_col].map(order))
        elif sort_col is not None and col in ("VehBrand", "Region"):
            g = g.assign(level_order=g[sort_col].rank(method="first").astype(int) - 1)
        else:
            g = g.assign(level_order=g[level_col].rank(method="dense").astype(int) - 1)
        parts.append(g.sort_values("level_order"))
    return pd.concat(parts, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. One row per rating variable level: GLM and one-way relativities (against the GLM base level)
    plan = pd.read_csv(TABLES / "rating_plan.csv", dtype={"level": str})
    ow = pd.read_csv(TABLES / "one_way.csv", dtype={"level": str})
    coefs = pd.read_csv(TABLES / "glm_coefficients.csv", dtype={"level": str})
    bases = coefs[(coefs["model"] == "frequency") & coefs["p_value"].isna() & (coefs["variable"] != "Intercept")]
    base_level = dict(zip(bases["variable"], bases["level"]))
    rows = []
    for _, r in plan.iterrows():
        o = ow[ow["variable"] == r["label"]].set_index("level")["pp_relativity"]
        rows.append({"variable": r["label"], "level": r["level"], "exposure_share": r["exposure_share"],
                     "frequency_relativity": r["frequency"], "severity_relativity": r["severity"],
                     "glm_relativity": r["relativity"],
                     "one_way_relativity": o.get(r["level"], np.nan) / o.get(base_level[r["variable"]], np.nan),
                     "is_base_level": r["level"] == base_level[r["variable"]]})
    seg = pd.DataFrame(rows)

    # 2. Actual versus expected claim counts (test sample) and premium change (impact), same 67 segments
    ae = pd.read_csv(TABLES / "validation_actual_vs_expected.csv", dtype={"level": str})
    imp = pd.read_csv(TABLES / "impact_by_segment.csv", dtype={"level": str})
    seg = seg.merge(ae[["variable", "level", "actual_claims", "expected_claims", "claims_ae", "claims_ae_low",
                        "claims_ae_high", "outside_95pct_band", "losses_ae"]], on=["variable", "level"], how="left")
    seg = seg.merge(imp[["variable", "level", "policies", "avg_change", "avg_change_capped"]],
                    on=["variable", "level"], how="left")
    seg = add_order(seg, sort_col="glm_relativity")
    seg.round(4).to_csv(OUT / "segments.csv", index=False)

    # 3. Underwriting tiers
    tiers = pd.read_csv(TABLES / "underwriting_tiers.csv")
    tiers["tier_order"] = tiers["tier"].str[0].astype(int)
    tiers.round(4).to_csv(OUT / "tiers.csv", index=False)

    # 4. Impact analysis
    dist = pd.read_csv(TABLES / "impact_distribution.csv")
    dist["band_order"] = range(len(dist))
    long = dist.melt(id_vars=["change_band", "band_order"], value_vars=["share_uncapped", "share_capped"],
                     var_name="scenario", value_name="share_of_policies")
    long["scenario"] = long["scenario"].map({"share_uncapped": "Full GLM change", "share_capped": "Capped at 15%"})
    long.round(4).to_csv(OUT / "impact_distribution.csv", index=False)

    # 5. Model comparison and lift chart
    pd.read_csv(TABLES / "validation_model_comparison.csv").to_csv(OUT / "model_comparison.csv", index=False)
    lift = pd.read_csv(TABLES / "validation_lift_chart.csv")
    lift.melt(id_vars=["glm_decile"], value_vars=["predicted", "actual"], var_name="series",
              value_name="loss_cost_per_policy_year").round(2).to_csv(OUT / "lift_chart.csv", index=False)

    for f in sorted(OUT.glob("*.csv")):
        print(f"{f}: {len(pd.read_csv(f))} rows")


if __name__ == "__main__":
    main()
