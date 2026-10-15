"""05_rating_plan.py
Turns the GLM coefficients into a rating plan an insurer could use:
  pure premium = base rate x relativity(driver age) x relativity(bonus malus) x ... (one factor per variable)
Each relativity = frequency relativity x severity relativity. The base rate is the pure premium for a
policy at every base level, rebalanced so the plan reproduces actual losses on the training sample.

Run from the project folder:  python scripts/05_rating_plan.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BANDS, CHARTS, CLEAN, LARGE_LOSS_LOADING, RATING_VARIABLES, TABLES, load_policies

NAVY, ORANGE, GRID, INK, MUTED = "#1f4e79", "#e08a3c", "#e6e6e6", "#222222", "#6b6b6b"
LABELS = {col: label for label, col in RATING_VARIABLES.items()}


def build_plan(coefs, train):
    rel = coefs[coefs["variable"] != "Intercept"].pivot_table(
        index=["variable", "level"], columns="model", values="relativity").reset_index()
    rel["severity"] = rel["severity"].fillna(1.0)          # variables not in the severity model
    rel["relativity"] = (rel["frequency"] * rel["severity"]).round(3)   # rounded as a filed rate table would be

    # exposure share of each level, for context in the table
    shares = []
    for v in rel["variable"].unique():
        s = train.groupby(v)["exposure"].sum() / train["exposure"].sum()
        shares.append(pd.DataFrame({"variable": v, "level": s.index.astype(str), "exposure_share": s.values}))
    rel = rel.merge(pd.concat(shares), on=["variable", "level"], how="left")
    rel["label"] = rel["variable"].map(LABELS)
    base_rows = coefs[(coefs["model"] == "frequency") & coefs["p_value"].isna() & (coefs["variable"] != "Intercept")]
    rel["is_base"] = rel.set_index(["variable", "level"]).index.isin(base_rows.set_index(["variable", "level"]).index)
    return rel


def plan_premium(df, plan, base_rate):
    premium = np.full(len(df), base_rate)
    for v in plan["variable"].unique():
        lookup = plan[plan["variable"] == v].set_index("level")["relativity"]
        premium = premium * df[v].astype(str).map(lookup).to_numpy()
    return premium


def order(g, v):
    if v in BANDS:
        return g.set_index("level").loc[BANDS[v][2]].reset_index()
    return g.sort_values("relativity" if v in ("VehBrand", "Region") else "level").reset_index(drop=True)


def chart(plan, one_way):
    variables = ["drv_age_band", "bm_band", "density_band", "veh_age_band",
                 "veh_power_band", "VehGas", "VehBrand", "Region"]
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    for ax, v in zip(axes.flat, variables):
        g = order(plan[plan["variable"] == v], v)
        ow = one_way[one_way["variable"] == LABELS[v]].set_index("level")["pp_relativity"]
        # restate one-way relativities against the same base level as the GLM
        base = g.loc[g["is_base"], "level"].iloc[0]
        ow = ow / ow.loc[base]
        x = np.arange(len(g))
        ax.bar(x - 0.2, ow.reindex(g["level"]).to_numpy(), width=0.38, color=ORANGE, label="One-way")
        ax.bar(x + 0.2, g["relativity"], width=0.38, color=NAVY, label="GLM rating plan")
        ax.axhline(1, color=MUTED, linewidth=1, linestyle="--")
        ax.set_xticks(x, g["level"], rotation=45 if len(g) > 8 else 0, fontsize=8, color=MUTED)
        ax.tick_params(axis="y", labelsize=8, colors=MUTED)
        ax.set_title(f"{LABELS[v]} (base {base})", loc="left", fontsize=12, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes.flat[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("GLM relativities versus one-way relativities, both relative to the same base level",
                 x=0.01, y=0.995, ha="left", fontsize=15, color=INK)
    fig.text(0.01, 0.975, "Where the bars differ, the one-way view was picking up the effect of another variable "
             "(for example young drivers also have high bonus malus levels).", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(CHARTS / "02_glm_vs_one_way.png", dpi=150)


def main():
    df = load_policies()
    preds = pd.read_parquet(CLEAN / "glm_predictions.parquet")
    df = df.merge(preds[["policy_id", "pred_pure_premium"]], on="policy_id")
    train = df[df["sample"] == "train"]
    coefs = pd.read_csv(TABLES / "glm_coefficients.csv", dtype={"level": str})
    one_way = pd.read_csv(TABLES / "one_way.csv", dtype={"level": str})

    plan = build_plan(coefs, train)

    # Base rate before rebalancing: base frequency x base severity x large loss loading
    intercepts = coefs[coefs["variable"] == "Intercept"].set_index("model")["relativity"]
    base_unbalanced = intercepts["frequency"] * intercepts["severity"] * LARGE_LOSS_LOADING
    # Rebalance so the plan reproduces actual (uncapped) training losses
    prem = plan_premium(train, plan, base_unbalanced)
    off_balance = train["claim_amount"].sum() / (prem * train["exposure"]).sum()
    base_rate = round(base_unbalanced * off_balance, 2)

    df["plan_pure_premium"] = plan_premium(df, plan, base_rate)
    df[["policy_id", "sample", "exposure", "claim_count", "claim_amount", "plan_pure_premium"]] \
        .to_parquet(CLEAN / "plan_premiums.parquet", index=False)

    out = plan[["label", "variable", "level", "exposure_share", "frequency", "severity", "relativity"]]
    variables_in_order = [v for v in RATING_VARIABLES.values() if v in set(out["variable"])]
    out = pd.concat([order(out[out["variable"] == v], v) for v in variables_in_order])
    out.round(4).to_csv(TABLES / "rating_plan.csv", index=False)
    pd.DataFrame([("Base rate (pure premium per policy year at base levels)", base_rate),
                  ("Base rate before rebalancing", round(base_unbalanced, 2)),
                  ("Off-balance factor", round(off_balance, 4))],
                 columns=["item", "value"]).to_csv(TABLES / "rating_plan_base_rate.csv", index=False)
    chart(plan, one_way)

    # Checks
    tr = df[df["sample"] == "train"]
    rounding = (tr["plan_pure_premium"] / tr["pred_pure_premium"]).describe()
    print(f"Base rate: {base_rate:,.2f} per policy year "
          f"(model {base_unbalanced:,.2f} x off-balance {off_balance:.4f})")
    print(f"Plan premium / actual losses, train: "
          f"{(tr['plan_pure_premium'] * tr['exposure']).sum() / tr['claim_amount'].sum():.4f}")
    te = df[df["sample"] == "test"]
    print(f"Plan premium / actual losses, test:  "
          f"{(te['plan_pure_premium'] * te['exposure']).sum() / te['claim_amount'].sum():.4f}")
    print(f"Plan / GLM prediction per policy (rounding effect): min {rounding['min']:.4f}, max {rounding['max']:.4f}")
    print(f"Plan pure premium range: {df['plan_pure_premium'].min():,.0f} to {df['plan_pure_premium'].max():,.0f}; "
          f"middle 90% {df['plan_pure_premium'].quantile(0.05):,.0f} to {df['plan_pure_premium'].quantile(0.95):,.0f}\n")
    for v in out["variable"].unique():
        g = out[out["variable"] == v]
        print(f"--- {g['label'].iloc[0]} ---")
        print(g[["level", "exposure_share", "frequency", "severity", "relativity"]].round(3).to_string(index=False), "\n")
    print(f"Saved {TABLES / 'rating_plan.csv'}, {TABLES / 'rating_plan_base_rate.csv'}, "
          f"{CHARTS / '02_glm_vs_one_way.png'} and {CLEAN / 'plan_premiums.parquet'}")


if __name__ == "__main__":
    main()
