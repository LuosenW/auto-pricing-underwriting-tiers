"""07_impact.py
Impact analysis: if the insurer moved every policy from the current one-way plan to the GLM plan,
how much would each customer's premium change? Changes are then capped at plus or minus 15% for this
renewal (a common way to limit customer disruption), with the base rate adjusted so the capped plan
still collects the same total premium as the full GLM plan.

Both plans are on a pure premium basis; loading both for expenses and profit by the same factor
would not change any percentage below.

Run from the project folder:  python scripts/07_impact.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BANDS, CHARTS, CLEAN, RATING_VARIABLES, TABLES, load_policies, one_way_plan

NAVY, ORANGE, GRID, INK, MUTED = "#1f4e79", "#e08a3c", "#e6e6e6", "#222222", "#6b6b6b"
CAP = 0.15
CHANGE_BANDS = [-np.inf, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, np.inf]
CHANGE_LABELS = ["Down 25%+", "Down 15-25%", "Down 5-15%", "Within 5%", "Up 5-15%", "Up 15-25%", "Up 25%+"]


def capped_premium(current, new):
    """Find the factor k so that capped premiums, current x clip(k x new/current, 1 - CAP, 1 + CAP),
    add up to the same total as the uncapped GLM plan (bisection search)."""
    ratio = new / current
    target = new.sum()
    lo, hi = 0.5, 2.0
    for _ in range(60):
        k = (lo + hi) / 2
        total = (current * np.clip(k * ratio, 1 - CAP, 1 + CAP)).sum()
        lo, hi = (k, hi) if total < target else (lo, k)
    return current * np.clip(k * ratio, 1 - CAP, 1 + CAP), k


def main():
    df = load_policies()
    plan = pd.read_parquet(CLEAN / "plan_premiums.parquet")[["policy_id", "plan_pure_premium"]]
    df = df.merge(plan, on="policy_id")
    train_mask = (df["sample"] == "train").to_numpy()

    df["current"] = one_way_plan(df, train_mask)          # annual pure premium, current one-way plan
    df["glm"] = df["plan_pure_premium"]                    # annual pure premium, GLM plan
    # premium collected this year = annual rate x exposure; changes are per policy, weighted by exposure
    w = df["exposure"].to_numpy()
    # set the current plan to collect the same total as the GLM plan, so the study shows only redistribution
    df["current"] = df["current"] * (df["glm"] * w).sum() / (df["current"] * w).sum()
    current_written, glm_written = df["current"] * w, df["glm"] * w
    df["change"] = df["glm"] / df["current"] - 1
    capped, k = capped_premium(current_written.to_numpy(), glm_written.to_numpy())
    df["capped_change"] = capped / current_written.to_numpy() - 1

    # Distribution of changes, share of policies
    dist = pd.DataFrame({"change_band": CHANGE_LABELS})
    for col, name in [("change", "share_uncapped"), ("capped_change", "share_capped")]:
        # a change of exactly +/-15% (the cap) belongs in the 5-15% bands, so nudge it a hair toward zero
        nudged = df[col] - np.sign(df[col]) * 1e-9
        b = pd.cut(nudged, CHANGE_BANDS, labels=CHANGE_LABELS)
        dist[name] = dist["change_band"].map(b.value_counts(normalize=True))
    dist = dist.fillna(0)

    summary = pd.DataFrame([
        ("Policies", len(df)),
        ("Total premium, current plan", round(current_written.sum(), 0)),
        ("Total premium, GLM plan", round(glm_written.sum(), 0)),
        ("Policies with a decrease", round((df["change"] < 0).mean(), 4)),
        ("Policies with an increase", round((df["change"] > 0).mean(), 4)),
        ("Policies changing more than 15% either way", round((df["change"].abs() > CAP).mean(), 4)),
        ("Median change", round(df["change"].median(), 4)),
        ("Largest decrease", round(df["change"].min(), 4)),
        ("Largest increase", round(df["change"].max(), 4)),
        ("Capping: policies held at the cap", round((df["capped_change"].abs() >= CAP - 1e-9).mean(), 4)),
        ("Capping: factor applied to GLM rates within the cap", round(k, 4)),
        ("Total premium, capped plan", round(capped.sum(), 0)),
    ], columns=["item", "value"], dtype=object)

    # Average change by rating variable: who moves?
    labels = {col: label for label, col in RATING_VARIABLES.items()}
    seg = []
    for v in ["drv_age_band", "bm_band", "density_band", "veh_age_band", "veh_power_band", "VehGas", "VehBrand", "Region"]:
        g = df.assign(cw=current_written, gw=glm_written, cap=capped).groupby(v).agg(
            policies=("policy_id", "size"), current=("cw", "sum"), glm=("gw", "sum"), capped=("cap", "sum")).reset_index()
        g = g.rename(columns={v: "level"})
        g.insert(0, "variable", labels[v])
        g["avg_change"] = g["glm"] / g["current"] - 1
        g["avg_change_capped"] = g["capped"] / g["current"] - 1
        if v in BANDS:
            g = g.set_index("level").loc[BANDS[v][2]].reset_index()
        seg.append(g)
    seg = pd.concat(seg, ignore_index=True)

    summary.to_csv(TABLES / "impact_summary.csv", index=False)
    dist.round(4).to_csv(TABLES / "impact_distribution.csv", index=False)
    seg.round(4).to_csv(TABLES / "impact_by_segment.csv", index=False)

    print(summary.to_string(index=False))
    print("\n--- Share of policies by premium change ---")
    print(dist.round(3).to_string(index=False))
    print("\n--- Average premium change by segment (largest moves) ---")
    big = seg.reindex(seg["avg_change"].abs().sort_values(ascending=False).index).head(15)
    print(big[["variable", "level", "policies", "avg_change", "avg_change_capped"]].round(3).to_string(index=False))

    # Chart: distribution before and after capping, plus the segments that move most
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(len(dist))
    axes[0].bar(x - 0.2, dist["share_uncapped"] * 100, width=0.38, color=ORANGE, label="Full GLM change")
    axes[0].bar(x + 0.2, dist["share_capped"] * 100, width=0.38, color=NAVY, label="Capped at 15% this renewal")
    axes[0].set_xticks(x, dist["change_band"], fontsize=9, color=MUTED)
    axes[0].set_ylabel("Share of policies (%)", color=MUTED)
    axes[0].legend(frameon=False)
    axes[0].set_title("Premium change moving from the one-way plan to the GLM plan", loc="left", fontsize=12, color=INK)
    top = seg.reindex(seg["avg_change"].abs().sort_values(ascending=False).index).head(10).iloc[::-1]
    names = top["variable"] + " " + top["level"].astype(str)
    axes[1].barh(names, top["avg_change"] * 100, color=[NAVY if c < 0 else ORANGE for c in top["avg_change"]])
    axes[1].axvline(0, color=MUTED, linewidth=1)
    axes[1].set_xlabel("Average premium change, full GLM (%)", color=MUTED)
    axes[1].set_title("Segments that move the most", loc="left", fontsize=12, color=INK)
    for ax in axes:
        ax.grid(axis="y" if ax is axes[0] else "x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
    fig.text(0.01, 0.01, "All 590,983 policies. Changes weighted by policy years; total premium is the same "
             "under the current, GLM and capped plans.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(CHARTS / "06_impact_analysis.png", dpi=150)
    print(f"\nSaved impact tables in {TABLES} and {CHARTS / '06_impact_analysis.png'}")


if __name__ == "__main__":
    main()
