"""08_tiers.py
Underwriting tiers: groups policies by the GLM's predicted cost relative to the book average,
then shows what each tier's loss ratio would be if everyone paid one flat rate versus the GLM plan,
and profiles who is in each tier. Loss ratios are measured on the test sample (policies the models
never saw), with claims capped at 100,000 x the large loss loading.

Premium = pure premium / 0.70 (an assumed 70% target loss ratio, leaving 30% for expenses and profit).

Run from the project folder:  python scripts/08_tiers.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CHARTS, CLEAN, LARGE_LOSS_LOADING, TABLES, load_policies

NAVY, ORANGE, GRID, INK, MUTED = "#1f4e79", "#e08a3c", "#e6e6e6", "#222222", "#6b6b6b"
TARGET_LOSS_RATIO = 0.70
TIER_EDGES = [0, 0.75, 1.25, 2.0, 4.0, np.inf]   # predicted cost / book average
TIERS = ["1 Preferred", "2 Standard", "3 Surcharge", "4 Refer", "5 Decline"]


def main():
    df = load_policies()
    plan = pd.read_parquet(CLEAN / "plan_premiums.parquet")[["policy_id", "plan_pure_premium"]]
    df = df.merge(plan, on="policy_id")
    w = df["exposure"]
    book_avg = (df["plan_pure_premium"] * w).sum() / w.sum()
    df["relative_cost"] = df["plan_pure_premium"] / book_avg
    df["tier"] = pd.cut(df["relative_cost"], TIER_EDGES, labels=TIERS, right=False).astype(str)

    test = df[df["sample"] == "test"].copy()
    test["actual"] = test["claim_amount_capped"] * LARGE_LOSS_LOADING
    test["flat_premium"] = book_avg / TARGET_LOSS_RATIO * test["exposure"]
    test["glm_premium"] = test["plan_pure_premium"] / TARGET_LOSS_RATIO * test["exposure"]

    all_tiers = df.groupby("tier").agg(policies=("policy_id", "size"), exposure=("exposure", "sum"),
                                       min_relative_cost=("relative_cost", "min"),
                                       max_relative_cost=("relative_cost", "max"))
    all_tiers["share_of_policies"] = all_tiers["policies"] / all_tiers["policies"].sum()
    t = test.groupby("tier").agg(test_policies=("policy_id", "size"), test_claims=("claim_count", "sum"),
                                 actual=("actual", "sum"), flat_premium=("flat_premium", "sum"),
                                 glm_premium=("glm_premium", "sum"), test_exposure=("exposure", "sum"))
    t["claim_frequency"] = t["test_claims"] / t["test_exposure"]
    t["loss_ratio_flat_rate"] = t["actual"] / t["flat_premium"]
    t["loss_ratio_glm_plan"] = t["actual"] / t["glm_premium"]
    t["share_of_losses"] = t["actual"] / t["actual"].sum()
    tiers = all_tiers.join(t).reset_index()

    # Who is in each tier: profile on all policies
    profile = df.assign(young=df["DrivAge"] <= 25, bm_surcharge=df["BonusMalus"] >= 100,
                        bm_best=df["BonusMalus"] == 50, dense=df["Density"] > 5000,
                        powerful=df["VehPower"] >= 11).groupby("tier").agg(
        avg_driver_age=("DrivAge", "mean"), share_age_25_or_under=("young", "mean"),
        avg_bonus_malus=("BonusMalus", "mean"), share_bonus_malus_50=("bm_best", "mean"),
        share_bonus_malus_100_plus=("bm_surcharge", "mean"), share_density_over_5000=("dense", "mean"),
        share_vehicle_power_11_plus=("powerful", "mean")).reset_index()

    tiers.round(4).to_csv(TABLES / "underwriting_tiers.csv", index=False)
    profile.round(4).to_csv(TABLES / "underwriting_tier_profile.csv", index=False)

    overall_flat = test["actual"].sum() / test["flat_premium"].sum()
    overall_glm = test["actual"].sum() / test["glm_premium"].sum()
    print(f"Book average pure premium {book_avg:,.2f}; flat premium at a 70% target loss ratio "
          f"{book_avg / TARGET_LOSS_RATIO:,.2f} per policy year")
    print(f"Test sample loss ratio overall: flat rate {overall_flat:.3f}, GLM plan {overall_glm:.3f}\n")
    cols = ["tier", "share_of_policies", "min_relative_cost", "max_relative_cost", "test_claims",
            "claim_frequency", "share_of_losses", "loss_ratio_flat_rate", "loss_ratio_glm_plan"]
    print(tiers[cols].round(3).to_string(index=False))
    print("\n--- Who is in each tier ---")
    print(profile.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(tiers))
    axes[0].bar(x - 0.2, tiers["share_of_policies"] * 100, width=0.38, color=NAVY, label="Share of policies")
    axes[0].bar(x + 0.2, tiers["share_of_losses"] * 100, width=0.38, color=ORANGE, label="Share of losses (test)")
    axes[0].set_ylabel("%", color=MUTED)
    axes[0].set_title("Policies and losses by underwriting tier", loc="left", fontsize=12, color=INK)
    axes[0].legend(frameon=False)
    axes[1].bar(x - 0.2, tiers["loss_ratio_flat_rate"] * 100, width=0.38, color=ORANGE, label="One flat rate for everyone")
    axes[1].bar(x + 0.2, tiers["loss_ratio_glm_plan"] * 100, width=0.38, color=NAVY, label="GLM rating plan")
    axes[1].axhline(TARGET_LOSS_RATIO * 100, color=MUTED, linestyle="--", linewidth=1)
    axes[1].text(-0.45, TARGET_LOSS_RATIO * 100 + 6, "70% target", ha="left", color=MUTED, fontsize=9)
    axes[1].set_ylabel("Loss ratio (%)", color=MUTED)
    axes[1].set_title("Loss ratio by tier: flat rate versus GLM plan (test sample)", loc="left", fontsize=12, color=INK)
    axes[1].legend(frameon=False, loc="upper left")
    for ax in axes:
        ax.set_xticks(x, tiers["tier"], fontsize=9, color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
    fig.text(0.01, 0.01, "Tiers by GLM predicted cost relative to the book average: Preferred under 0.75, "
             "Standard 0.75 to 1.25, Surcharge 1.25 to 2, Refer 2 to 4, Decline 4 and over.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(CHARTS / "07_underwriting_tiers.png", dpi=150)
    print(f"\nSaved {TABLES / 'underwriting_tiers.csv'}, {TABLES / 'underwriting_tier_profile.csv'} "
          f"and {CHARTS / '07_underwriting_tiers.png'}")


if __name__ == "__main__":
    main()
