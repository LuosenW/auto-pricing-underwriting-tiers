"""06_validate.py
Tests the GLM rating plan on the test sample (policies the models never saw) against three alternatives:
  one-way plan:  base rate x one-way relativities (what you get without a GLM)
  Tweedie GLM:   one model fit directly on pure premium, same variables
  boosted trees: scikit-learn gradient boosting on the raw variables (a machine learning benchmark)
Measures: Gini coefficient, lift chart, double lift chart against the one-way plan, and
actual versus expected by rating variable.

"Actual" losses use claims capped at 100,000 times the large loss loading, so a handful of huge claims
does not decide the comparison (see decisions.md).

Run from the project folder:  python scripts/06_validate.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import TweedieRegressor

from common import (BANDS, CHARTS, CLEAN, LARGE_LOSS_LOADING, PLAN_VARIABLES, RATING_VARIABLES, TABLES,
                    load_policies, one_way_plan, rebalance)

NAVY, ORANGE, GREY, GRID, INK, MUTED = "#1f4e79", "#e08a3c", "#9aa5b1", "#e6e6e6", "#222222", "#6b6b6b"
RAW_FEATURES = ["DrivAge", "BonusMalus", "VehAge", "VehPower", "Density", "VehGas", "VehBrand", "Region", "Area"]


def tweedie_glm(df, train_mask):
    """One GLM fit directly on pure premium (Tweedie power 1.5, log link), same banded variables.
    scikit-learn's version is used here because it needs far less memory than statsmodels for this one."""
    X = pd.get_dummies(df[PLAN_VARIABLES].astype(str), drop_first=True, dtype=np.float32)
    y = (df["claim_amount_capped"] / df["exposure"]).to_numpy()
    model = TweedieRegressor(power=1.5, link="log", alpha=0.0, max_iter=1000)
    model.fit(X[train_mask], y[train_mask], sample_weight=df.loc[train_mask, "exposure"])
    return rebalance(model.predict(X), train_mask, df)


def boosted_trees(df, train_mask):
    X = df[RAW_FEATURES].copy()
    for c in ["VehGas", "VehBrand", "Region", "Area"]:
        X[c] = X[c].astype("category")
    y = df["claim_amount_capped"] / df["exposure"]
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=31, min_samples_leaf=500,
                                          categorical_features="from_dtype", random_state=42)
    model.fit(X[train_mask], y[train_mask], sample_weight=df.loc[train_mask, "exposure"])
    return rebalance(model.predict(X), train_mask, df)


def gini(actual, pred, exposure):
    """Gini from the ordered Lorenz curve: sort policies from lowest to highest predicted cost and see
    how quickly actual losses pile up. 0 = no better than random, higher = better at ranking risk."""
    order = np.argsort(pred)
    cum_exp = np.cumsum(exposure[order]) / exposure.sum()
    cum_loss = np.cumsum(actual[order]) / actual.sum()
    cum_exp, cum_loss = np.r_[0, cum_exp], np.r_[0, cum_loss]
    area = np.sum((cum_exp[1:] - cum_exp[:-1]) * (cum_loss[1:] + cum_loss[:-1]) / 2)
    return 1 - 2 * area, cum_exp, cum_loss


def deciles(sort_key, exposure):
    """Equal-exposure groups (10% of policy years each) after sorting by sort_key."""
    order = np.argsort(sort_key, kind="stable")
    cum = np.cumsum(exposure[order]) / exposure.sum()
    groups = np.empty(len(sort_key), dtype=int)
    groups[order] = np.minimum((cum * 10).astype(int), 9) + 1
    return groups


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def main():
    df = load_policies()
    plan = pd.read_parquet(CLEAN / "plan_premiums.parquet")[["policy_id", "plan_pure_premium"]]
    glm_preds = pd.read_parquet(CLEAN / "glm_predictions.parquet")[["policy_id", "pred_frequency"]]
    df = df.merge(plan, on="policy_id").merge(glm_preds, on="policy_id")
    train_mask = (df["sample"] == "train").to_numpy()

    print("Fitting benchmarks (one-way plan, Tweedie GLM, boosted trees)...")
    preds = {
        "One-way plan": one_way_plan(df, train_mask),
        "GLM rating plan": df["plan_pure_premium"].to_numpy(),
        "Tweedie GLM": tweedie_glm(df, train_mask),
        "Boosted trees": boosted_trees(df, train_mask),
    }

    test = df[~train_mask].copy()
    for name, p in preds.items():
        test[name] = p[~train_mask]
    exposure = test["exposure"].to_numpy()
    actual = (test["claim_amount_capped"] * LARGE_LOSS_LOADING).to_numpy()
    test["actual"] = actual

    # 1. Gini and overall balance on the test sample
    rows, lorenz = [], {}
    for name in preds:
        expected = (test[name] * exposure).to_numpy()
        g, cx, cy = gini(actual, test[name].to_numpy(), exposure)
        lorenz[name] = (cx, cy)
        rows.append((name, round(g, 4), round(actual.sum() / expected.sum(), 4)))
    summary = pd.DataFrame(rows, columns=["model", "gini_test", "actual_over_expected_test"])
    print("\n--- Test sample: ranking power and overall balance ---")
    print(summary.to_string(index=False))

    # 2. Lift chart: GLM plan deciles, actual vs predicted loss cost per policy year
    glm = test["GLM rating plan"].to_numpy()
    test["glm_decile"] = deciles(glm, exposure)
    lift = test.groupby("glm_decile").apply(lambda d: pd.Series({
        "exposure": d["exposure"].sum(),
        "predicted": (d["GLM rating plan"] * d["exposure"]).sum() / d["exposure"].sum(),
        "actual": d["actual"].sum() / d["exposure"].sum()}), include_groups=False).reset_index()
    lift["actual_over_predicted"] = lift["actual"] / lift["predicted"]

    # 3. Double lift: sort by GLM / one-way ratio; in each group, which plan's premium is closer to actual?
    ratio = glm / test["One-way plan"].to_numpy()
    test["dl_decile"] = deciles(ratio, exposure)
    dlift = test.groupby("dl_decile").apply(lambda d: pd.Series({
        "glm_over_one_way": (d["GLM rating plan"] * d["exposure"]).sum() / (d["One-way plan"] * d["exposure"]).sum(),
        "loss_ratio_one_way_plan": d["actual"].sum() / (d["One-way plan"] * d["exposure"]).sum(),
        "loss_ratio_glm_plan": d["actual"].sum() / (d["GLM rating plan"] * d["exposure"]).sum()}),
        include_groups=False).reset_index()

    # 4. Actual vs expected by rating variable (GLM, test sample).
    # Claim counts are the main test: losses swing with a few large claims, counts do not.
    # A claim count A/E is flagged only if it falls outside a 95% band for Poisson noise.
    test["expected_claims"] = test["pred_frequency"] * test["exposure"]
    ae = []
    labels = {col: label for label, col in RATING_VARIABLES.items()}
    for v in PLAN_VARIABLES:
        g = test.groupby(v).apply(lambda d: pd.Series({
            "exposure_share": d["exposure"].sum() / exposure.sum(),
            "actual_claims": d["claim_count"].sum(),
            "expected_claims": d["expected_claims"].sum(),
            "actual_losses": d["actual"].sum(),
            "expected_losses": (d["GLM rating plan"] * d["exposure"]).sum()}), include_groups=False).reset_index()
        g = g.rename(columns={v: "level"})
        g.insert(0, "variable", labels[v])
        g["claims_ae"] = g["actual_claims"] / g["expected_claims"]
        band = 1.96 / np.sqrt(g["expected_claims"])
        g["claims_ae_low"], g["claims_ae_high"] = 1 - band, 1 + band
        g["outside_95pct_band"] = (g["claims_ae"] < g["claims_ae_low"]) | (g["claims_ae"] > g["claims_ae_high"])
        g["losses_ae"] = g["actual_losses"] / g["expected_losses"]
        if v in BANDS:
            g = g.set_index("level").loc[BANDS[v][2]].reset_index()
        ae.append(g)
    ae = pd.concat(ae, ignore_index=True)

    for name, t in [("model_comparison", summary), ("lift_chart", lift), ("double_lift", dlift),
                    ("actual_vs_expected", ae)]:
        t.round(4).to_csv(TABLES / f"validation_{name}.csv", index=False)

    print("\n--- Lift chart (GLM plan deciles, test sample) ---")
    print(lift.round(3).to_string(index=False))
    print("\n--- Double lift (sorted by GLM / one-way premium) ---")
    print(dlift.round(3).to_string(index=False))
    flagged = ae[ae["outside_95pct_band"]]
    print(f"\n--- Claim count actual / expected: {len(flagged)} of {len(ae)} segments outside the 95% band "
          f"(about {0.05 * len(ae):.0f} expected by chance) ---")
    cols = ["variable", "level", "exposure_share", "actual_claims", "expected_claims", "claims_ae", "losses_ae"]
    print(flagged[cols].round(3).to_string(index=False) if len(flagged) else "None")

    # Charts
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x = lift["glm_decile"]
    axes[0].bar(x - 0.2, lift["predicted"], width=0.38, color=NAVY, label="Predicted (GLM plan)")
    axes[0].bar(x + 0.2, lift["actual"], width=0.38, color=ORANGE, label="Actual")
    axes[0].set_xticks(x)
    axes[0].set_xlabel("Policies grouped by predicted cost, lowest to highest (10% of policy years each)", color=MUTED)
    axes[0].set_ylabel("Loss cost per policy year", color=MUTED)
    style(axes[0], "Lift chart: the plan separates low and high risks")
    axes[0].legend(frameon=False)
    x = dlift["dl_decile"]
    axes[1].plot(x, dlift["loss_ratio_one_way_plan"], color=ORANGE, marker="o", linewidth=2, label="One-way plan")
    axes[1].plot(x, dlift["loss_ratio_glm_plan"], color=NAVY, marker="o", linewidth=2, label="GLM plan")
    axes[1].axhline(1, color=MUTED, linestyle="--", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xlabel("Policies grouped by GLM premium / one-way premium, lowest to highest", color=MUTED)
    axes[1].set_ylabel("Actual / premium charged", color=MUTED)
    style(axes[1], "Double lift: where the plans disagree, the GLM is closer to actual")
    axes[1].legend(frameon=False)
    fig.text(0.01, 0.01, "Test sample, 118,358 policies the models never saw. Actual = claims capped at 100,000 "
             "x large loss loading.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(CHARTS / "03_lift_and_double_lift.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = {"One-way plan": ORANGE, "GLM rating plan": NAVY, "Tweedie GLM": GREY, "Boosted trees": "#4a90c2"}
    for name, (cx, cy) in lorenz.items():
        g = summary.set_index("model").loc[name, "gini_test"]
        ax.plot(cx, cy, color=colors[name], linewidth=2, label=f"{name} (Gini {g:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Random (Gini 0)")
    ax.set_xlabel("Share of policy years, sorted from lowest to highest predicted cost", color=MUTED)
    ax.set_ylabel("Share of actual losses", color=MUTED)
    style(ax, "Lorenz curves on the test sample")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(CHARTS / "04_lorenz_gini.png", dpi=150)

    fig, axes = plt.subplots(4, 2, figsize=(14, 15))
    for ax, v in zip(axes.flat, PLAN_VARIABLES):
        g = ae[ae["variable"] == labels[v]]
        if v in ("VehBrand", "Region"):
            g = g.sort_values("claims_ae")
        lv = g["level"].astype(str)
        ax.fill_between(lv, g["claims_ae_low"], g["claims_ae_high"], color=GRID, step="mid", label="95% band")
        colors_ = [ORANGE if o else NAVY for o in g["outside_95pct_band"]]
        ax.scatter(lv, g["claims_ae"], color=colors_, s=40, zorder=3)
        ax.axhline(1, color=MUTED, linestyle="--", linewidth=1)
        ax.tick_params(axis="x", labelrotation=45 if len(g) > 8 else 0)
        style(ax, labels[v])
    fig.suptitle("Actual / expected claim counts by rating variable, test sample (1.00 = priced right; "
                 "orange = outside the 95% band)",
                 x=0.01, y=0.995, ha="left", fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(CHARTS / "05_actual_vs_expected.png", dpi=150)
    print("\nSaved validation tables in output/tables and charts 03, 04, 05 in output/charts")


if __name__ == "__main__":
    main()
