"""03_one_way.py
One-way analysis on the training sample: for each rating variable, claim frequency,
severity and pure premium by level, and each level's pure premium relative to the whole book.
This is the "before" view a pricing team starts from, and the baseline the GLM is compared against.

Run from the project folder:  python scripts/03_one_way.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import BANDS, CHARTS, RATING_VARIABLES, TABLES, load_policies

NAVY, GRID, INK, MUTED = "#1f4e79", "#e6e6e6", "#222222", "#6b6b6b"


def one_way(df, label, col):
    g = df.groupby(col, observed=True).agg(
        policies=("policy_id", "size"),
        exposure=("exposure", "sum"),
        claims=("claim_count", "sum"),
        loss=("claim_amount_capped", "sum"),
    ).reset_index().rename(columns={col: "level"})
    g.insert(0, "variable", label)
    g["exposure_share"] = g["exposure"] / g["exposure"].sum()
    g["frequency"] = g["claims"] / g["exposure"]
    g["severity"] = g["loss"] / g["claims"].where(g["claims"] > 0)
    g["pure_premium"] = g["loss"] / g["exposure"]
    book_pp = df["claim_amount_capped"].sum() / df["exposure"].sum()
    g["pp_relativity"] = g["pure_premium"] / book_pp
    return g


def order_levels(g, col):
    """Bands keep their natural order; coded variables (brand, region) sort by pure premium."""
    if col in BANDS:
        return g.set_index("level").loc[BANDS[col][2]].reset_index()
    if col in ("VehBrand", "Region"):
        return g.sort_values("pure_premium").reset_index(drop=True)
    return g.sort_values("level").reset_index(drop=True)


def chart(tables):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (label, g) in zip(axes.flat, tables.items()):
        ax.bar(g["level"].astype(str), g["pp_relativity"], color=NAVY, width=0.65)
        ax.axhline(1, color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(label, loc="left", fontsize=12, color=INK)
        ax.tick_params(axis="x", labelrotation=45 if len(g) > 7 else 0, labelsize=8, colors=MUTED)
        ax.tick_params(axis="y", labelsize=8, colors=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        hi = g.loc[g["pp_relativity"].idxmax()]
        ax.annotate(f"{hi['pp_relativity']:.2f}", (str(hi["level"]), hi["pp_relativity"]),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=INK)
    fig.suptitle("Pure premium by rating variable, relative to the whole book (1.00 = average)",
                 x=0.01, y=0.99, ha="left", fontsize=15, color=INK)
    fig.text(0.01, 0.955, "Training sample, 472,625 policies. Claims capped at 100,000. "
             "Dashed line = book average. Brand and region sorted from lowest to highest.",
             fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / "01_one_way_relativities.png", dpi=150)


def main():
    df = load_policies()
    train = df[df["sample"] == "train"]
    tables = {}
    for label, col in RATING_VARIABLES.items():
        tables[label] = order_levels(one_way(train, label, col), col)
    out = pd.concat(tables.values(), ignore_index=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(TABLES / "one_way.csv", index=False)
    chart(tables)

    book = train["claim_amount_capped"].sum() / train["exposure"].sum()
    print(f"Book (training sample): frequency {train['claim_count'].sum() / train['exposure'].sum():.4f}, "
          f"severity {train['claim_amount_capped'].sum() / train['claim_count'].sum():,.0f}, "
          f"pure premium {book:,.2f} per policy year (capped)\n")
    show = ["level", "exposure_share", "frequency", "severity", "pure_premium", "pp_relativity"]
    for label, g in tables.items():
        print(f"--- {label} ---")
        print(g[show].round(3).to_string(index=False), "\n")
    print(f"Saved {TABLES / 'one_way.csv'} and {CHARTS / '01_one_way_relativities.png'}")


if __name__ == "__main__":
    main()
