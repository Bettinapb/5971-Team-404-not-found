from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "eda_demo_showcase.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# EDA Demo Showcase

This notebook is a presentation-ready EDA companion for the Instacart reorder project.

It focuses on three story beats:

1. Users show strong heterogeneity in frequency, diversity, and reorder behavior.
2. Basket size and user-product interactions follow a long-tail distribution.
3. Recency strongly affects reorder probability and becomes a key predictive signal.

All plots are built from the finalized modeling table at `archive/artifacts/model_dataset_full.parquet`, so the notebook stays lightweight and demo-friendly.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path.cwd()
DATA_PATH = ROOT / "archive" / "artifacts" / "model_dataset_full.parquet"
EXPORT_DIR = ROOT / "outputs" / "eda_demo"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "axes.edgecolor": "#d9d9d9",
        "grid.color": "#e8e8e8",
        "grid.linewidth": 0.8,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.family": "DejaVu Sans",
    }
)

COLORS = {
    "ink": "#17324d",
    "coral": "#d66a4f",
    "gold": "#d9a441",
    "sage": "#617a55",
    "rose": "#b04b69",
    "mist": "#87a7b3",
}

MODEL_COLS = [
    "user_id",
    "product_id",
    "label",
    "u_total_orders",
    "u_unique_products",
    "u_reorder_ratio",
    "u_avg_basket_size",
    "u_total_items",
    "up_buy_cnt",
    "up_orders_since_last",
    "up_days_since_last",
]

df = pd.read_parquet(DATA_PATH, columns=MODEL_COLS)
user_df = (
    df[
        [
            "user_id",
            "u_total_orders",
            "u_unique_products",
            "u_reorder_ratio",
            "u_avg_basket_size",
            "u_total_items",
        ]
    ]
    .drop_duplicates("user_id")
    .reset_index(drop=True)
)
pair_df = df[
    [
        "user_id",
        "product_id",
        "label",
        "up_buy_cnt",
        "up_orders_since_last",
        "up_days_since_last",
    ]
].copy()

print(f"Loaded {len(df):,} candidate rows")
print(f"Unique users: {user_df['user_id'].nunique():,}")
print(f"Unique user-product pairs: {len(pair_df):,}")
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """summary = pd.DataFrame(
    {
        "Metric": [
            "Users",
            "Candidate rows",
            "Median total orders per user",
            "Median unique products per user",
            "Median reorder ratio",
            "Median average basket size",
        ],
        "Value": [
            f"{user_df['user_id'].nunique():,}",
            f"{len(df):,}",
            f"{user_df['u_total_orders'].median():.0f}",
            f"{user_df['u_unique_products'].median():.0f}",
            f"{user_df['u_reorder_ratio'].median():.2%}",
            f"{user_df['u_avg_basket_size'].median():.1f}",
        ],
    }
)
summary
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. User Heterogeneity

The goal here is to show that users are not a homogeneous population. Some users shop frequently, some explore broadly, and some have much stronger reorder tendencies than others.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """sample_users = user_df.sample(min(len(user_df), 15000), random_state=42).copy()

fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), constrained_layout=True)

sns.histplot(
    user_df["u_total_orders"],
    bins=40,
    ax=axes[0],
    color=COLORS["ink"],
    alpha=0.88,
    edgecolor="white",
)
axes[0].axvline(user_df["u_total_orders"].median(), color=COLORS["gold"], lw=2.5, ls="--")
axes[0].set_xscale("log")
axes[0].set_title("Order Frequency Varies Widely")
axes[0].set_xlabel("Total historical orders per user (log scale)")
axes[0].set_ylabel("Users")
axes[0].text(
    0.03,
    0.95,
    f"Median: {user_df['u_total_orders'].median():.0f}\\n90th pct: {user_df['u_total_orders'].quantile(0.9):.0f}",
    transform=axes[0].transAxes,
    va="top",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#d9d9d9"),
)

sns.histplot(
    user_df["u_unique_products"],
    bins=40,
    ax=axes[1],
    color=COLORS["sage"],
    alpha=0.88,
    edgecolor="white",
)
axes[1].axvline(user_df["u_unique_products"].median(), color=COLORS["gold"], lw=2.5, ls="--")
axes[1].set_xscale("log")
axes[1].set_title("Exploration Breadth Is Also Uneven")
axes[1].set_xlabel("Unique products purchased per user (log scale)")
axes[1].set_ylabel("Users")
axes[1].text(
    0.03,
    0.95,
    f"Median: {user_df['u_unique_products'].median():.0f}\\n90th pct: {user_df['u_unique_products'].quantile(0.9):.0f}",
    transform=axes[1].transAxes,
    va="top",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#d9d9d9"),
)

scatter = axes[2].scatter(
    sample_users["u_total_orders"],
    sample_users["u_reorder_ratio"],
    s=sample_users["u_avg_basket_size"] * 3,
    c=sample_users["u_unique_products"],
    cmap="cividis",
    alpha=0.55,
    edgecolors="none",
)
axes[2].set_xscale("log")
axes[2].set_title("Reorder Tendency Differs Across Users")
axes[2].set_xlabel("Total historical orders (log scale)")
axes[2].set_ylabel("Reorder ratio")
cb = plt.colorbar(scatter, ax=axes[2], pad=0.02)
cb.set_label("Unique products")

for ax in axes:
    sns.despine(ax=ax)

fig.savefig(EXPORT_DIR / "01_user_heterogeneity.png", dpi=200, bbox_inches="tight")
plt.show()
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """**Talk track:** the user base spans light shoppers, heavy routine shoppers, and broad explorers, which justifies segment-aware modeling instead of one-size-fits-all recommendation logic."""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Long-Tail Structure

For the demo, we show the long-tail pattern through two lenses:

- user average basket size as a stable basket-size summary
- user-product repeat counts as a direct interaction-frequency signal
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """basket_ccdf = (
    user_df["u_avg_basket_size"]
    .value_counts()
    .sort_index()
    .rename_axis("avg_basket_size")
    .reset_index(name="n_users")
)
basket_ccdf["share_users"] = basket_ccdf["n_users"] / basket_ccdf["n_users"].sum()
basket_ccdf["ccdf"] = basket_ccdf["share_users"][::-1].cumsum()[::-1]

buy_count_summary = (
    pair_df["up_buy_cnt"]
    .value_counts()
    .sort_index()
    .rename_axis("up_buy_cnt")
    .reset_index(name="n_pairs")
)
buy_count_summary["share_pairs"] = buy_count_summary["n_pairs"] / buy_count_summary["n_pairs"].sum()
buy_count_summary["bucket"] = pd.cut(
    buy_count_summary["up_buy_cnt"],
    bins=[0, 1, 2, 5, 10, 20, 100],
    labels=["1", "2", "3-5", "6-10", "11-20", "21+"],
)
bucket_view = (
    buy_count_summary.groupby("bucket", observed=False)["n_pairs"].sum().reset_index()
)
bucket_view["share_pairs"] = bucket_view["n_pairs"] / bucket_view["n_pairs"].sum()

fig = plt.figure(figsize=(18, 6.5), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 0.9])

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(
    basket_ccdf["avg_basket_size"],
    basket_ccdf["ccdf"],
    color=COLORS["coral"],
    lw=3,
)
ax1.fill_between(
    basket_ccdf["avg_basket_size"],
    basket_ccdf["ccdf"],
    color=COLORS["coral"],
    alpha=0.15,
)
ax1.set_title("Basket Size Summary Has a Long Tail")
ax1.set_xlabel("Average basket size per user")
ax1.set_ylabel("Share of users with basket size >= x")
ax1.yaxis.set_major_formatter(PercentFormatter(1.0))

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(
    buy_count_summary["up_buy_cnt"],
    buy_count_summary["share_pairs"],
    color=COLORS["ink"],
    lw=2.8,
    marker="o",
    ms=4,
)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_title("User-Product Repeats Are Even More Skewed")
ax2.set_xlabel("Times a user bought the same product (log scale)")
ax2.set_ylabel("Share of user-product pairs (log scale)")

ax3 = fig.add_subplot(gs[0, 2])
sns.barplot(
    data=bucket_view,
    x="bucket",
    y="share_pairs",
    palette=[COLORS["mist"], COLORS["sage"], COLORS["gold"], COLORS["coral"], COLORS["rose"], COLORS["ink"]],
    ax=ax3,
)
ax3.set_title("Most Pairs Occur Only a Few Times")
ax3.set_xlabel("Repeat-count bucket")
ax3.set_ylabel("Share of user-product pairs")
ax3.yaxis.set_major_formatter(PercentFormatter(1.0))
for patch, value in zip(ax3.patches, bucket_view["share_pairs"]):
    ax3.text(
        patch.get_x() + patch.get_width() / 2,
        value + 0.01,
        f"{value:.0%}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

for ax in [ax1, ax2, ax3]:
    sns.despine(ax=ax)

fig.savefig(EXPORT_DIR / "02_long_tail_structure.png", dpi=200, bbox_inches="tight")
plt.show()
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """**Talk track:** most behavior sits in the sparse tail, while only a small slice of users or user-product pairs generate many repeated interactions. That is why historical interaction features matter so much."""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. Recency as a Predictive Signal

This section links feature engineering directly back to behavior: recently purchased products are far more likely to appear again in the target order.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """recency_orders = pair_df[["up_orders_since_last", "label"]].copy()
recency_orders["orders_bin"] = pd.cut(
    recency_orders["up_orders_since_last"],
    bins=[0, 1, 2, 3, 5, 8, 13, 21, 100],
    labels=["1", "2", "3", "4-5", "6-8", "9-13", "14-21", "22+"],
    include_lowest=True,
)
recency_orders_view = (
    recency_orders.groupby("orders_bin", observed=False)
    .agg(reorder_prob=("label", "mean"), n_pairs=("label", "size"))
    .reset_index()
)
recency_orders_view["pair_share"] = recency_orders_view["n_pairs"] / recency_orders_view["n_pairs"].sum()

recency_days = pair_df[["up_days_since_last", "label"]].copy()
recency_days["days_bin"] = pd.cut(
    recency_days["up_days_since_last"],
    bins=[-0.1, 7, 14, 30, 60, 90, 180, 1000],
    labels=["0-7", "8-14", "15-30", "31-60", "61-90", "91-180", "180+"],
)
recency_days_view = (
    recency_days.groupby("days_bin", observed=False)
    .agg(reorder_prob=("label", "mean"), n_pairs=("label", "size"))
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(18, 6.3), constrained_layout=True)

bars = axes[0].bar(
    recency_orders_view["orders_bin"].astype(str),
    recency_orders_view["pair_share"],
    color=COLORS["mist"],
    alpha=0.45,
    label="Pair share",
)
ax0b = axes[0].twinx()
ax0b.plot(
    recency_orders_view["orders_bin"].astype(str),
    recency_orders_view["reorder_prob"],
    color=COLORS["rose"],
    lw=3,
    marker="o",
    ms=8,
    label="Reorder probability",
)
axes[0].set_title("Fewer Orders Since Last Purchase -> Higher Reorder Chance")
axes[0].set_xlabel("Orders since last purchase")
axes[0].set_ylabel("Share of candidate pairs")
axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
ax0b.set_ylabel("Observed reorder probability")
ax0b.yaxis.set_major_formatter(PercentFormatter(1.0))

sns.lineplot(
    data=recency_days_view,
    x="days_bin",
    y="reorder_prob",
    marker="o",
    linewidth=3,
    color=COLORS["coral"],
    ax=axes[1],
)
axes[1].set_title("The Same Pattern Appears in Calendar Time")
axes[1].set_xlabel("Days since last purchase")
axes[1].set_ylabel("Observed reorder probability")
axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))

for ax in axes:
    sns.despine(ax=ax)
sns.despine(ax=ax0b, left=False)

fig.savefig(EXPORT_DIR / "03_recency_signal.png", dpi=200, bbox_inches="tight")
plt.show()
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """**Talk track:** the relationship is monotonic enough that recency should not just be discussed in EDA; it deserves a central role in the feature set, which is exactly why `up_orders_since_last` and `up_days_since_last` show up in the final model."""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Exported Figures

Running the notebook saves presentation-ready images to `outputs/eda_demo/`:

- `01_user_heterogeneity.png`
- `02_long_tail_structure.png`
- `03_recency_signal.png`

That gives you both a notebook version for live demo and standalone image assets for slides.
"""
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
    }

    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
