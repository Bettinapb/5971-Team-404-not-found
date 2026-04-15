from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    import kagglehub
except Exception:  # pragma: no cover - optional import for local fallback
    kagglehub = None


RANDOM_STATE = 42
BASE_FEATURE_COLS = [
    "up_orders_since_last",
    "up_days_since_last",
    "up_freq",
    "up_buy_cnt",
    "up_reorder_ratio",
    "p_reorder_ratio",
    "u_total_orders",
    "cluster_product_target_rate",
    "p_avg_cart_order",
    "u_reorder_ratio",
    "u_unique_products",
    "p_total_purchases",
    "up_first_order",
    "u_avg_days_between_orders",
    "p_unique_users",
    "up_last_order",
    "u_total_items",
    "u_avg_basket_size",
    "up_avg_cart_order",
    "apriori_rule_hits",
    "user_cluster_1",
    "user_cluster_2",
    "user_cluster_3",
]
ALS_FEATURE_COLS = ["als_dot", "als_cos"]
CATBOOST_CAT_COLS = ["aisle_id", "department_id"]


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def build_aisle_rules(
    single_counts: dict[str, int],
    pair_counts: dict[tuple[str, str], int],
    n_baskets: int,
    min_support: float,
    min_confidence: float,
    min_lift: float,
) -> pd.DataFrame:
    rows = []
    for (a, b), pair_cnt in pair_counts.items():
        support = pair_cnt / n_baskets
        conf_a_to_b = pair_cnt / single_counts[a]
        conf_b_to_a = pair_cnt / single_counts[b]
        lift_a_to_b = conf_a_to_b / (single_counts[b] / n_baskets)
        lift_b_to_a = conf_b_to_a / (single_counts[a] / n_baskets)

        rows.append(
            {
                "antecedent": a,
                "consequent": b,
                "pair_count": pair_cnt,
                "support": support,
                "confidence": conf_a_to_b,
                "lift": lift_a_to_b,
            }
        )
        rows.append(
            {
                "antecedent": b,
                "consequent": a,
                "pair_count": pair_cnt,
                "support": support,
                "confidence": conf_b_to_a,
                "lift": lift_b_to_a,
            }
        )

    rules = pd.DataFrame(rows)
    if rules.empty:
        return rules

    rules = rules[
        (rules["support"] >= min_support)
        & (rules["confidence"] >= min_confidence)
        & (rules["lift"] >= min_lift)
    ].copy()
    if rules.empty:
        return rules

    return rules.sort_values(
        ["lift", "confidence", "pair_count"], ascending=[False, False, False]
    ).reset_index(drop=True)


def add_als_features(
    model_df: pd.DataFrame,
    up_agg: pd.DataFrame,
    include_als: bool,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    status = {"als_status": "skipped"}
    feature_cols: list[str] = []

    if not include_als:
        model_df["als_dot"] = 0.0
        model_df["als_cos"] = 0.0
        status["als_status"] = "disabled"
        return model_df, ALS_FEATURE_COLS.copy(), status

    try:
        from scipy import sparse
        import implicit

        work = up_agg[["user_id", "product_id", "up_buy_cnt"]].copy()
        u_codes, u_uniques = pd.factorize(work["user_id"], sort=False)
        p_codes, p_uniques = pd.factorize(work["product_id"], sort=False)

        rows = u_codes.astype(np.int32)
        cols = p_codes.astype(np.int32)
        values = work["up_buy_cnt"].astype(np.float32).values

        n_users = int(rows.max()) + 1 if len(rows) else 0
        n_items = int(cols.max()) + 1 if len(cols) else 0
        if n_users <= 0 or n_items <= 0:
            raise ValueError(f"Empty ALS matrix: n_users={n_users}, n_items={n_items}")

        ui = sparse.coo_matrix((values, (rows, cols)), shape=(n_users, n_items)).tocsr()
        als = implicit.als.AlternatingLeastSquares(
            factors=32,
            regularization=0.01,
            iterations=15,
            use_gpu=False,
            random_state=RANDOM_STATE,
        )
        ui_conf = (ui * 15.0).astype(np.float32)
        als.fit(ui_conf)

        user_factors = als.user_factors
        item_factors = als.item_factors
        if user_factors.shape[0] == n_items and item_factors.shape[0] == n_users:
            user_factors, item_factors = item_factors, user_factors
        if user_factors.shape[0] != n_users or item_factors.shape[0] != n_items:
            raise ValueError(
                "Unexpected ALS factor shapes: "
                f"{user_factors.shape=} {item_factors.shape=} {n_users=} {n_items=}"
            )

        user_map = pd.Series(np.arange(len(u_uniques), dtype=np.int32), index=u_uniques)
        item_map = pd.Series(np.arange(len(p_uniques), dtype=np.int32), index=p_uniques)
        model_user_codes = model_df["user_id"].map(user_map)
        model_item_codes = model_df["product_id"].map(item_map)

        valid = model_user_codes.notna() & model_item_codes.notna()
        model_df["als_dot"] = 0.0
        model_df["als_cos"] = 0.0

        u_idx = model_user_codes[valid].astype(np.int32).to_numpy()
        i_idx = model_item_codes[valid].astype(np.int32).to_numpy()
        uf = user_factors[u_idx]
        vf = item_factors[i_idx]
        dots = np.sum(uf * vf, axis=1).astype(np.float32)
        denom = np.maximum(
            np.linalg.norm(uf, axis=1) * np.linalg.norm(vf, axis=1),
            1e-8,
        )
        cos = (dots / denom).astype(np.float32)

        model_df.loc[valid, "als_dot"] = dots
        model_df.loc[valid, "als_cos"] = cos
        status["als_status"] = "enabled"
        feature_cols = ALS_FEATURE_COLS.copy()
    except Exception as exc:  # pragma: no cover - best effort feature
        model_df["als_dot"] = 0.0
        model_df["als_cos"] = 0.0
        status["als_status"] = f"failed: {exc}"
        feature_cols = ALS_FEATURE_COLS.copy()

    return model_df, feature_cols, status


def build_model_dataset(
    data_dir: Path,
    output_path: Path,
    sample_users: int | None = None,
    include_als: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not data_dir.exists():
        if kagglehub is None:
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}. "
                "kagglehub is also unavailable, so the raw dataset cannot be downloaded automatically."
            )
        dataset_path = Path(
            kagglehub.dataset_download(
                "yasserh/instacart-online-grocery-basket-analysis-dataset"
            )
        )
        data_dir = dataset_path
    required_files = [
        "orders.csv",
        "order_products__prior.csv",
        "order_products__train.csv",
        "products.csv",
        "aisles.csv",
        "departments.csv",
    ]
    missing_files = [name for name in required_files if not (data_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required raw files under {data_dir}: {missing_files}"
        )

    orders = pd.read_csv(data_dir / "orders.csv")
    op_prior = pd.read_csv(data_dir / "order_products__prior.csv")
    op_train = pd.read_csv(data_dir / "order_products__train.csv")
    products = pd.read_csv(data_dir / "products.csv")[
        ["product_id", "aisle_id", "department_id"]
    ]
    aisles = pd.read_csv(data_dir / "aisles.csv")[["aisle_id", "aisle"]]

    target_orders = orders.loc[
        orders["eval_set"] == "train", ["order_id", "user_id", "order_number"]
    ].copy()
    target_orders = target_orders.rename(
        columns={"order_id": "target_order_id", "order_number": "target_order_number"}
    )

    if sample_users is not None:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_uids = rng.choice(
            target_orders["user_id"].unique(),
            size=min(sample_users, target_orders["user_id"].nunique()),
            replace=False,
        )
        target_orders = target_orders[
            target_orders["user_id"].isin(sample_uids)
        ].copy()

    prior_orders = orders.loc[
        orders["eval_set"] == "prior",
        [
            "order_id",
            "user_id",
            "order_number",
            "days_since_prior_order",
            "order_dow",
            "order_hour_of_day",
        ],
    ].copy()
    prior_orders = prior_orders.merge(
        target_orders[["user_id", "target_order_id", "target_order_number"]],
        on="user_id",
        how="inner",
    )

    history = op_prior.merge(
        prior_orders[
            [
                "order_id",
                "user_id",
                "order_number",
                "days_since_prior_order",
                "target_order_number",
            ]
        ],
        on="order_id",
        how="inner",
    )

    positives = op_train.merge(
        target_orders[["target_order_id", "user_id"]],
        left_on="order_id",
        right_on="target_order_id",
        how="inner",
    )
    positives = positives[["user_id", "product_id"]].drop_duplicates().assign(label=1)

    candidates = history[["user_id", "product_id"]].drop_duplicates().copy()

    up_agg = (
        history.groupby(["user_id", "product_id"]).agg(
            up_buy_cnt=("order_id", "size"),
            up_reorder_cnt=("reordered", "sum"),
            up_reorder_ratio=("reordered", "mean"),
            up_last_order=("order_number", "max"),
            up_first_order=("order_number", "min"),
            up_avg_cart_order=("add_to_cart_order", "mean"),
            up_days_since_last_raw=("days_since_prior_order", "mean"),
        )
    ).reset_index()

    prod_agg = (
        history.groupby("product_id").agg(
            p_total_purchases=("order_id", "size"),
            p_reorder_ratio=("reordered", "mean"),
            p_avg_cart_order=("add_to_cart_order", "mean"),
            p_unique_users=("user_id", "nunique"),
        )
    ).reset_index()

    user_order_size = (
        history.groupby(["user_id", "order_id"]).size().rename("basket_size").reset_index()
    )
    user_agg = (
        history.groupby("user_id").agg(
            u_total_orders=("order_id", "nunique"),
            u_reorder_ratio=("reordered", "mean"),
            u_unique_products=("product_id", "nunique"),
            u_total_items=("order_id", "size"),
            u_avg_days_between_orders=("days_since_prior_order", "mean"),
        )
    ).reset_index()
    user_agg = user_agg.merge(
        user_order_size.groupby("user_id")["basket_size"]
        .mean()
        .rename("u_avg_basket_size")
        .reset_index(),
        on="user_id",
        how="left",
    )

    model_df = candidates.merge(up_agg, on=["user_id", "product_id"], how="left")
    model_df = model_df.merge(
        target_orders[["user_id", "target_order_number"]], on="user_id", how="left"
    )
    model_df = model_df.merge(prod_agg, on="product_id", how="left")
    model_df = model_df.merge(user_agg, on="user_id", how="left")
    model_df = model_df.merge(positives, on=["user_id", "product_id"], how="left")
    model_df["label"] = model_df["label"].fillna(0).astype(int)

    model_df["up_orders_since_last"] = (
        model_df["target_order_number"] - model_df["up_last_order"]
    ).clip(lower=0)
    model_df["up_days_since_last"] = (
        model_df["up_orders_since_last"] * model_df["u_avg_days_between_orders"]
    ).fillna(0)
    model_df["up_freq"] = (
        model_df["up_buy_cnt"] / model_df["u_total_orders"].replace(0, np.nan)
    ).fillna(0)

    prod_meta = products.merge(aisles, on="aisle_id", how="left")
    prod_aisle = prod_meta[["product_id", "aisle"]].copy()
    history_with_aisle = history.merge(prod_aisle, on="product_id", how="left")

    basket_aisles = (
        history_with_aisle.groupby("order_id")["aisle"]
        .apply(lambda x: sorted(set(x.dropna().astype(str))))
        .reset_index(name="aisle_basket")
    )
    basket_aisles = basket_aisles[basket_aisles["aisle_basket"].str.len() >= 2].copy()

    single_counts: dict[str, int] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    for basket in basket_aisles["aisle_basket"]:
        for item in basket:
            single_counts[item] = single_counts.get(item, 0) + 1
        for i in range(len(basket)):
            for j in range(i + 1, len(basket)):
                pair = (basket[i], basket[j])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    rules_filtered = build_aisle_rules(
        single_counts=single_counts,
        pair_counts=pair_counts,
        n_baskets=max(len(basket_aisles), 1),
        min_support=0.01,
        min_confidence=0.30,
        min_lift=1.30,
    )

    ante_to_cons: dict[str, set[str]] = {}
    if not rules_filtered.empty:
        for a, b in rules_filtered[["antecedent", "consequent"]].itertuples(index=False):
            ante_to_cons.setdefault(a, set()).add(b)

    user_last_order = (
        prior_orders.groupby("user_id")["order_number"]
        .max()
        .rename("last_order_number")
        .reset_index()
    )
    last_orders = prior_orders.merge(user_last_order, on="user_id", how="inner")
    last_orders = last_orders[
        last_orders["order_number"] == last_orders["last_order_number"]
    ][["user_id", "order_id"]].drop_duplicates()

    last_basket_aisles = (
        last_orders.merge(history_with_aisle[["order_id", "aisle"]], on="order_id", how="left")
        .groupby("user_id")["aisle"]
        .apply(lambda x: set(x.dropna().astype(str)))
    )

    user_allowed_aisles: dict[int, set[str]] = {}
    for uid, aisle_set in last_basket_aisles.items():
        allowed: set[str] = set()
        for aisle in aisle_set:
            allowed.update(ante_to_cons.get(aisle, set()))
        user_allowed_aisles[uid] = allowed

    model_df = model_df.merge(prod_meta, on="product_id", how="left")
    model_df["apriori_rule_hits"] = model_df.apply(
        lambda row: int(row["aisle"] in user_allowed_aisles.get(row["user_id"], set())),
        axis=1,
    )

    cluster_source = user_agg[
        [
            "user_id",
            "u_total_orders",
            "u_avg_days_between_orders",
            "u_avg_basket_size",
            "u_reorder_ratio",
            "u_unique_products",
        ]
    ].copy()
    cluster_feature_cols = [
        "u_total_orders",
        "u_avg_days_between_orders",
        "u_avg_basket_size",
        "u_reorder_ratio",
        "u_unique_products",
    ]
    for col in ["u_total_orders", "u_avg_basket_size", "u_unique_products"]:
        cluster_source[col] = np.log1p(cluster_source[col])

    cluster_x = cluster_source[cluster_feature_cols].fillna(0)
    cluster_scaler = StandardScaler()
    cluster_x_scaled = cluster_scaler.fit_transform(cluster_x)
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20)
    cluster_source["user_cluster"] = kmeans.fit_predict(cluster_x_scaled)

    model_df = model_df.merge(
        cluster_source[["user_id", "user_cluster"]], on="user_id", how="left"
    )
    model_df["user_cluster"] = model_df["user_cluster"].fillna(0).astype(int)

    cluster_dummies = pd.get_dummies(model_df["user_cluster"], prefix="user_cluster", dtype=int)
    for col in ["user_cluster_1", "user_cluster_2", "user_cluster_3"]:
        model_df[col] = cluster_dummies[col] if col in cluster_dummies.columns else 0

    cp_rate = (
        model_df.groupby(["user_cluster", "product_id"])["label"]
        .mean()
        .rename("cluster_product_target_rate")
        .reset_index()
    )
    model_df = model_df.merge(cp_rate, on=["user_cluster", "product_id"], how="left")
    model_df["cluster_product_target_rate"] = model_df["cluster_product_target_rate"].fillna(
        model_df["label"].mean()
    )

    model_df, als_feature_cols, als_status = add_als_features(
        model_df=model_df,
        up_agg=up_agg,
        include_als=include_als,
    )

    for col in BASE_FEATURE_COLS + CATBOOST_CAT_COLS + als_feature_cols:
        if col not in model_df.columns:
            model_df[col] = 0

    model_df[BASE_FEATURE_COLS + als_feature_cols] = (
        model_df[BASE_FEATURE_COLS + als_feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    model_df[CATBOOST_CAT_COLS] = model_df[CATBOOST_CAT_COLS].fillna(-1).astype(int)

    export_cols = unique_in_order(
        ["user_id", "product_id", "label", "u_avg_basket_size", "user_cluster"]
        + BASE_FEATURE_COLS
        + als_feature_cols
        + CATBOOST_CAT_COLS
    )
    model_df = model_df[export_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_parquet(output_path, index=False)

    metadata = {
        "data_path": str(output_path),
        "raw_data_dir": str(data_dir),
        "rows": int(model_df.shape[0]),
        "users": int(model_df["user_id"].nunique()),
        "positive_rate": float(model_df["label"].mean()),
        "base_feature_cols": BASE_FEATURE_COLS,
        "als_feature_cols": als_feature_cols,
        "numeric_feature_cols": BASE_FEATURE_COLS + als_feature_cols,
        "catboost_cat_cols": CATBOOST_CAT_COLS,
        "apriori_rule_count": int(len(rules_filtered)),
        "sample_users": sample_users,
        "include_als": include_als,
    }
    metadata.update(als_status)
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return model_df, metadata


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a model-ready Instacart dataset for final-model experiments."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the raw Instacart CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/model_dataset_full.parquet"),
        help="Output parquet path.",
    )
    parser.add_argument(
        "--sample-users",
        type=int,
        default=None,
        help="Optional sampled user count. Omit for full data.",
    )
    parser.add_argument(
        "--disable-als",
        action="store_true",
        help="Skip ALS feature generation even if implicit is installed.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    model_df, metadata = build_model_dataset(
        data_dir=args.data_dir,
        output_path=args.output,
        sample_users=args.sample_users,
        include_als=not args.disable_als,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "shape": model_df.shape,
                "users": metadata["users"],
                "positive_rate": round(metadata["positive_rate"], 6),
                "apriori_rule_count": metadata["apriori_rule_count"],
                "als_status": metadata["als_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


#{
    # "output": "artifacts/model_dataset_full.parquet",
    # "shape": [ 8474661,31],
    # "users": 131209,
    # "positive_rate": 0.0978,
    # "apriori_rule_count": 138,
# "als_status": "enabled"}
