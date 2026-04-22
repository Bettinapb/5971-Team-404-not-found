from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path(__file__).resolve().parent / "leakage_safe_preprocess.ipynb"


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).lstrip("\n").splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).lstrip("\n").splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        md_cell(
            """
            # Leakage-Safe Preprocess

            This notebook version mirrors `leakage_safe_preprocess.py`.

            It:

            1. resolves the raw Instacart source tables
            2. rebuilds the supervised cohort and target/history definition
            3. removes `"missing"` metadata noise
            4. writes the globally safe base bundle to `Final version/artifacts/base_bundle/`
            """
        ),
        code_cell(
            """
            from __future__ import annotations

            import json
            from pathlib import Path

            import numpy as np
            import pandas as pd

            try:
                import kagglehub
                HAS_KAGGLEHUB = True
            except Exception:
                kagglehub = None
                HAS_KAGGLEHUB = False

            print({"HAS_KAGGLEHUB": HAS_KAGGLEHUB})
            """
        ),
        code_cell(
            """
            RANDOM_STATE = 42
            DEFAULT_DATASET = "yasserh/instacart-online-grocery-basket-analysis-dataset"

            BASE_DIR = Path.cwd().resolve()
            if (BASE_DIR / "Final version").exists():
                PROJECT_ROOT = BASE_DIR
                WORKDIR = BASE_DIR / "Final version"
            else:
                PROJECT_ROOT = BASE_DIR.parent if BASE_DIR.name == "Final version" else BASE_DIR
                WORKDIR = BASE_DIR if BASE_DIR.name == "Final version" else BASE_DIR

            DATA_DIR = None
            DATASET_NAME = DEFAULT_DATASET
            OUTPUT_DIR = WORKDIR / "artifacts" / "base_bundle"

            print({
                "PROJECT_ROOT": str(PROJECT_ROOT),
                "WORKDIR": str(WORKDIR),
                "DATA_DIR": str(DATA_DIR) if DATA_DIR is not None else None,
                "DATASET_NAME": DATASET_NAME,
                "OUTPUT_DIR": str(OUTPUT_DIR),
            })
            """
        ),
        code_cell(
            """
            def resolve_raw_data_dir(data_dir: Path | None, dataset_name: str) -> Path:
                if data_dir is not None and data_dir.exists():
                    return data_dir

                if kagglehub is None:
                    raise FileNotFoundError(
                        "Raw Instacart data directory was not provided and `kagglehub` is unavailable. "
                        "Install kagglehub or set DATA_DIR."
                    )

                downloaded = Path(kagglehub.dataset_download(dataset_name))
                return downloaded


            def verify_required_files(data_dir: Path) -> None:
                required = [
                    "orders.csv",
                    "order_products__prior.csv",
                    "order_products__train.csv",
                    "products.csv",
                    "aisles.csv",
                    "departments.csv",
                ]
                missing = [name for name in required if not (data_dir / name).exists()]
                if missing:
                    raise FileNotFoundError(f"Missing required raw files under {data_dir}: {missing}")


            def read_raw_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
                orders = pd.read_csv(data_dir / "orders.csv")
                prior = pd.read_csv(data_dir / "order_products__prior.csv")
                train = pd.read_csv(data_dir / "order_products__train.csv")
                products = pd.read_csv(data_dir / "products.csv")
                aisles = pd.read_csv(data_dir / "aisles.csv")
                departments = pd.read_csv(data_dir / "departments.csv")

                orders["days_since_prior_order"] = orders["days_since_prior_order"].fillna(0).astype(np.float32)
                orders = orders.sort_values(["user_id", "order_number"]).reset_index(drop=True)
                orders["user_cum_days"] = (
                    orders.groupby("user_id")["days_since_prior_order"].cumsum().astype(np.float32)
                )

                return {
                    "orders": orders,
                    "prior": prior,
                    "train": train,
                    "products": products,
                    "aisles": aisles,
                    "departments": departments,
                }


            def is_literal_missing(series: pd.Series) -> pd.Series:
                return series.astype(str).str.strip().str.lower().eq("missing")
            """
        ),
        code_cell(
            """
            def build_base_bundle(
                orders: pd.DataFrame,
                prior: pd.DataFrame,
                train: pd.DataFrame,
                products: pd.DataFrame,
                aisles: pd.DataFrame,
                departments: pd.DataFrame,
            ) -> dict[str, pd.DataFrame]:
                aisle_missing_mask = is_literal_missing(aisles["aisle"])
                department_missing_mask = is_literal_missing(departments["department"])
                product_missing_mask = is_literal_missing(products["product_name"])

                missing_aisle_ids = set(aisles.loc[aisle_missing_mask, "aisle_id"])
                missing_department_ids = set(departments.loc[department_missing_mask, "department_id"])
                invalid_product_mask = (
                    product_missing_mask
                    | products["aisle_id"].isin(missing_aisle_ids)
                    | products["department_id"].isin(missing_department_ids)
                )
                valid_products = products.loc[~invalid_product_mask].copy()
                valid_product_ids = set(valid_products["product_id"])

                aisles = aisles.loc[~aisle_missing_mask].copy()
                departments = departments.loc[~department_missing_mask].copy()
                prior = prior[prior["product_id"].isin(valid_product_ids)].copy()
                train = train[train["product_id"].isin(valid_product_ids)].copy()

                product_meta = (
                    valid_products.merge(aisles, on="aisle_id", how="left")
                    .merge(departments, on="department_id", how="left")
                    .rename(columns={"aisle": "aisle_name", "department": "department_name"})
                )

                train_orders = orders.loc[orders["eval_set"] == "train"].copy()
                test_orders = orders.loc[orders["eval_set"] == "test"].copy()
                supervised_users = set(orders.loc[orders["eval_set"].isin(["train", "test"]), "user_id"])
                prior_orders = orders.loc[
                    (orders["eval_set"] == "prior") & (orders["user_id"].isin(supervised_users))
                ].copy()
                orders = orders.loc[orders["user_id"].isin(supervised_users)].copy()

                train_target_orders = train_orders[
                    [
                        "order_id",
                        "user_id",
                        "order_number",
                        "order_dow",
                        "order_hour_of_day",
                        "days_since_prior_order",
                        "user_cum_days",
                    ]
                ].rename(
                    columns={
                        "order_id": "target_order_id",
                        "order_number": "target_order_number",
                        "order_dow": "target_order_dow",
                        "order_hour_of_day": "target_order_hour_of_day",
                        "days_since_prior_order": "target_days_since_prior_order",
                        "user_cum_days": "target_user_cum_days",
                    }
                )
                train_target_orders["target_source"] = "train_order"

                test_target_orders = (
                    prior_orders.loc[prior_orders["user_id"].isin(test_orders["user_id"])]
                    .sort_values(["user_id", "order_number"])
                    .groupby("user_id")
                    .tail(1)[
                        [
                            "order_id",
                            "user_id",
                            "order_number",
                            "order_dow",
                            "order_hour_of_day",
                            "days_since_prior_order",
                            "user_cum_days",
                        ]
                    ]
                    .rename(
                        columns={
                            "order_id": "target_order_id",
                            "order_number": "target_order_number",
                            "order_dow": "target_order_dow",
                            "order_hour_of_day": "target_order_hour_of_day",
                            "days_since_prior_order": "target_days_since_prior_order",
                            "user_cum_days": "target_user_cum_days",
                        }
                    )
                )
                test_target_orders["target_source"] = "prior_as_target"

                target_orders = (
                    pd.concat([train_target_orders, test_target_orders], ignore_index=True)
                    .sort_values("user_id")
                    .reset_index(drop=True)
                )

                prior_detail_full = (
                    prior.merge(
                        prior_orders[
                            [
                                "order_id",
                                "user_id",
                                "order_number",
                                "order_dow",
                                "order_hour_of_day",
                                "days_since_prior_order",
                                "user_cum_days",
                            ]
                        ],
                        on="order_id",
                        how="left",
                    )
                    .merge(product_meta, on="product_id", how="left")
                    .sort_values(["user_id", "order_number", "add_to_cart_order", "product_id"])
                    .reset_index(drop=True)
                )

                train_target_detail = (
                    train.merge(
                        train_target_orders[["target_order_id", "user_id", "target_order_number", "target_source"]],
                        left_on="order_id",
                        right_on="target_order_id",
                        how="left",
                    )
                    .merge(product_meta, on="product_id", how="left")
                    .sort_values(["user_id", "target_order_number", "add_to_cart_order", "product_id"])
                    .reset_index(drop=True)
                )
                test_target_detail = (
                    prior_detail_full.merge(
                        test_target_orders[["target_order_id", "user_id", "target_order_number", "target_source"]],
                        left_on=["order_id", "user_id"],
                        right_on=["target_order_id", "user_id"],
                        how="inner",
                    )
                    .drop(columns=["order_id", "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order", "user_cum_days"])
                    .sort_values(["user_id", "target_order_number", "add_to_cart_order", "product_id"])
                    .reset_index(drop=True)
                )
                target_detail = (
                    pd.concat([train_target_detail, test_target_detail], ignore_index=True)
                    .sort_values(["user_id", "target_order_number", "add_to_cart_order", "product_id"])
                    .reset_index(drop=True)
                )

                history_orders = prior_orders.merge(
                    target_orders[["user_id", "target_order_number"]],
                    on="user_id",
                    how="inner",
                )
                history_orders = history_orders[history_orders["order_number"] < history_orders["target_order_number"]].copy()

                history_detail = (
                    prior.merge(
                        history_orders[
                            [
                                "order_id",
                                "user_id",
                                "order_number",
                                "order_dow",
                                "order_hour_of_day",
                                "days_since_prior_order",
                                "user_cum_days",
                            ]
                        ],
                        on="order_id",
                        how="inner",
                    )
                    .merge(product_meta, on="product_id", how="left")
                    .sort_values(["user_id", "order_number", "add_to_cart_order", "product_id"])
                    .reset_index(drop=True)
                )

                labels = (
                    target_detail[["user_id", "product_id"]]
                    .drop_duplicates()
                    .assign(label=np.int8(1))
                    .sort_values(["user_id", "product_id"])
                    .reset_index(drop=True)
                )

                candidate_universe = (
                    history_detail[["user_id", "product_id"]]
                    .drop_duplicates()
                    .sort_values(["user_id", "product_id"])
                    .reset_index(drop=True)
                )

                user_last_history_order = (
                    history_orders.groupby("user_id")
                    .agg(
                        last_history_order_id=("order_id", "last"),
                        last_history_order_number=("order_number", "last"),
                        last_history_user_cum_days=("user_cum_days", "last"),
                    )
                    .reset_index()
                )

                last_history_products = (
                    history_detail.merge(
                        user_last_history_order[["user_id", "last_history_order_id"]],
                        left_on=["user_id", "order_id"],
                        right_on=["user_id", "last_history_order_id"],
                        how="inner",
                    )[["user_id", "product_id", "aisle_id", "aisle_name"]]
                    .sort_values(["user_id", "product_id"])
                    .reset_index(drop=True)
                )

                order_sizes = (
                    history_detail.groupby(["user_id", "order_id"])
                    .size()
                    .rename("basket_size")
                    .reset_index()
                )

                order_level_history = (
                    history_orders[
                        [
                            "order_id",
                            "user_id",
                            "order_number",
                            "order_dow",
                            "order_hour_of_day",
                            "days_since_prior_order",
                            "user_cum_days",
                        ]
                    ]
                    .merge(order_sizes, on=["user_id", "order_id"], how="left")
                    .sort_values(["user_id", "order_number"])
                    .reset_index(drop=True)
                )

                return {
                    "orders_enriched": orders.reset_index(drop=True),
                    "product_meta": product_meta.sort_values("product_id").reset_index(drop=True),
                    "prior_detail": prior_detail_full,
                    "history_detail": history_detail,
                    "target_orders": target_orders.sort_values("user_id").reset_index(drop=True),
                    "target_detail": target_detail,
                    "labels": labels,
                    "candidate_universe": candidate_universe,
                    "user_last_history_order": user_last_history_order.sort_values("user_id").reset_index(drop=True),
                    "last_history_products": last_history_products,
                    "order_level_history": order_level_history,
                }


            def write_bundle(bundle: dict[str, pd.DataFrame], output_dir: Path, raw_data_dir: Path) -> None:
                output_dir.mkdir(parents=True, exist_ok=True)
                for name, frame in bundle.items():
                    frame.to_parquet(output_dir / f"{name}.parquet", index=False)

                metadata = {
                    "raw_data_dir": str(raw_data_dir),
                    "tables": {name: {"rows": int(len(frame)), "cols": int(frame.shape[1])} for name, frame in bundle.items()},
                    "random_state": RANDOM_STATE,
                    "design": {
                        "supervised_cohort": "users with eval_set in {train, test}",
                        "candidate_definition": "previously purchased products only",
                        "target_definition": (
                            "For eval_set=train users, target is the train order. "
                            "For eval_set=test users, target is the last prior order with product-level information."
                        ),
                        "history_definition": "All orders earlier than the chosen target order",
                        "label_definition": "product appears in the user's chosen target order",
                        "missing_cleanup": (
                            'Drop products whose product/aisle/department metadata is literally "missing", '
                            "plus the associated prior/train order-product rows."
                        ),
                        "leakage_note": (
                            "This bundle contains only globally safe base tables. "
                            "Train-only learned features must be rebuilt inside the experiment notebook."
                        ),
                    },
                }
                (output_dir / "bundle_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            """
        ),
        code_cell(
            """
            raw_data_dir = resolve_raw_data_dir(DATA_DIR, DATASET_NAME)
            verify_required_files(raw_data_dir)
            print("Resolved raw data dir:", raw_data_dir)
            """
        ),
        code_cell(
            """
            tables = read_raw_tables(raw_data_dir)
            print({name: frame.shape for name, frame in tables.items()})
            """
        ),
        code_cell(
            """
            bundle = build_base_bundle(**tables)
            summary_df = pd.DataFrame(
                [{"table": name, "rows": frame.shape[0], "cols": frame.shape[1]} for name, frame in bundle.items()]
            ).sort_values("table").reset_index(drop=True)
            display(summary_df)
            """
        ),
        code_cell(
            """
            write_bundle(bundle, OUTPUT_DIR, raw_data_dir)

            summary = {
                "raw_data_dir": str(raw_data_dir),
                "output_dir": str(OUTPUT_DIR),
                "tables": {name: frame.shape for name, frame in bundle.items()},
            }
            print(json.dumps(summary, indent=2, default=str))
            """
        ),
        md_cell(
            """
            ## Output

            After the last cell finishes, the base bundle is written to:

            - `Final version/artifacts/base_bundle/`

            Then you can open:

            - [final_experiments.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/Final%20version/final_experiments.ipynb)
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), indent=2), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
