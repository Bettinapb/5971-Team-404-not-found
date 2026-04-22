from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path("/")
NOTEBOOK_PATH = ROOT / "redraw_output_figures.ipynb"


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        md_cell(
            """
            # Redraw Output Figures

            This notebook reads the latest artifacts in `outputs/` and redraws:

            - `main_validation_vs_test_auc_clean.png`
            - `segment_analysis_bars_clean.png`

            with legends moved outside the plotting area.
            """
        ),
        code_cell(
            """
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.metrics import f1_score, roc_auc_score
            """
        ),
        code_cell(
            """
            PROJECT_ROOT = Path.cwd().resolve()
            OUTPUT_DIR = PROJECT_ROOT / "outputs" if (PROJECT_ROOT / "outputs").exists() else PROJECT_ROOT

            print({
                "PROJECT_ROOT": str(PROJECT_ROOT),
                "OUTPUT_DIR": str(OUTPUT_DIR),
            })
            """
        ),
        code_cell(
            """
            main_summary = pd.read_csv(OUTPUT_DIR / "main_model_summary.csv")
            main_predictions = pd.read_parquet(OUTPUT_DIR / "main_predictions.parquet")

            display(main_summary.head())
            display(main_predictions.head())
            """
        ),
        code_cell(
            """
            def make_k_map(df_input: pd.DataFrame, strategy: str = "rounded_avg_basket", fixed_k: int = 11) -> dict[int, int]:
                user_k = df_input.groupby("user_id")["u_avg_basket_size"].first().reset_index()
                if strategy == "fixed_11":
                    user_k["pred_k"] = fixed_k
                elif strategy == "rounded_avg_basket":
                    user_k["pred_k"] = np.rint(user_k["u_avg_basket_size"]).astype(int).clip(lower=1)
                elif strategy == "ceil_avg_basket":
                    user_k["pred_k"] = np.ceil(user_k["u_avg_basket_size"]).astype(int).clip(lower=1)
                else:
                    raise ValueError(strategy)
                return dict(zip(user_k["user_id"], user_k["pred_k"]))


            def compute_segment_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
                rows = []
                for model_name, model_df in pred_df.groupby("model"):
                    for split_name, split_df in model_df.groupby("split"):
                        dynamic_map = make_k_map(split_df, "rounded_avg_basket")
                        ranked = split_df.sort_values(
                            ["user_id", "prob", "product_id"],
                            ascending=[True, False, True],
                        ).copy()
                        ranked["rank_within_user"] = ranked.groupby("user_id").cumcount() + 1
                        ranked["pred_k_dynamic"] = ranked["user_id"].map(dynamic_map).astype(int)
                        ranked["pred_top_dynamic"] = (
                            ranked["rank_within_user"] <= ranked["pred_k_dynamic"]
                        ).astype(int)

                        for segment_value, g in ranked.groupby("user_cluster"):
                            per_user_f1_dyn = []
                            for _, gu in g.groupby("user_id"):
                                y_true = gu["label"].to_numpy()
                                pred_dyn = gu["pred_top_dynamic"].to_numpy()
                                per_user_f1_dyn.append(f1_score(y_true, pred_dyn, zero_division=0))

                            rows.append({
                                "model": model_name,
                                "split": split_name,
                                "segment": int(segment_value),
                                "users": int(g["user_id"].nunique()),
                                "auc": float(roc_auc_score(g["label"].to_numpy(), g["prob"].to_numpy())),
                                "f1@dynamic_k": float(np.mean(per_user_f1_dyn)),
                            })
                return pd.DataFrame(rows).sort_values(["split", "model", "segment"]).reset_index(drop=True)


            segment_summary = compute_segment_summary(main_predictions)
            segment_summary.to_csv(OUTPUT_DIR / "segment_summary_rebuilt.csv", index=False)
            display(segment_summary)
            """
        ),
        code_cell(
            """
            plot_df = (
                main_summary.pivot(index="model", columns="split", values="auc")
                .reset_index()
                .dropna(subset=["validation", "test"], how="any")
            )

            model_order = plot_df["model"].tolist()
            x = np.arange(len(model_order))
            ymin = max(0.0, plot_df[["validation", "test"]].min().min() - 0.01)
            ymax = min(1.0, plot_df[["validation", "test"]].max().max() + 0.01)

            fig, ax = plt.subplots(figsize=(11, 5.5))
            ax.plot(x, plot_df["validation"], marker="o", linewidth=2.2, markersize=8, label="Validation AUC")
            ax.plot(x, plot_df["test"], marker="o", linewidth=2.2, markersize=8, label="Test AUC")

            for xi, v_val, v_test in zip(x, plot_df["validation"], plot_df["test"]):
                ax.vlines(xi, ymin=min(v_val, v_test), ymax=max(v_val, v_test), color="gray", alpha=0.35, linewidth=1.5)

            ax.set_xticks(x)
            ax.set_xticklabels(model_order, rotation=20)
            ax.set_ylabel("AUC")
            ax.set_title("Validation vs test AUC by model")
            ax.set_ylim(ymin, ymax)
            ax.grid(alpha=0.25)
            ax.legend(loc="center left", bbox_to_anchor=(0.90, 0.5), frameon=True)
            fig.tight_layout(rect=[0, 0, 0.88, 1])
            fig.savefig(OUTPUT_DIR / "main_validation_vs_test_auc_clean.png", dpi=180, bbox_inches="tight")
            plt.show()
            """
        ),
        code_cell(
            """
            segment_plot_df = segment_summary[segment_summary["split"] == "test"].copy()
            model_order = list(segment_plot_df["model"].drop_duplicates())
            segments = sorted(segment_plot_df["segment"].unique())
            fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))

            for model_name in model_order:
                g = (
                    segment_plot_df[segment_plot_df["model"] == model_name]
                    .set_index("segment")
                    .reindex(segments)
                    .reset_index()
                )
                axes[0].plot(g["segment"], g["auc"], marker="o", linewidth=2.2, markersize=7, label=model_name)
                axes[1].plot(g["segment"], g["f1@dynamic_k"], marker="o", linewidth=2.2, markersize=7, label=model_name)

            axes[0].set_title("Segment analysis on test: AUC by user cluster")
            axes[0].set_ylabel("AUC")
            axes[1].set_title("Segment analysis on test: F1@dynamic-k by user cluster")
            axes[1].set_ylabel("F1@dynamic-k")

            auc_min = max(0.0, segment_plot_df["auc"].min() - 0.015)
            auc_max = min(1.0, segment_plot_df["auc"].max() + 0.015)
            f1_min = max(0.0, segment_plot_df["f1@dynamic_k"].min() - 0.02)
            f1_max = min(1.0, segment_plot_df["f1@dynamic_k"].max() + 0.02)
            axes[0].set_ylim(auc_min, auc_max)
            axes[1].set_ylim(f1_min, f1_max)

            for ax in axes:
                ax.set_xticks(segments)
                ax.set_xticklabels([str(s) for s in segments])
                ax.set_xlabel("User cluster")
                ax.grid(alpha=0.25)

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.87, 0.5), frameon=True)
            fig.tight_layout(rect=[0, 0, 0.90, 1])
            fig.savefig(OUTPUT_DIR / "segment_analysis_bars_clean.png", dpi=180, bbox_inches="tight")
            plt.show()
            """
        ),
        md_cell(
            """
            Cleaned figures are saved to:

            - `outputs/main_validation_vs_test_auc_clean.png`
            - `outputs/segment_analysis_bars_clean.png`
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
