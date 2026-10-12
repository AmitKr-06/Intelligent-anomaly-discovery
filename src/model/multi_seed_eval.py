import os
import sys
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.constants import TARGET_COL
from src.features.data_preprocessing import split_x_y, add_features, build_preprocessor
from src.model.model_building import (
    load_params, train_isolation_forest, train_dbscan, train_gmm,
)
from src.model.metrics import score_all, ranking_metrics, review_metrics
from src.model.scorers import consensus_score
from src.pipeline.predict_pipeline import flag_top      # same flagging logic as the app

METHODS = ["consensus", "iforest", "dbscan", "gmm"]

# Consensus variants, each with one method left out
ABLATIONS = {
    "cons_no_gmm": ["iforest", "dbscan"],
    "cons_no_dbscan": ["iforest", "gmm"],
    "cons_no_iforest": ["dbscan", "gmm"],
}


def agreement_precision(y_true: np.ndarray, scores: dict, review_pct: float) -> dict:
    """Among the flagged rows, precision of rows where all three methods agree vs the rest."""
    _, order, agree, _ = flag_top(scores, review_pct)
    y_top, a_top = y_true[order], agree[order]
    all3 = a_top == 3
    return {
        "top_precision": float(y_top.mean()),
        "agree3_n": int(all3.sum()),
        "agree3_precision": float(y_top[all3].mean()) if all3.any() else np.nan,
        "other_n": int((~all3).sum()),
        "other_precision": float(y_top[~all3].mean()) if (~all3).any() else np.nan,
    }


def run_one_seed(df: pd.DataFrame, seed: int, params: dict) -> dict:
    """Full pipeline for one seed: split, preprocess, fit, score, measure.
    The seed changes both the split and the model randomness."""
    try:
        # 1. Stratified split (the label is used only to split)
        train_df, test_df = train_test_split(
            df, test_size=params["data_ingestion"]["test_size"],
            random_state=seed, stratify=df[TARGET_COL],
        )

        # 2. Features and preprocessing, fit on TRAIN only
        X_train, _ = split_x_y(train_df)
        X_test, y_test = split_x_y(test_df)
        X_train, X_test = add_features(X_train), add_features(X_test)
        pre = build_preprocessor()
        X_train_t = pre.fit_transform(X_train)
        X_test_t = pre.transform(X_test)

        # 3. Fit the three models on the training features only
        mb = params["model_building"]
        models = {
            "iforest": train_isolation_forest(X_train_t, mb["isolation_forest"], seed),
            "dbscan": train_dbscan(X_train_t, mb["dbscan"]),
            "gmm": train_gmm(X_train_t, mb["gmm"], seed),
        }

        # 4. Score the test rows, including the consensus variants
        scores = score_all(models, X_test_t)
        for name, members in ABLATIONS.items():
            scores[name] = consensus_score([scores[m] for m in members])

        # 5. Measure (labels are used here and nowhere else)
        y_true = y_test[TARGET_COL].values
        row = {"seed": seed, "n_test_failures": int(y_true.sum())}
        row.update(ranking_metrics(y_true, scores))
        row.update(review_metrics(y_true, scores, params["model_evaluation"]["review_budgets"]))
        row.update(agreement_precision(y_true, scores, params["app"]["review_pct_default"]))
        logging.info("Seed %s done: consensus AP= %.3f, iforest AP= %.3f",
                     seed, row["consensus_ap"], row["iforest_ap"])
        return row
    except Exception as e:
        logging.error("Failed on seed %s: %s", seed, e)
        raise CustomException(e, sys)


def summarize(results: pd.DataFrame) -> dict:
    """Mean and standard deviation of every metric over the seeds."""
    try:
        summary = {}
        for col in [c for c in results.columns if c not in ("seed", "n_test_failures")]:
            summary[f"{col}_mean"] = float(results[col].mean())
            summary[f"{col}_std"] = float(results[col].std(ddof=1))
        return summary
    except Exception as e:
        logging.error("Failed to summarize results: %s", e)
        raise CustomException(e, sys)


def paired_differences(results: pd.DataFrame) -> dict:
    """AP difference for every pair of methods, seed by seed (paired), plus how many
    seeds each method wins."""
    try:
        out = {}
        for a, b in itertools.combinations(METHODS, 2):
            diff = results[f"{a}_ap"] - results[f"{b}_ap"]
            out[f"ap_diff_{a}_minus_{b}_mean"] = float(diff.mean())
            out[f"ap_diff_{a}_minus_{b}_std"] = float(diff.std(ddof=1))
            out[f"ap_diff_{a}_minus_{b}_seeds_won"] = int((diff > 0).sum())
        return out
    except Exception as e:
        logging.error("Failed to compute paired differences: %s", e)
        raise CustomException(e, sys)


def ablation_differences(results: pd.DataFrame) -> dict:
    """AP of the three-method consensus minus each two-method variant, seed by seed.
    A positive difference means the third method helps."""
    try:
        out = {}
        for v in ABLATIONS:
            diff = results["consensus_ap"] - results[f"{v}_ap"]
            out[f"ap_diff_consensus_minus_{v}_mean"] = float(diff.mean())
            out[f"ap_diff_consensus_minus_{v}_std"] = float(diff.std(ddof=1))
            out[f"ap_diff_consensus_minus_{v}_seeds_won"] = int((diff > 0).sum())
        return out
    except Exception as e:
        logging.error("Failed to compute ablation differences: %s", e)
        raise CustomException(e, sys)


def agreement_gap(results: pd.DataFrame) -> dict:
    """Precision of rows where all three methods agree minus precision of the other flagged rows."""
    try:
        gap = results["agree3_precision"] - results["other_precision"]
        return {
            "agree3_minus_other_precision_mean": float(gap.mean()),
            "agree3_minus_other_precision_std": float(gap.std(ddof=1)),
            "agree3_minus_other_precision_seeds_higher": int((gap > 0).sum()),
        }
    except Exception as e:
        logging.error("Failed to compute the agreement gap: %s", e)
        raise CustomException(e, sys)


def main():
    try:
        params = load_params("params.yaml")
        ms = params["multi_seed_eval"]
        seeds = list(range(ms["base_seed"], ms["base_seed"] + ms["n_seeds"]))

        df = pd.read_csv(params["data_ingestion"]["raw_path"])
        logging.info("Running %s seeds: %s", len(seeds), seeds)

        results = pd.DataFrame([run_one_seed(df, s, params) for s in seeds])

        summary = {"n_seeds": len(seeds)}
        summary.update(summarize(results))
        summary.update(paired_differences(results))
        summary.update(ablation_differences(results))
        summary.update(agreement_gap(results))

        os.makedirs("reports", exist_ok=True)
        results.to_csv("reports/multi_seed_per_seed.csv", index=False)
        with open("reports/multi_seed_metrics.json", "w") as f:
            json.dump(summary, f, indent=4)
        logging.info("Multi-seed evaluation completed")

    except Exception as e:
        logging.error("Multi-seed evaluation failed: %s", e)
        raise


if __name__ == "__main__":
    main()