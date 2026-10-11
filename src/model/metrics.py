import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from src.logger import logging
from src.exception import CustomException
from src.constants import LABEL_COLS, TARGET_COL
from src.model.scorers import consensus_score


def score_all(models: dict, X: pd.DataFrame) -> dict:
    """Anomaly score from every model (higher = more unusual), plus the consensus."""
    try:
        scores = {
            "iforest": -models["iforest"].score_samples(X),   # score_samples is negative for anomalies
            "dbscan": models["dbscan"].score(X),              # distance to the nearest core point
            "gmm": -models["gmm"].score_samples(X),           # negative log-likelihood
        }
        scores["consensus"] = consensus_score(list(scores.values()))
        return scores
    except Exception as e:
        logging.error("Failed to score the data: %s", e)
        raise CustomException(e, sys)


def ranking_metrics(y_true: np.ndarray, scores: dict) -> dict:
    """ROC-AUC and Average Precision for every score."""
    try:
        metrics = {}
        for name, s in scores.items():
            metrics[f"{name}_roc_auc"] = roc_auc_score(y_true, s)
            metrics[f"{name}_ap"] = average_precision_score(y_true, s)
        return metrics
    except Exception as e:
        logging.error("Failed to compute ranking metrics: %s", e)
        raise CustomException(e, sys)


def review_metrics(y_true: np.ndarray, scores: dict, budgets: list) -> dict:
    """Precision and recall if an analyst reviews only the top X% of rows."""
    try:
        metrics = {}
        for name, s in scores.items():
            for b in budgets:
                k = int(len(s) * b)
                top = np.argsort(s)[-k:]                      # the k most suspicious rows
                pct = round(b * 100)
                metrics[f"{name}_precision_top{pct}pct"] = float(y_true[top].mean())
                metrics[f"{name}_recall_top{pct}pct"] = float(y_true[top].sum() / y_true.sum())
        return metrics
    except Exception as e:
        logging.error("Failed to compute review metrics: %s", e)
        raise CustomException(e, sys)


def per_type_recall(y_test: pd.DataFrame, consensus: np.ndarray, budgets: list) -> dict:
    """Recall of each failure type among the top X% of the consensus score."""
    try:
        metrics = {}
        for b in budgets:
            k = int(len(consensus) * b)
            top = np.argsort(consensus)[-k:]
            for col in [c for c in LABEL_COLS if c != TARGET_COL]:     # TWF, HDF, PWF, OSF, RNF
                y_col = y_test[col].values
                if y_col.sum() > 0:
                    metrics[f"consensus_recall_{col}_top{round(b * 100)}pct"] = float(y_col[top].sum() / y_col.sum())
        return metrics
    except Exception as e:
        logging.error("Failed to compute per-type recall: %s", e)
        raise CustomException(e, sys)


def bootstrap_ci(y_true: np.ndarray, scores: dict, n_boot: int, seed: int) -> dict:
    """95% confidence intervals by resampling the test rows. The same resample is used
    for every method, so the differences between methods are paired."""
    try:
        rng = np.random.default_rng(seed)
        n = len(y_true)
        aucs = {k: [] for k in scores}
        aps = {k: [] for k in scores}
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if y_true[idx].sum() == 0:                        # skip a resample with no failures
                continue
            for name, s in scores.items():
                aucs[name].append(roc_auc_score(y_true[idx], s[idx]))
                aps[name].append(average_precision_score(y_true[idx], s[idx]))

        ci = {}
        for name in scores:
            ci[f"{name}_roc_auc_ci_low"], ci[f"{name}_roc_auc_ci_high"] = np.percentile(aucs[name], [2.5, 97.5])
            ci[f"{name}_ap_ci_low"], ci[f"{name}_ap_ci_high"] = np.percentile(aps[name], [2.5, 97.5])

        # Does the consensus really beat Isolation Forest? If the interval contains 0, we can't claim it.
        diff = np.array(aps["consensus"]) - np.array(aps["iforest"])
        ci["ap_diff_consensus_minus_iforest_mean"] = float(diff.mean())
        ci["ap_diff_consensus_minus_iforest_ci_low"], ci["ap_diff_consensus_minus_iforest_ci_high"] = np.percentile(diff, [2.5, 97.5])
        return {k: float(v) for k, v in ci.items()}
    except Exception as e:
        logging.error("Failed to compute bootstrap intervals: %s", e)
        raise CustomException(e, sys)