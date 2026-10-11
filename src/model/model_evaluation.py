import os
import sys
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")       # draw plots without a window
import matplotlib.pyplot as plt
import mlflow
import dagshub
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from src.logger import logging
from src.exception import CustomException
from src.constants import LABEL_COLS, TARGET_COL
from src.model.scorers import consensus_score

# Load environment variables from .env file
load_dotenv()

# Below code block is for Production use
# ----------------------------------------------------------------------------------------------
# Set up DagsHub credentials for Mlflow tracking
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "AmitKr-06"
repo_name = "Intelligent-anomaly-discovery"

# Setup Mlflow tracking URI
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# Below code block is for local use
# dagshub.init(repo_owner='AmitKr-06', repo_name='Intelligent-anomaly-discovery', mlflow=True)

def load_params(params_path: str) -> dict:
    """Load parameters from the YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.debug("Parameters retrieved from %s", params_path)
        return params
    except Exception as e:
        logging.error("Failed to load parameter from %s: %s", params_path, e)
        raise CustomException(e, sys)

def load_models(model_dir: str) -> dict:
    """Load the three fitted models made by model building."""
    try:
        models = {name: joblib.load(os.path.join(model_dir, f"{name}.joblib"))
                  for name in ["iforest", "dbscan", "gmm"]}
        logging.info("Loaded models: %s", list(models))
        return models
    except Exception as e:
        logging.error("Failed to load models: %s", e)
        raise CustomException(e, sys)

def load_test(processed_dir: str):
    """Load the processed test features and the test labels."""
    try:
        X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))
        logging.info("Loaded test data: X= %s, y= %s", X_test.shape, y_test.shape)
        return X_test, y_test
    except Exception as e:
        logging.error("Failed to load test data: %s", e)
        raise CustomException(e, sys)

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
        logging.error("Failed to score the test data: %s", e)
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
    """Precisionn and recall if an analyst reviews only the top X% of rows."""
    try:
        metrics = {}
        for name, s in scores.items():
            for b in budgets:
                k = int(len(s) * b)
                top = np.argsort(s)[-k:]     # the k most suspicious rows
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
            for col in [c for c in LABEL_COLS if c != TARGET_COL]:         # TWF, HDF, PWF, OSF, RNF
                y_col = y_test[col].values
                if y_col.sum() > 0:
                    metrics[f"consensus_recall_{col}_top{round(b * 100)}pct"] = float(y_col[top].sum() / y_col.sum())
        return metrics
    except Exception as e:
        logging.error("Failed to compute per-type recall: %s", e)
        raise CustomException(e, sys)

def bootstrap_ci(y_true: np.ndarray, scores: dict, n_boot: int, seed: int) -> dict:
    """95% confidence intervals by resampling the test rows. The same resample is used
    for evry method, so the differences between methods are paired."""
    try:
        rng = np.random.default_rng(seed)
        n = len(y_true)
        aucs = {k: [] for k in scores}
        aps = {k: [] for k in scores}
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if y_true[idx].sum() == 0:                          # skip a resample with no failures
                continue
            for name, s in scores.items():
                aucs[name].append(roc_auc_score(y_true[idx], s[idx]))
                aps[name].append(average_precision_score(y_true[idx], s[idx]))
        ci = {}
        for name in scores:
            ci[f"{name}_roc_auc_ci_low"], ci[f"{name}_roc_auc_ci_high"] = np.percentile(aucs[name], [2.5, 97.5])
            ci[f"{name}_ap_ci_low"], ci[f"{name}_ap_ci_high"] = np.percentile(aps[name], [2.5, 97.5])

        # Does the consensus rally beat Isolation Forest ? If the interval contains 0, we can't claim it.
        diff = np.array(aps["consensus"]) - np.array(aps["iforest"])
        ci["ap_diff_consensus_minus_iforest_mean"] = float(diff.mean())
        ci["ap_diff_consensus_minus_iforest_ci_low"], ci["ap_diff_consensus_minus_iforest_ci_high"] = np.percentile(diff, [2.5, 97.5])
        return {k: float(v) for k, v in ci.items()}
    except Exception as e:
        logging.error("Failed to compute bootstrap intervals: %s", e)
        raise CustomException(e, sys)

def save_pr_curve(y_true: np.ndarray, scores: dict, file_path: str) -> None:
    """Precision-recall curve for every method."""
    try:
        plt.figure(figsize=(7, 5))
        for name, s in scores.items():
            p, r, _ = precision_recall_curve(y_true, s)
            plt.plot(r, p, label=f"{name} (AP={average_precision_score(y_true, s):.3f})")
        plt.axhline(y_true.mean(), color="grey", linestyle="--", label="random")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-recall on the test set")
        plt.legend()
        plt.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close()
        logging.debug("PR curve saved to %s", file_path)
    except Exception as e:
        logging.error("Failed to save the PR curve: %s", e)
        raise CustomException(e, sys)

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logging.info("Metrics saved to %s", file_path)
    except Exception as e:
        logging.error("Failed to save metrics: %s", e)
        raise CustomException(e, sys)

def flatten_params(d: dict, prefix: str = "") -> dict:
    """{'gmm': {'n_init': 5}} -> {'gmm.n_init': 5}, the format MLflow expects."""
    flat = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_params(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat

def main():
    mlflow.set_experiment("anomaly-discovery")
    with mlflow.start_run(run_name="iforest+dbscan+gmm-baseline") as run:                      # Start an MLflow run
        try:
            params = load_params("params.yaml")
            eval_params = params["model_evaluation"]
            budgets = eval_params["review_budgets"]

            # 1. Load models and test data, then score the test set
            models = load_models("models")
            X_test, y_test = load_test("data/processed")
            y_true = y_test[TARGET_COL].values
            scores = score_all(models, X_test)

            # 2. Compute metrics (labels are used here and nowhere else)
            metrics = {}
            metrics.update(ranking_metrics(y_true, scores))
            metrics.update(review_metrics(y_true, scores, budgets))
            metrics.update(per_type_recall(y_test, scores["consensus"], budgets))
            metrics.update(bootstrap_ci(y_true, scores, eval_params["n_bootstrap"], eval_params["random_state"]))
            metrics["random_baseline_ap"] = float(y_true.mean())
            metrics["n_test_failures"] = int(y_true.sum())

            # 3. Save locally
            os.makedirs("reports", exist_ok=True)
            save_metrics(metrics, "reports/metrics.json")
            save_pr_curve(y_true, scores, "reports/pr_curve.png")

            # 4. Log to MLflow: parameters, metrics, plot and model files
            mlflow.log_params(flatten_params(params["model_building"]))
            mlflow.log_params(flatten_params(params["data_ingestion"], prefix="data."))
            mlflow.log_metrics(metrics)
            mlflow.log_artifact("reports/metrics.json")
            mlflow.log_artifact("reports/pr_curve.png")
            for file in ["iforest", "dbscan", "gmm", "preprocessor"]:
                mlflow.log_artifact(f"models/{file}.joblib", artifact_path="models")

            # 5. Save the run id for later stages
            with open("reports/experiment_info.json", "w") as f:
                json.dump({"run_id": run.info.run_id}, f, indent=4)

            logging.info("Model evaluation completed, MLflow run id= %s", run.info.run_id)

        except Exception as e:
            logging.error("Model evaluation failed: %s", e)
            raise                                         # raise, so DVC marks the stage as failed


if __name__ == "__main__":
    main()