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
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.model.anomaly_model import log_model
from src.logger import logging
from src.exception import CustomException
from src.constants import TARGET_COL
from src.model.metrics import (
    score_all, ranking_metrics, review_metrics, per_type_recall, bootstrap_ci)

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

            # 5. Log the model itself (code + settings), so the registry step can register it
            model_info = log_model()

            # 6. Save the run id and the model location for the registry step
            with open("reports/experiment_info.json", "w") as f:
                json.dump({"run_id": run.info.run_id, "model_uri": model_info.model_uri}, f, indent=4)

            logging.info("Model evaluation completed, MLflow run id= %s", run.info.run_id)

        except Exception as e:
            logging.error("Model evaluation failed: %s", e)
            raise                                         # raise, so DVC marks the stage as failed


if __name__ == "__main__":
    main()