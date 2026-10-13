import os
import sys
import json
import argparse
import subprocess
from dotenv import load_dotenv
import mlflow
from mlflow.exceptions import MlflowException

from src.logger import logging
from src.exception import CustomException
from src.model.model_building import load_params


def connect_to_dagshub() -> None:
    """Point Mlflow at DagsHub. Called from main(), so importing this file needs no token."""
    load_dotenv()

    # Belog code block is for Production use
    # -----------------------------------------------------------------------------------------------
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "AmitKr-06"
    repo_name = "Intelligent-anomaly-discovery"
    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    # Below code block is for local use
    # import dagshub
    # dagshub.init(repo_owner="AmitKr-06", repo_name="Intelligent-anomaly-discovery", mlflow=True)

def load_json(file_path: str) -> dict:
    """Load a JSON file (model info or metrics)."""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        logging.debug("Loaded %s", file_path)
        return data
    except Exception as e:
        logging.error("Failed to load %s: %s", file_path, e)
        raise CustomException(e, sys)


def check_gate(metrics: dict, gate: dict) -> list:
    """Quality gate: returns a list of reasons why the model must NOT be registered.
    An empty list means it passed. It stops a worse method (for example after a parameter
    change) from becoming the 'staging' version. """
    problems = []
    ap = metrics["consensus_ap_mean"]
    if ap < gate["min_consensus_ap_mean"]:
        problems.append(f"consensus AP {ap:.3f} is below the minium {gate['min_consensus_ap_mean']}")

    n_seeds = int(metrics["n_seeds"])
    for method in ["iforest", "dbscan", "gmm"]:
        won = int(metrics[f"ap_diff_consensus_minus_{method}_seeds_won"])
        if won < gate["min_seeds_won"]:
            problems.append(f"The consesus beats {method} on only {won} of {n_seeds} splits; "
                            f"minimum required: {gate['min_seeds_won']}")
    return problems


def git_commit() -> str:
    """Current git commit, so a registry version can be traced back to the code."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception :
        return "unknown"

def verify_loadable(model_name: str, version: str) -> None:
    """Load the registered version once. A version that cannot be loaded must never become 'staging'."""
    try:
        mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
    except Exception as e:
        raise RuntimeError(f"{model_name} version {version} cannot be loaded: {e}") from e

def register_model(model_name: str, model_info: dict, metrics: dict) -> str:
    """Register the logged model as a new version and give it the 'staging' alias."""
    try:
        client = mlflow.tracking.MlflowClient()

        # Create the registered model once; and other error is a real error
        try:
            client.create_registered_model(model_name)
        except MlflowException as e:
            if e.error_code != "RESOURCE_ALREADY_EXISTS":
                raise

        # In Mlflow 3 the source must be the logged model itself (models:/m-.......).
        # A runs:/<run_id>/model path is accepted by the registry but the version cannot be loaded.
        version = str(client.create_model_version(
            name=model_name, source=model_info["model_uri"], run_id=model_info["run_id"],
            description="Isolation Forest + DBSCAN + Gaussian Mixture, rank-averaged consensus. "
                        "Models are fitted on the batch passed to predict.",
        ).version)
        try:
            verify_loadable(model_name, version)
        except Exception:
            try:
                client.delete_model_version(model_name, version)
            except Exception as cleanup_error:
                logging.warning("Could not delete the broken version %s: %s", version, cleanup_error)
                raise
        client.set_registered_model_alias(model_name, "staging", version)

        # Evidence and traceability. A tagging problem must not undo the registration.
        tags = {
            "consensus_ap_mean": f"{metrics['consensus_ap_mean']:.3f}",
            "consensus_ap_std": f"{metrics['consensus_ap_std']:.3f}",
            "consensus_roc_auc_mean": f"{metrics['consensus_roc_auc_mean']:.3f}",
            "n_seeds": str(int(metrics["n_seeds"])),
            "git_commit": git_commit(),
        }
        for key, value in tags.items():
            try:
                client.set_model_version_tag(model_name, version, key, value)
            except Exception as e:
                logging.warning("Could not set tag %s: %s", key, e)

        logging.info("Registered %s version %s as 'staging'", model_name, version)
        return version
    except Exception as e:
        logging.error("Error during model registration: %s", e)
        raise CustomException(e, sys)


def promote_to_production(model_name: str) -> str:
    """Move the 'production' alias to the version that is currently 'staging'."""
    try:
        client = mlflow.tracking.MlflowClient()
        version = client.get_model_version_by_alias(model_name, "staging").version
        client.set_registered_model_alias(model_name, "production", version)
        logging.info("%s version %s is now 'production'", model_name, version)
        return str(version)
    except Exception as e:
        logging.error("Error during promotion: %s", e)
        raise CustomException(e, sys)

def main():
    parser = argparse.ArgumentParser(description="Register the evaluated model, or promote it.")
    parser.add_argument("--promote", action="store_true", help="move the 'production' alias to the current 'staging' version")
    args = parser.parse_args()

    try:
        params = load_params("params.yaml")["registry"]
        connect_to_dagshub()

        if args.promote:
            version = promote_to_production(params["model_name"])
            print(f"{params['model_name']} version {version} is now 'productio'")
            return

        model_info = load_json("reports/experiment_info.json")
        if "model_uri" not in model_info:
            raise SystemExit("reports/experiment_info.json has no model_uri. "
                             "Run 'dvc repro' so the evaluation stage logs the model first.")
        metrics = load_json("reports/multi_seed_metrics.json")

        problems = check_gate(metrics, params)
        if problems:
            for p in problems:
                logging.error("Quality gate failed: %s", p)
            raise SystemExit("Not registered, the quality gate failed:\n - " + "\n - ".join(problems))

        version = register_model(params["model_name"], model_info, metrics)
        os.makedirs("reports", exist_ok=True)
        with open("reports/registered_model.json", "w") as f:
            json.dump({"name":params["model_name"], "version": version, "run_id":model_info["run_id"]}, f, indent=4)
        print(f"{params['model_name']} version {version} registered as 'staging'")

    except SystemExit:
        raise
    except Exception as e:
        logging.error("Failed to complete the model registration process: %s", e)
        raise


if __name__ == "__main__":
    main()