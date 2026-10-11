import os
import sys
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

from src.logger import logging
from src.exception import CustomException
from src.model.scorers import DBSCANScorer


def load_params(params_path: str) -> dict:
    """Load parameters from the YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.debug("Parameters retrieved from %s", params_path)
        return params
    except Exception as e:
        logging.error("Failed to load parameters from %s: %s", params_path, e)
        raise CustomException(e, sys)

def load_train(processed_dir: str) -> pd.DataFrame:
    """Load the processed training features (no labels here)."""
    try:
        X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
        logging.info("Loaded X_train, shape= %s", X_train.shape)
        return X_train
    except Exception as e:
        logging.error("Failed to load X_train: %s", e)
        raise CustomException(e, sys)

def train_isolation_forest(X: pd.DataFrame, params: dict, random_state: int):
    """Isolation FOrest: rows that are easy to isolate are unusual."""
    try:
        model = IsolationForest(
            n_estimators=params["n_estimators"],
            random_state=random_state,
            n_jobs=-1,
        ).fit(X)
        logging.info("Isolation Forest trained")
        return model
    except Exception as e:
        logging.error("Failed to train Isolation Forest: %s", e)
        raise CustomException(e, sys)

def train_dbscan(X: pd.DataFrame, params: dict):
    """DBSCAN: distance to the nearest dense (core) point."""
    try:
        model = DBSCANScorer(eps = params["eps"], min_samples = params["min_samples"]).fit(X)
        logging.info("DBSCAN trained: clusters= %s, train noise= %.1f%%", model.n_clusters_, model.train_noise_rate_ * 100)
        return model
    except Exception as e:
        logging.error("Failed to train DBSCAN: %s", e)
        raise CustomException(e, sys)

def train_gmm(X: pd.DataFrame, params: dict, random_state: int):
    """Gaussian Mixture: low likelihood means an unusual row."""
    try:
        model = GaussianMixture(
            n_components= params["n_components"],
            covariance_type= params["covariance_type"],
            n_init= params["n_init"],
            random_state=random_state,
        ).fit(X)
        logging.info("GMM trained: converged= %s, BIC= %.0f", model.converged_, model.bic(X))
        return model
    except Exception as e:
        logging.error("Failed to train GMM: %s", e)
        raise CustomException(e, sys)

def save_models(models: dict, model_dir: str) -> None:
    """Save each fitted model with joblib."""
    try:
        os.makedirs(model_dir, exist_ok=True)
        for name, model in models.items():
            joblib.dump(model, os.path.join(model_dir, f"{name}.joblib"))
        logging.debug("Saved models %s to %s", list(models), model_dir)
    except Exception as e:
        logging.error("Failed to save models: %s", e)
        raise CustomException(e, sys)

def main():
    try:
        # 1. Read settings and data
        params = load_params("params.yaml")["model_building"]
        X_train = load_train("data/processed")
        seed = params["random_state"]

        # 2. Train the three models on the training features only
        models = {
            "iforest": train_isolation_forest(X_train, params["isolation_forest"], seed),
            "dbscan": train_dbscan(X_train, params["dbscan"]),
            "gmm": train_gmm(X_train, params["gmm"], seed),
        }

        # 3. Save
        save_models(models, "models")
        logging.info("Model building completed")

    except Exception as e:
        logging.error("Model building failed: %s", e)
        raise

if __name__ == "__main__":
    main()