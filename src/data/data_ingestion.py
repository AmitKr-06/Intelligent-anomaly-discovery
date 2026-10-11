import os
import sys
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.constants import TARGET_COL, TRAIN_REQUIRED_COLS

def load_params(params_path: str) -> dict:
    """Load parameters from the YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.debug("Parameters retreived from %s", params_path)
        return params
    except Exception as e:
        logging.error("Failed to load parameters from %s: %s", params_path, e)
        raise CustomException(e, sys)

def load_data(data_path: str) -> pd.DataFrame:
    """Load the raw CSV file."""
    try:
        df = pd.read_csv(data_path)
        logging.info("Data loaded from %s, shape= %s", data_path, df.shape)
        return df
    except Exception as e:
        logging.error("Failed to laod data from %s: %s", data_path, e)
        raise CustomException(e, sys)

def validate_data(df: pd.DataFrame) -> None:
    """Check that the columns the pipeline needs are present."""
    try:
        missing = [c for c in TRAIN_REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        logging.info("Data validation passed")
    except Exception as e:
        logging.error("Data validation failed: %s", e)
        raise CustomException(e, sys)

def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    """Split into train and test. Stratify on the failure label so both
    sets keep the same failure rate (about 3.4%). The label is used here 
    only to split, never to train a model."""
    try:
        train_df, test_df = train_test_split(
            df, test_size=test_size,
            random_state=random_state,
            stratify=df[TARGET_COL],
        )
        logging.info("Split done: train = %s, test = %s", train_df.shape, test_df.shape)
        return train_df, test_df
    except Exception as e:
        logging.error("Failed to split the data: %s", e)
        raise CustomException(e, sys)

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_dir: str) -> None:
    """Save train and test files to data/interim (never overwrite data/raw)."""
    try:
        interim_path = os.path.join(data_dir, "interim")
        os.makedirs(interim_path, exist_ok= True)
        train_df.to_csv(os.path.join(interim_path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(interim_path, "test.csv"), index=False)
        logging.debug("Train and test data saved to %s", interim_path)
    except Exception as e:
        logging.error("Failed to save the data: %s", e)
        raise CustomException(e, sys)

def main():
    try:
        # 1. Read settings
        params = load_params("params.yaml")["data_ingestion"]

        # Load and check the raw data
        df = load_data(params["raw_path"])
        validate_data(df)
        # Split (the same random_state always gives the same split)
        train_df, test_df = split_data(df, params["test_size"], params["random_state"])

        # Save
        save_data(train_df, test_df, data_dir="data")
        logging.info("Data Ingestion Completed")
    except Exception as e:
        logging.error("Data Ingestion Failed: %s", e)
        raise   # re-raise so DVC marks the stage as failed


if __name__ == "__main__":
    main()
