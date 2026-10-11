import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.constants import ID_COLS, LABEL_COLS, CATEGORICAL_COL, SKEWED_COLS, NORMAL_COLS

def load_interim(interim_dir: str):
    """Load the train and test files made by data ingestion."""
    try:
        train_df = pd.read_csv(os.path.join(interim_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(interim_dir, "test.csv"))
        logging.info("Loaded train %s and test %s", train_df.shape, test_df.shape)
        return train_df, test_df
    except Exception as e:
        logging.error("Failed to load interim data: %s", e)
        raise CustomException(e, sys)

def split_x_y(df: pd.DataFrame):
    """Separate the features (X) from the labels (y).
    Labels are kept only for the final evaluation, never for training."""
    try:
        y = df[LABEL_COLS].copy()
        X = df.drop(columns = ID_COLS + LABEL_COLS + [CATEGORICAL_COL] )   # Type dropped: no effect on sensors
        return X, y
    except Exception as e:
        logging.error("Failed to split X and y: %s", e)
        raise CustomException(e, sys)

def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add two engineered features. Row-by-row maths, so no leakage."""
    try:
        X = X.copy()
        # Heat difference between process and air
        X["temp_diff"] = X["Process temperature [K]"] - X["Air temperature [K]"]
        # Mechanical power in watts: torque * angular speed
        X["power_W"] = X["Torque [Nm]"] * X["Rotational speed [rpm]"] * 2 * np.pi / 60
        return X
    except Exception as e:
        logging.error("Failed to add features: %s", e)
        raise CustomException(e, sys)

def build_preprocessor() -> ColumnTransformer:
    """Yo-Johnson (skewed columns) + StandardScaler (the rest)."""
    try:
        return ColumnTransformer(
            transformers=[
                ("skewed", PowerTransformer(method="yeo-johnson"), SKEWED_COLS),
                ("normal", StandardScaler(), NORMAL_COLS),
            ],
            verbose_feature_names_out=False,     # Keep the original column names
        ).set_output(transform="pandas")
    except Exception as e:
        logging.error("Failed to build the preprocessor: %s", e)
        raise CustomException(e, sys)

def save_outputs(X_train, X_test, y_train, y_test, preprocessor, processed_dir: str, model_dir: str) -> None:
    """Save the processed data and the fitted preprocessor."""
    try:
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok= True)
        X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index = False)
        X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index = False)
        y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
        joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.joblib"))
        logging.debug("Saved processed data to %s and preprocessor to %s", processed_dir, model_dir)
    except Exception as e:
        logging.error("Failed to save outputs: %s", e)
        raise CustomException(e, sys)

def main():
    try:
        # 1. Load train and test
        train_df, test_df = load_interim("data/interim")

        # 2. Separate features and labels
        X_train, y_train = split_x_y(train_df)
        X_test, y_test = split_x_y(test_df)

        # 3. Add engineered features
        X_train = add_features(X_train)
        X_test = add_features(X_test)

        # 4. Fit on TRAIN only, then transform both (this avoids leakage)
        preprocessor = build_preprocessor()
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        # 5. Quick sanity check in the log
        logging.info("Processed shapes: train %s, test %s", X_train_t.shape, X_test_t.shape)
        logging.info("Skew after transform:\n%s", X_train_t.skew().round(3).to_string())

        # 6. Save everything
        save_outputs(X_train_t, X_test_t, y_train, y_test, preprocessor, processed_dir="data/processed", model_dir="models")
        logging.info("Data preprocessing Complete")
    except Exception as e:
        logging.error("Data preprocessing failed: %s", e)
        raise

if __name__ == "__main__":
    main()