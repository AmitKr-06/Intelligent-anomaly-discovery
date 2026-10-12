import sys
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score

from src.logger import logging
from src.exception import CustomException, InvalidInputError
from src.constants import ID_COLS, TARGET_COL, SENSOR_COLS, REQUIRED_COLS, FEATURE_COLS
from src.features.data_preprocessing import add_features, build_preprocessor
from src.model.model_building import load_params, train_isolation_forest, train_dbscan, train_gmm
from src.model.metrics import score_all

MODEL_NAMES = ["iforest", "dbscan", "gmm"]


def validate_upload(df: pd.DataFrame, app_params: dict) -> None:
    """Check the things a user can fix: columns and file size."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise InvalidInputError(f"Missing required columns: {missing}")
    if len(df) < app_params["min_rows"]:
        raise InvalidInputError(
            f"The file has {len(df)} rows. At least {app_params['min_rows']} are needed "
            "for the density-based methods to work reliably.")
    if len(df) > app_params["max_rows"]:
        raise InvalidInputError(
            f"The file has {len(df)} rows. The limit is {app_params['max_rows']}.")


def clean_sensors(df: pd.DataFrame, app_params: dict):
    """Keep the 5 sensor columns, make them numeric and fill gaps with the median.
    Text or other bad values become missing values first."""
    X = df[SENSOR_COLS].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # A column that is mostly empty cannot be repaired by imputation
    missing_frac = X.isna().mean()
    too_empty = missing_frac[missing_frac > app_params["max_missing_frac"]]
    if len(too_empty):
        raise InvalidInputError(
            f"Too many missing or non-numeric values in: {list(too_empty.index)}")

    n_imputed = int(X.isna().sum().sum())
    X = X.fillna(X.median())

    # A constant column carries no information and breaks the power transform
    constant = [c for c in SENSOR_COLS if X[c].nunique() <= 1]
    if constant:
        raise InvalidInputError(f"These columns have a single constant value: {constant}")
    return X, n_imputed


def build_features(X_raw: pd.DataFrame):
    """Add temp_diff and power_W, then Yeo-Johnson + scaling, fitted on THIS file."""
    try:
        X_fe = add_features(X_raw)                           # raw units, used for display
        pre = build_preprocessor()
        X_t = pre.fit_transform(X_fe)[FEATURE_COLS]          # fixed column order
        return X_fe, X_t
    except Exception as e:
        logging.error("Failed to build features: %s", e)
        raise CustomException(e, sys)


def fit_models(X_t: pd.DataFrame, params: dict) -> dict:
    """Fit the three models on the uploaded file (settings come from params.yaml)."""
    try:
        mb = params["model_building"]
        seed = mb["random_state"]
        return {
            "iforest": train_isolation_forest(X_t, mb["isolation_forest"], seed),
            "dbscan": train_dbscan(X_t, mb["dbscan"]),
            "gmm": train_gmm(X_t, mb["gmm"], seed),
        }
    except Exception as e:
        logging.error("Failed to fit the models: %s", e)
        raise CustomException(e, sys)


def flag_top(scores: dict, review_pct: float):
    """Flag the top review_pct of rows by consensus score. Also count how many of the
    three methods put each row in their own top group (3 = highest confidence)."""
    n = len(scores["consensus"])
    k = max(1, int(np.ceil(n * review_pct)))

    order = np.argsort(scores["consensus"])[::-1][:k]        # most suspicious first
    flagged = np.zeros(n, dtype=bool)
    flagged[order] = True

    agree = np.zeros(n, dtype=int)
    for name in MODEL_NAMES:
        s = scores[name]
        own_top = np.argsort(s)[-k:]
        if name == "dbscan":                                 # most rows score exactly 0: ignore ties
            own_top = own_top[s[own_top] > 0]
        agree[own_top] += 1
    return flagged, order, agree, k


def top_feature_text(X_t: pd.DataFrame, idx: np.ndarray, top_n: int) -> list:
    """For each row, the features that are furthest from typical. Values are in
    standard-deviation units of the transformed data. These are clues, not causes."""
    Z = X_t.iloc[idx].to_numpy()
    cols = np.array(X_t.columns)
    best = np.argsort(-np.abs(Z), axis=1)[:, :top_n]
    return [", ".join(f"{cols[j]} ({Z[i, j]:+.1f} sd)" for j in best[i]) for i in range(len(idx))]


def shap_top_feature(iforest, X_rows: pd.DataFrame):
    """Feature that pushes each row most towards 'anomalous' in Isolation Forest.
    Optional: if SHAP fails, the app still works without it."""
    try:
        import shap                                          # slow import, so only here
        values = shap.TreeExplainer(iforest).shap_values(X_rows)
        return pd.DataFrame(values, columns=X_rows.columns).idxmin(axis=1).to_numpy()
    except Exception as e:
        logging.warning("SHAP explanation skipped: %s", e)
        return None


def pca_2d(X_t: pd.DataFrame):
    """Two principal components for the scatter plot."""
    pca = PCA(n_components=2, random_state=0)
    return pca.fit_transform(X_t), pca.explained_variance_ratio_


def evaluate_if_labels(df: pd.DataFrame, scores: dict):
    """If the file has a 0/1 failure column, report how well the scores rank it.
    The labels are never used to fit anything."""
    if TARGET_COL not in df.columns:
        return None
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if y.isna().any() or not set(y.unique()) <= {0, 1} or y.nunique() < 2:
        return None
    y = y.astype(int).to_numpy()
    out = {"n_failures": int(y.sum()), "random_baseline_ap": float(y.mean())}
    for name, s in scores.items():
        out[name] = {"roc_auc": float(roc_auc_score(y, s)),
                     "ap": float(average_precision_score(y, s))}
    return out


def make_warnings(n_rows: int, n_imputed: int, models: dict) -> list:
    """Plain-language notes shown on the results page."""
    notes = []
    if n_rows < 2000:
        notes.append("This is a small file. In our tests, on files under 2,000 rows the quality of the ranking "
                     "varied a lot between samples of the same data, and the DBSCAN settings were tuned on "
                     "about 8,000 rows. Treat the flagged list as a starting point.")
    elif n_rows < 5000:
        notes.append("DBSCAN settings were tuned on about 8,000 rows, so results on a smaller file are less reliable.")
    if n_imputed > 0:
        notes.append(f"{n_imputed} missing or non-numeric values were replaced by the column median.")
    noise = models["dbscan"].train_noise_rate_
    if noise > 0.2 or noise == 0:
        notes.append(f"DBSCAN marked {noise:.0%} of rows as noise, so its eps value may not suit this data.")
    if not models["gmm"].converged_:
        notes.append("The Gaussian Mixture Model did not fully converge.")
    return notes


def predict(df: pd.DataFrame, review_pct: float, params: dict, use_shap: bool = True) -> dict:
    """Upload -> validate -> features -> fit -> score -> flag -> explain.
    Returns {'scores': all rows, 'flagged': table of flagged rows, 'summary': dict}.
    row_id starts at 1 (first data row of the file)."""
    try:
        start = time.time()
        app_params = params["app"]

        if not (0 < review_pct <= app_params["review_pct_max"]):
            raise InvalidInputError(f"review_pct must be between 0 and {app_params['review_pct_max']}")

        df = df.copy()
        df.columns = df.columns.str.strip()                  # tolerate stray spaces in headers
        df = df.reset_index(drop=True)
        validate_upload(df, app_params)

        # 1. Features
        X_raw, n_imputed = clean_sensors(df, app_params)
        X_fe, X_t = build_features(X_raw)
        n = len(df)

        # 2. Models and scores
        models = fit_models(X_t, params)
        scores = score_all(models, X_t)
        flagged, order, agree, k = flag_top(scores, review_pct)
        coords, var_ratio = pca_2d(X_t)

        # 3. Table with every row (used for the plots)
        scored = pd.DataFrame({
            "row_id": np.arange(1, n + 1),
            "consensus": scores["consensus"],
            "iforest": scores["iforest"],
            "dbscan": scores["dbscan"],
            "gmm": scores["gmm"],
            "methods_agree": agree,
            "flagged": flagged,
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
        })

        # 4. Table of flagged rows, most suspicious first
        flagged_df = X_fe.iloc[order].round(2).copy()        # raw sensor values + derived features
        flagged_df.insert(0, "row_id", order + 1)
        pos = 1
        for c in ID_COLS:                                    # keep IDs if the file has them
            if c in df.columns:
                flagged_df.insert(pos, c, df[c].iloc[order].to_numpy())
                pos += 1
        flagged_df["consensus_score"] = scores["consensus"][order].round(4)
        flagged_df["methods_agree"] = agree[order]
        flagged_df["top_features"] = top_feature_text(X_t, order, app_params["top_features"])

        shap_col = np.array([""] * len(order), dtype=object)
        if use_shap:
            m = min(len(order), app_params["shap_max_rows"])
            shap_vals = shap_top_feature(models["iforest"], X_t.iloc[order[:m]])
            if shap_vals is not None:
                shap_col[:m] = shap_vals
        flagged_df["iforest_shap_top"] = shap_col

        # 5. Summary
        summary = {
            "n_rows": n,
            "n_flagged": int(k),
            "review_pct": float(review_pct),
            "cutoff_consensus_score": float(scores["consensus"][order].min()),
            "n_flagged_by_all_three": int((agree[order] == 3).sum()),
            "n_imputed_cells": n_imputed,
            "features_used": FEATURE_COLS,
            "id_columns_found": [c for c in ID_COLS if c in df.columns],
            "typical_values": {c: float(v) for c, v in X_fe.median().round(2).items()},
            "pca_explained_variance": [float(v) for v in var_ratio],
            "dbscan_train_noise_rate": float(models["dbscan"].train_noise_rate_),
            "dbscan_n_clusters": int(models["dbscan"].n_clusters_),
            "gmm_converged": bool(models["gmm"].converged_),
            "warnings": make_warnings(n, n_imputed, models),
            "evaluation_against_labels": evaluate_if_labels(df, scores),
            "runtime_seconds": round(time.time() - start, 1),
        }
        logging.info("Analysis done: %s rows, %s flagged, %.1f s", n, k, summary["runtime_seconds"])
        return {"scores": scored, "flagged": flagged_df, "summary": summary}

    except InvalidInputError:
        raise                                                # the API turns this into a clean message
    except Exception as e:
        logging.error("Analysis failed: %s", e)
        raise CustomException(e, sys)