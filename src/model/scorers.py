import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from src.logger import logging
from src.exception import CustomException

class DBSCANScorer:
    """DBSCAN anomaly score = distance to the nearest core point.
    Higher means more unusual. Rows farther thatn eps from every core point are noise."""
    def __init__(self, eps: float, min_samples: int):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X: pd.DataFrame):
        try:
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1).fit(X)
            core_points = X.iloc[db.core_sample_indices_]      # dense-region points only
            self.nn_ = NearestNeighbors(n_neighbors=1).fit(core_points)
            self.train_noise_rate_ = float(np.mean(db.labels_ == -1))
            self.n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
            return self
        except Exception as e:
            logging.error("Failed to fit DBSCANScorer: %s", e)
            raise CustomException(e, sys)

    def score (self, X:pd.DataFrame) -> np.ndarray:
        try:
            return self.nn_.kneighbors(X)[0].ravel()
        except Exception as e:
            logging.error("Failed to score with DBSCANScorer: %s", e)
            raise CustomException(e, sys)

def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Turn scores into ranks between 0 and 1, so different methods can be averaged."""
    return rankdata(scores) /len(scores)

def consensus_score(score_list: list) -> np.ndarray:
    """Average of the rank-normalized scores of all models (higher = more unusual)."""
    try:
        return np.mean([rank_normalize(s) for s in score_list], axis=0)
    except Exception as e:
        logging.error("Failed to compute the consensus score: %s", e)
        raise CustomException(e, sys)
    