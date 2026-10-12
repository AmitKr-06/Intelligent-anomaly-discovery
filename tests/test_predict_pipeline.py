import json
import numpy as np
import pandas as pd
import pytest

from src.constants import SENSOR_COLS
from src.exception import InvalidInputError
from src.pipeline.predict_pipeline import predict, load_params

N_PLANTED = 10          # the first 10 rows are planted anomalies (row_id 1..10)


@pytest.fixture(scope="module")
def params():
    return load_params("params.yaml")


def make_data(n=1000, seed=0):
    """Synthetic data shaped like AI4I: torque falls as speed rises. The first 10 rows are
    planted anomalies, each off that curve by a different amount, so they cannot form
    one tight group (a GMM would treat such a group as a normal mode)."""
    rng = np.random.default_rng(seed)
    air = rng.normal(300, 2, n)
    process = air + 10 + rng.normal(0, 1, n)
    speed = rng.normal(1500, 150, n).clip(1200, None)
    torque = 60000 / speed + rng.normal(0, 3, n)
    wear = rng.uniform(0, 250, n)

    torque[:5] = rng.uniform(72, 95, 5)          # heavy load at a normal speed
    speed[5:10] = rng.uniform(2200, 2800, 5)     # very high speed, torque not lowered to match
    return pd.DataFrame(dict(zip(SENSOR_COLS, [air, process, speed, torque, wear])))


def test_flags_planted_anomalies(params):
    res = predict(make_data(), review_pct=0.03, params=params, use_shap=False)
    flagged_ids = set(res["flagged"]["row_id"])
    assert len(res["flagged"]) == 30
    assert len(flagged_ids & set(range(1, N_PLANTED + 1))) >= 6


def test_output_structure(params):
    res = predict(make_data(), 0.05, params, use_shap=False)
    assert len(res["scores"]) == 1000
    assert res["summary"]["n_flagged"] == 50
    assert res["summary"]["evaluation_against_labels"] is None      # no label column in the file
    assert {"row_id", "consensus_score", "methods_agree", "top_features"} <= set(res["flagged"].columns)


def test_summary_is_json_serialisable(params):
    res = predict(make_data(), 0.03, params, use_shap=False)
    json.dumps(res["summary"])                                      # the API will return this


def test_shap_column_present(params):
    res = predict(make_data(), 0.03, params, use_shap=True)
    assert "iforest_shap_top" in res["flagged"].columns


def test_missing_values_are_filled(params):
    df = make_data()
    df.iloc[::50, 2] = np.nan                                       # 20 blanks in the speed column
    res = predict(df, 0.03, params, use_shap=False)
    assert res["summary"]["n_imputed_cells"] == 20


def test_labels_are_evaluated_but_not_required(params):
    df = make_data()
    df["Machine failure"] = 0
    df.loc[:N_PLANTED - 1, "Machine failure"] = 1
    res = predict(df, 0.03, params, use_shap=False)
    ev = res["summary"]["evaluation_against_labels"]
    assert ev is not None
    assert ev["iforest"]["ap"] > 0.5 and ev["dbscan"]["ap"] > 0.5    # the two robust methods
    assert ev["consensus"]["ap"] > 0.5                                # random baseline is 0.01


def test_missing_column_is_rejected(params):
    with pytest.raises(InvalidInputError):
        predict(make_data().drop(columns=["Torque [Nm]"]), 0.03, params)


def test_small_file_is_rejected(params):
    with pytest.raises(InvalidInputError):
        predict(make_data(n=100), 0.03, params)


def test_bad_review_pct_is_rejected(params):
    with pytest.raises(InvalidInputError):
        predict(make_data(), 0.9, params)