import subprocess
import sys
import textwrap

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.constants import SENSOR_COLS
from src.model.anomaly_model import log_model
from src.model.model_building import load_params
from src.model import register_model as reg
from src.pipeline.predict_pipeline import predict

GATE = {"min_consensus_ap_mean": 0.30, "min_seeds_won": 9}
GOOD = {"n_seeds": 10, "consensus_ap_mean": 0.405, "consensus_ap_std": 0.049, "consensus_roc_auc_mean": 0.875,
        "ap_diff_consensus_minus_iforest_seeds_won": 10, "ap_diff_consensus_minus_dbscan_seeds_won": 10,
        "ap_diff_consensus_minus_gmm_seeds_won": 10}


def make_df(n=800, seed=0):
    rng = np.random.default_rng(seed)
    air = rng.normal(300, 2, n)
    speed = rng.normal(1500, 150, n).clip(1200, None)
    df = pd.DataFrame({
        SENSOR_COLS[0]: air, SENSOR_COLS[1]: air + 10 + rng.normal(0, 1, n), SENSOR_COLS[2]: speed,
        SENSOR_COLS[3]: 60000 / speed + rng.normal(0, 3, n), SENSOR_COLS[4]: rng.uniform(0, 250, n)})
    df.loc[:4, SENSOR_COLS[3]] = rng.uniform(72, 95, 5)            # a few planted anomalies
    return df


# ---------------------------------------------------------------- the quality gate
def test_gate_passes_for_the_current_benchmark():
    assert reg.check_gate(GOOD, GATE) == []


def test_gate_rejects_low_ap():
    problems = reg.check_gate({**GOOD, "consensus_ap_mean": 0.25}, GATE)
    assert len(problems) == 1 and "AP" in problems[0]


def test_gate_rejects_a_method_that_beats_the_consensus_too_often():
    problems = reg.check_gate({**GOOD, "ap_diff_consensus_minus_dbscan_seeds_won": 6}, GATE)
    assert len(problems) == 1 and "dbscan" in problems[0]


# ---------------------------------------------------------------- log, register, promote, load
@pytest.fixture(scope="module")
def registered(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mlflow")
    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")          # a local registry, no network
    experiment_id = mlflow.create_experiment("test", artifact_location=(tmp / "artifacts").as_uri())
    mlflow.set_experiment(experiment_id=experiment_id)             # artifacts stay in pytest's temp folder
    with mlflow.start_run() as run:
        model_info = log_model()
    info = {"run_id": run.info.run_id, "model_uri": model_info.model_uri}
    version = reg.register_model("anomaly-consensus", info, GOOD)
    return {"version": version, "uri": "models:/anomaly-consensus@staging", "dir": tmp}


def test_version_gets_staging_alias_and_tags(registered):
    client = mlflow.tracking.MlflowClient()
    mv = client.get_model_version_by_alias("anomaly-consensus", "staging")
    assert str(mv.version) == registered["version"]
    assert mv.tags["consensus_ap_mean"] == "0.405" and "git_commit" in mv.tags


def test_registering_twice_makes_version_2(registered):
    mv = mlflow.tracking.MlflowClient().get_model_version_by_alias("anomaly-consensus", "staging")
    info = {"run_id": mv.run_id, "model_uri": mv.source}
    assert reg.register_model("anomaly-consensus", info, GOOD) == "2"


def test_promote_moves_production_to_the_staging_version(registered):
    version = reg.promote_to_production("anomaly-consensus")
    mv = mlflow.tracking.MlflowClient().get_model_version_by_alias("anomaly-consensus", "production")
    assert str(mv.version) == version


def test_registered_model_matches_predict(registered):
    df = make_df()
    scores = mlflow.pyfunc.load_model(registered["uri"]).predict(df)
    expected = predict(df, 0.03, load_params("params.yaml"), use_shap=False)["scores"]
    assert list(scores.columns) == ["row_id", "consensus", "iforest", "dbscan", "gmm", "methods_agree", "flagged"]
    np.testing.assert_allclose(scores["consensus"], expected["consensus"])
    assert int(scores["flagged"].sum()) == 24                      # 3% of 800 rows


def test_integer_sensor_columns_are_accepted(registered):
    df = make_df()
    df[SENSOR_COLS[2]] = df[SENSOR_COLS[2]].round().astype(int)    # like the rotational speed column in AI4I
    assert len(mlflow.pyfunc.load_model(registered["uri"]).predict(df)) == 800


def test_review_pct_can_be_changed_per_call(registered):
    model = mlflow.pyfunc.load_model(registered["uri"])
    assert int(model.predict(make_df(), params={"review_pct": 0.05})["flagged"].sum()) == 40


def test_model_runs_in_a_fresh_process_without_the_project(registered, tmp_path):
    """The code and settings travel inside the model: load it from an empty folder."""
    make_df().to_csv(tmp_path / "in.csv", index=False)

    tracking_uri = f"sqlite:///{registered['dir'].as_posix()}/mlflow.db"

    script = textwrap.dedent(f"""
        import mlflow, pandas as pd
        mlflow.set_tracking_uri("{tracking_uri}")
        model = mlflow.pyfunc.load_model("models:/anomaly-consensus@staging")
        out = model.predict(pd.read_csv("in.csv"))
        print(len(out), int(out["flagged"].sum()))
    """)

    r = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr[-1500:]
    assert r.stdout.strip().splitlines()[-1] == "800 24"



def test_a_version_that_cannot_be_loaded_is_rejected_and_never_becomes_staging(registered):
    client = mlflow.tracking.MlflowClient()

    before = client.get_model_version_by_alias("anomaly-consensus", "staging").version
    run_id = client.get_model_version_by_alias("anomaly-consensus", "staging").run_id
    broken = {"run_id": run_id,"model_uri": f"runs:/{run_id}/model",}

    with pytest.raises(Exception):
        reg.register_model("anomaly-consensus", broken, GOOD)
    after = client.get_model_version_by_alias("anomaly-consensus", "staging").version
    assert after == before