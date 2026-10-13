import io
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main
from src.constants import SENSOR_COLS

N_PLANTED = 10


def make_df(n=1000, seed=0):
    """Synthetic data shaped like AI4I. The first 10 rows are planted anomalies."""
    rng = np.random.default_rng(seed)
    air = rng.normal(300, 2, n)
    process = air + 10 + rng.normal(0, 1, n)
    speed = rng.normal(1500, 150, n).clip(1200, None)
    torque = 60000 / speed + rng.normal(0, 3, n)
    wear = rng.uniform(0, 250, n)
    torque[:5] = rng.uniform(72, 95, 5)
    speed[5:10] = rng.uniform(2200, 2800, 5)
    return pd.DataFrame(dict(zip(SENSOR_COLS, [air, process, speed, torque, wear])))


def to_bytes(df, sep=",", bom=False):
    text = df.to_csv(index=False, sep=sep)
    return (("\ufeff" if bom else "") + text).encode("utf-8")


def upload(client, content, name="data.csv", review_pct="0.03"):
    return client.post("/api/analyze", files={"file": (name, content, "text/csv")},
                       data={"review_pct": review_pct})


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:          # the context manager also runs the start-up code
        yield c


@pytest.fixture(scope="module")
def analysis(client):
    r = upload(client, to_bytes(make_df()))
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------- JSON API
def test_health(client):
    assert client.get("/health").json()["status"] == "ok"

def test_health_shows_the_model_version(client, monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "7")
    assert client.get("/health").json()["model_version"] == "7"

def test_analyze_returns_summary_and_links(analysis):
    assert analysis["summary"]["n_rows"] == 1000
    assert analysis["summary"]["n_flagged"] == 30
    assert analysis["results_page"] == f"/results/{analysis['job_id']}"
    assert set(analysis["downloads"]) == {"flagged_csv", "all_scores_csv", "summary_json"}


def test_results_endpoint(client, analysis):
    r = client.get(f"/api/results/{analysis['job_id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["total_flagged"] == 30 and len(body["flagged_rows"]) == 30
    assert {"row_id", "consensus_score", "methods_agree", "top_features"} <= set(body["flagged_rows"][0])
    assert len(client.get(f"/api/results/{analysis['job_id']}?limit=5").json()["flagged_rows"]) == 5


def test_plot_data(client, analysis):
    p = client.get(f"/api/results/{analysis['job_id']}/plots").json()
    assert sum(p["histogram"]["counts"]) == 1000
    assert len(p["histogram"]["edges"]) == 41
    assert len(p["scatter"]["flagged"]["row_id"]) == 30
    assert len(p["scatter"]["normal"]["row_id"]) == 970


def test_agreement_counts_are_consistent(analysis):
    s = analysis["summary"]
    assert 0 <= s["n_flagged_by_all_three"] <= s["n_flagged"] <= s["n_rows"]


# ---------------------------------------------------------------- downloads
def test_downloads(client, analysis):
    base = f"/report/{analysis['job_id']}"
    flagged = client.get(f"{base}/flagged.csv")
    assert flagged.status_code == 200 and "attachment" in flagged.headers["content-disposition"]
    table = pd.read_csv(io.StringIO(flagged.text))
    assert len(table) == 30 and table.columns[0] == "row_id"          # no pandas index column

    scores = pd.read_csv(io.StringIO(client.get(f"{base}/all_scores.csv").text))
    assert len(scores) == 1000

    summary = client.get(f"{base}/summary.json").json()
    assert summary["n_flagged"] == 30


def test_semicolon_separated_file_with_bom(client):
    r = upload(client, to_bytes(make_df(), sep=";", bom=True))
    assert r.status_code == 200 and r.json()["summary"]["n_rows"] == 1000


def test_text_ids_with_formula_are_neutralised_in_csv(client):
    df = make_df()
    df["Product ID"] = [f"L{i}" for i in range(len(df))]
    df.loc[0, "Product ID"] = "=1+1"                                   # a planted anomaly, so it is flagged
    job = upload(client, to_bytes(df)).json()["job_id"]
    text = client.get(f"/report/{job}/flagged.csv").text
    assert "'=1+1" in text and ",=1+1," not in text


def test_missing_ids_do_not_break_json(client):
    df = make_df()
    df["Product ID"] = None
    r = upload(client, to_bytes(df))
    assert r.status_code == 200
    rows = client.get(f"/api/results/{r.json()['job_id']}").json()["flagged_rows"]
    assert rows[0]["Product ID"] is None


# ---------------------------------------------------------------- errors the user can fix
@pytest.mark.parametrize("content,name,expected", [
    (to_bytes(make_df().drop(columns=["Torque [Nm]"])), "data.csv", "Torque"),
    (to_bytes(make_df(n=100)), "data.csv", "rows"),
    (b"", "data.csv", "empty"),
    (to_bytes(make_df()), "data.xlsx", ".csv"),
    (b"a,b\n" * (3 * 1024 * 1024), "big.csv", "larger"),
])
def test_user_errors_are_400_with_a_clear_message(client, content, name, expected):
    r = upload(client, content, name=name)
    assert r.status_code == 400
    assert expected in r.json()["detail"]


def test_review_pct_too_large(client):
    assert upload(client, to_bytes(make_df()), review_pct="0.9").status_code == 400


def test_unknown_job_is_404(client):
    for path in ["/api/results/nope", "/api/results/nope/plots", "/report/nope/flagged.csv"]:
        assert client.get(path).status_code == 404


# ---------------------------------------------------------------- server limits
def test_busy_server_returns_503(client):
    held = [main.SLOTS.acquire(blocking=False) for _ in range(main.MAX_PARALLEL)]
    try:
        assert upload(client, to_bytes(make_df())).status_code == 503
    finally:
        for ok in held:
            if ok:
                main.SLOTS.release()


def test_expired_jobs_are_removed(client, monkeypatch, analysis):
    monkeypatch.setattr(main, "JOB_TTL_SECONDS", -1)
    assert client.get(f"/api/results/{analysis['job_id']}").status_code == 404


def test_oldest_jobs_are_dropped_when_there_are_too_many(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_JOBS", 1)
    first = upload(client, to_bytes(make_df(seed=1))).json()["job_id"]
    second = upload(client, to_bytes(make_df(seed=2))).json()["job_id"]
    assert client.get(f"/api/results/{first}").status_code == 404
    assert client.get(f"/api/results/{second}").status_code == 200


# ---------------------------------------------------------------- HTML pages
def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and 'action="/analyze"' in r.text


def test_form_upload_redirects_to_results_page(client):
    r = client.post("/analyze", files={"file": ("data.csv", to_bytes(make_df()), "text/csv")},
                    data={"review_pct": "0.03"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/results/")
    assert client.get(r.headers["location"]).status_code == 200


def test_form_upload_shows_the_error_message(client):
    r = client.post("/analyze", files={"file": ("data.csv", to_bytes(make_df(n=100)), "text/csv")},
                    data={"review_pct": "0.03"})
    assert r.status_code == 400 and "At least 500 are needed" in r.text


def test_results_page_escapes_text_from_the_file(client):
    df = make_df()
    df["Product ID"] = [f"L{i}" for i in range(len(df))]
    df.loc[0, "Product ID"] = "</script><script>alert(1)</script>"      # row 1 is a planted anomaly
    r = client.post("/analyze", files={"file": ("<b>x</b>.csv", to_bytes(df), "text/csv")},
                    data={"review_pct": "0.03"}, follow_redirects=False)
    page = client.get(r.headers["location"]).text
    assert "<script>alert(1)" not in page                                 # not injected as a tag
    assert "\\u003cscript\\u003ealert(1)" in page                         # embedded safely as JSON
    assert "&lt;b&gt;x&lt;/b&gt;.csv" in page                             # file name escaped


def test_results_page_shows_notices_and_label_check(client):
    df = make_df()                                                        # 1,000 rows: a small file
    df["Machine failure"] = 0
    df.loc[:N_PLANTED - 1, "Machine failure"] = 1
    r = client.post("/analyze", files={"file": ("data.csv", to_bytes(df), "text/csv")},
                    data={"review_pct": "0.03"}, follow_redirects=False)
    page = client.get(r.headers["location"]).text
    assert "This is a small file" in page
    assert "Check against the failure labels" in page


def test_unknown_results_page_is_404(client):
    assert client.get("/results/nope").status_code == 404