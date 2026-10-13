import io
import os
import json
import time
import uuid
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.logger import logging
from src.exception import InvalidInputError
from src.pipeline.predict_pipeline import predict, load_params

# Project paths and settings
ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(str(ROOT / "params.yaml"))
APP_PARAMS = PARAMS["app"]

# Server limits
MAX_UPLOAD_BYTES = 10 * 1024 * 1024    # 10 MB
JOB_TTL_SECONDS = 60 * 60              # results are kept for 1 hour
MAX_JOBS = 20                          # results kept in memory at the same time
MAX_PARALLEL = 2                       # analyses running at the same time
TABLE_ROWS = 300                       # flagged rows shown in the page (the CSV has all of them)
SCATTER_POINTS = 4000                  # normal rows drawn in the scatter plot (all flagged rows are drawn)

# In-memory job store: lost on restart, so run the app with ONE worker process
JOBS = {}
JOBS_LOCK = threading.Lock()
SLOTS = threading.BoundedSemaphore(MAX_PARALLEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Import SHAP once at start-up, so the first upload is not slow."""
    try:
        import shap  # noqa: F401
    except Exception as e:
        logging.warning("SHAP is not available, explanations will be skipped: %s", e)
    yield


app = FastAPI(title="Intelligent Anomaly Discovery Platform", lifespan=lifespan)
templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static"), check_dir=False), name="static")


@app.exception_handler(InvalidInputError)
async def invalid_input_handler(request: Request, exc: InvalidInputError):
    """Problems the user can fix (missing column, file too small) are shown as a clear 400 message."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ---------------------------------------------------------------- helpers
def _purge_old_jobs() -> None:
    """Drop expired jobs, then the oldest ones if there are still too many. Caller holds the lock."""
    now = time.time()
    for job_id in [j for j, v in JOBS.items() if now - v["created"] > JOB_TTL_SECONDS]:
        del JOBS[job_id]
    while len(JOBS) > MAX_JOBS:
        del JOBS[min(JOBS, key=lambda j: JOBS[j]["created"])]


def _get_job(job_id: str) -> dict:
    with JOBS_LOCK:
        _purge_old_jobs()
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Result not found. Results are kept for 1 hour.")
    return job


def _read_csv(file: UploadFile) -> pd.DataFrame:
    """Read the uploaded CSV (comma, semicolon or tab separated)."""
    if not (file.filename or "").lower().endswith(".csv"):
        raise InvalidInputError("Please upload a .csv file.")
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise InvalidInputError(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not raw.strip():
        raise InvalidInputError("The file is empty.")

    try:
        text = raw.decode("utf-8-sig")       # also removes the BOM that Excel adds
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        if df.shape[1] > 1:
            return df
    raise InvalidInputError("The file could not be read as a table (comma, semicolon or tab separated).")


def _run_job(file: UploadFile, review_pct: float):
    """Read the file, run the analysis and store the result. Returns (job_id, summary)."""
    df = _read_csv(file)

    if not SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="The server is busy with other analyses. Try again in a minute.")
    try:
        result = predict(df, review_pct, PARAMS, use_shap=True)
    except InvalidInputError:
        raise
    except Exception as e:
        logging.error("Analysis failed: %s", e)
        raise HTTPException(status_code=500, detail="The analysis failed unexpectedly. Details are in the server log.")
    finally:
        SLOTS.release()

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        _purge_old_jobs()
        JOBS[job_id] = {"created": time.time(), "filename": file.filename or "upload.csv", **result}
    return job_id, result["summary"]


def _csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Stop spreadsheet programs from running a text cell that starts with = + - @ as a formula."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].map(
                lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r") else v)
    return out


def _records(df: pd.DataFrame) -> list:
    """DataFrame -> list of dicts that is safe to send as JSON (NaN becomes null)."""
    return json.loads(df.to_json(orient="records"))


def _download(content: str, media_type: str, filename: str) -> Response:
    return Response(content=content, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def _plot_data(job: dict) -> dict:
    """Histogram bins and 2D PCA points for the charts."""
    scores, summary = job["scores"], job["summary"]
    counts, edges = np.histogram(scores["consensus"], bins=40, range=(0, 1))

    flagged = scores[scores["flagged"]]
    normal = scores[~scores["flagged"]]
    if len(normal) > SCATTER_POINTS:
        normal = normal.sample(SCATTER_POINTS, random_state=0)

    def pack(d: pd.DataFrame) -> dict:
        return {"row_id": d["row_id"].tolist(),
                "pc1": d["pc1"].round(4).tolist(), "pc2": d["pc2"].round(4).tolist(),
                "consensus": d["consensus"].round(4).tolist(),
                "methods_agree": d["methods_agree"].tolist()}

    return {"histogram": {"edges": edges.round(4).tolist(), "counts": counts.tolist()},
            "cutoff": summary["cutoff_consensus_score"],
            "pca_explained_variance": summary["pca_explained_variance"],
            "scatter": {"normal": pack(normal), "flagged": pack(flagged)}}


def _download_links(job_id: str) -> dict:
    return {"flagged_csv": f"/report/{job_id}/flagged.csv",
            "all_scores_csv": f"/report/{job_id}/all_scores.csv",
            "summary_json": f"/report/{job_id}/summary.json"}


# ---------------------------------------------------------------- pages
@app.get("/health")
def health():
    with JOBS_LOCK:
        return {"status": "ok", "jobs_in_memory": len(JOBS), "model_version": os.getenv("MODEL_VERSION", "dev")}


def _index_context(error: str = None) -> dict:
    return {"error": error,
            "default_pct": APP_PARAMS["review_pct_default"], "max_pct": APP_PARAMS["review_pct_max"],
            "min_rows": APP_PARAMS["min_rows"], "max_rows": APP_PARAMS["max_rows"]}


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context=_index_context())


@app.post("/analyze")
def analyze_form(request: Request, file: UploadFile = File(...),
                 review_pct: float = Form(APP_PARAMS["review_pct_default"])):
    """Form from the upload page: on success go to the results page, otherwise show the message."""
    try:
        job_id, _ = _run_job(file, review_pct)
    except InvalidInputError as e:
        return templates.TemplateResponse(request=request, name="index.html",
                                          context=_index_context(str(e)), status_code=400)
    except HTTPException as e:
        return templates.TemplateResponse(request=request, name="index.html",
                                          context=_index_context(str(e.detail)), status_code=e.status_code)
    return RedirectResponse(url=f"/results/{job_id}", status_code=303)


@app.get("/results/{job_id}")
def results_page(request: Request, job_id: str):
    job = _get_job(job_id)
    flagged = job["flagged"]
    context = {"job_id": job_id, "filename": job["filename"], "summary": job["summary"],
               "columns": list(flagged.columns), "rows": _records(flagged.head(TABLE_ROWS)),
               "total_flagged": len(flagged), "downloads": _download_links(job_id)}
    return templates.TemplateResponse(request=request, name="results.html", context=context)


# ---------------------------------------------------------------- JSON API
@app.post("/api/analyze")
def api_analyze(file: UploadFile = File(...), review_pct: float = Form(APP_PARAMS["review_pct_default"])):
    """Upload a CSV and run the analysis. review_pct is a fraction (0.03 = flag the top 3% of rows)."""
    job_id, summary = _run_job(file, review_pct)
    return {"job_id": job_id, "results_page": f"/results/{job_id}",
            "downloads": _download_links(job_id), "summary": summary}


@app.get("/api/results/{job_id}")
def api_results(job_id: str, limit: int = TABLE_ROWS):
    job = _get_job(job_id)
    limit = max(1, min(limit, len(job["flagged"])))
    return {"summary": job["summary"], "flagged_rows": _records(job["flagged"].head(limit)),
            "total_flagged": len(job["flagged"])}


@app.get("/api/results/{job_id}/plots")
def api_plots(job_id: str):
    return _plot_data(_get_job(job_id))


# ---------------------------------------------------------------- downloads
@app.get("/report/{job_id}/flagged.csv")
def report_flagged(job_id: str):
    job = _get_job(job_id)
    return _download(_csv_safe(job["flagged"]).to_csv(index=False), "text/csv; charset=utf-8",
                     f"flagged_rows_{job_id[:8]}.csv")


@app.get("/report/{job_id}/all_scores.csv")
def report_all_scores(job_id: str):
    job = _get_job(job_id)
    return _download(job["scores"].round(5).to_csv(index=False), "text/csv; charset=utf-8",
                     f"all_scores_{job_id[:8]}.csv")


@app.get("/report/{job_id}/summary.json")
def report_summary(job_id: str):
    job = _get_job(job_id)
    return _download(json.dumps(job["summary"], indent=2), "application/json", f"summary_{job_id[:8]}.json")