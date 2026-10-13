import mlflow
import pandas as pd
import yaml
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema

from src.pipeline.predict_pipeline import predict

# Packages needed to load and run the model elsewhere (SHAP is not needed: explanations are switched off)
MODEL_REQUIREMENTS = [
    "numpy>=1.26", "pandas>=2.1", "scipy>=1.11",
    "scikit-learn>=1.4", "joblib>=1.3", "pyyaml>=6.0",
]


class AnomalyConsensusModel(mlflow.pyfunc.PythonModel):
    """The whole method (code + settings) as ONE versioned Mlflow model.

    Input: a DataFrame with the 5 sensor columns (extra columns are ignored)
    Output: one row per input row with the three scores, the consensus, how many methods
            agree and whether the row is flagged (the top review_pct of rows).
    The models are fitted on the rows you pass in, so scores are ranks within that batch.
    """

    def load_context(self, context):
        with open(context.artifacts["params"], "r") as f:
            self.params = yaml.safe_load(f)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        review_pct = float((params or {}).get("review_pct", self.params["app"]["review_pct_default"]))
        result = predict(model_input, review_pct, self.params, use_shap=False)
        return result["scores"].drop(columns=["pc1", "pc2"])


def log_model(params_path: str = "params.yaml", code_path: str = "src"):
    """Log the model to the ACTIVE Mlflow run. The settings (params.yaml) and the code (src/)
    are saved inside the model, so a registered version can be loaded and run on its own."""

    with open(params_path, "r") as f:
        default_pct = float(yaml.safe_load(f)["app"]["review_pct_default"])

    # No input schema on purpose: Mlflow refuses int64 -> double, but real files have integer
    # sensor columns (AI4I speed and tool wear). predict() checks the columns itself and
    # converts the values. review_pct must be declared here, or Mlflow drops it at call time.
    signature = ModelSignature(
        inputs=None,
        outputs=Schema([
            ColSpec("long", "row_id"),
            ColSpec("double", "consensus"),
            ColSpec("double", "iforest"),
            ColSpec("double", "dbscan"),
            ColSpec("double", "gmm"),
            ColSpec("long", "methods_agree"),
            ColSpec("boolean", "flagged"),
        ]),
        params=ParamSchema([ParamSpec("review_pct", "double", default_pct)]),
    )

    return mlflow.pyfunc.log_model(
        name="model",
        python_model=AnomalyConsensusModel(),
        artifacts={"params": params_path},
        code_paths=[code_path],
        pip_requirements=MODEL_REQUIREMENTS,
        signature=signature,
    )
