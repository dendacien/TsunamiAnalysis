#!/usr/bin/env python3
"""
Flask app: Tsunami Prediction (HGBT, grouped CV, calibration, plots) + PERSISTENCE
- Upload a CSV to train ONCE, then save the calibrated model + threshold to disk
- Later, load the saved model and run predictions on user-provided inputs (form or JSON)

Tested with:
  Python 3.10+
  scikit-learn==1.7.2
  Flask>=3.0
  matplotlib>=3.6
  numpy, pandas, joblib

Run:
  pip install -r requirements.txt
  FLASK_ENV=development flask --app tsunami-hgbt-fla-app.py run --port 5000

Or:
  python tsunami-hgbt-flask-app.py
"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

# Use non-interactive backend for servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    RandomizedSearchCV,
    PredefinedSplit,
)
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted
import joblib

# ----------------------
# Config
# ----------------------
FEATS = [
    "magnitude",
    "cdi",
    "mmi",
    "sig",
    "nst",
    "dmin",
    "gap",
    "depth",
    "latitude",
    "longitude",
]
TARGET = "tsunami"
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ARTIFACT_DIR / "tsunami_model_bundle.pkl"

# HTML template (inline to keep single-file app)
HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tsunami Prediction (HGBT)</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css" />
  <style>
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }
    img { width: 100%; height: auto; border-radius: 12px; }
    .code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .muted { opacity: .7; }
    form.inline-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: .6rem; }
  </style>
</head>
<body class="container">
  <main>
    <h1>🌊 Tsunami Prediction — HGBT (scikit-learn)</h1>
    <details open>
      {% if model_loaded %}
        <p class="muted">Saved model found at <code>{{ model_path }}</code>. You can run predictions below without retraining.</p>
      {% else %}
        <p class="muted">No saved model detected yet.</p>
      {% endif %}
    </details>

    {% if report %}
    <hr/>
    <h2>Training Results</h2>
    <p><strong>Best CV PR-AUC:</strong> {{ report.best_cv_ap }} | <strong>Best Params:</strong> <span class="code">{{ report.best_params }}</span></p>
    <p><strong>Test PR-AUC:</strong> {{ report.test_pr_auc }} | <strong>Test ROC-AUC:</strong> {{ report.test_roc_auc }}</p>
    <p><strong>Test Precision:</strong> {{ report.precision }} | <strong>Recall:</strong> {{ report.recall }} | <strong>F1:</strong> {{ report.f1 }} | <strong>Threshold:</strong> {{ report.threshold }}</p>

    <h3>Plots</h3>
    <div class="grid">
      <figure><img src="data:image/png;base64,{{ imgs.pr }}" alt="PR curve"/><figcaption>Precision–Recall (Test)</figcaption></figure>
      <figure><img src="data:image/png;base64,{{ imgs.roc }}" alt="ROC curve"/><figcaption>ROC (Test)</figcaption></figure>
      <figure><img src="data:image/png;base64,{{ imgs.cal }}" alt="Calibration"/><figcaption>Calibration (Reliability) — Test</figcaption></figure>
      <figure><img src="data:image/png;base64,{{ imgs.thresh }}" alt="Threshold diagnostics"/><figcaption>Precision/Recall vs Threshold (Train)</figcaption></figure>
      <figure><img src="data:image/png;base64,{{ imgs.cm }}" alt="Confusion matrix"/><figcaption>Confusion Matrix (Test)</figcaption></figure>
      {% if imgs.imp %}
      <figure><img src="data:image/png;base64,{{ imgs.imp }}" alt="Permutation importance"/><figcaption>Permutation Importance (Test)</figcaption></figure>
      {% endif %}
    </div>
    {% endif %}

    <hr/>
    <h2>Predict with Saved Model</h2>
    {% if not model_loaded %}
      <p class="muted">Train once (above) to enable predictions, or drop a pre-trained <code>model.joblib</code> into <code>artifacts/</code> and refresh.</p>
    {% else %}
      <details open>
        <summary>Single Prediction (Form)</summary>
        <form class="inline-grid" method="POST" action="/predict">
          {% for f in features %}
            <label>{{ f }}
              <input type="number" step="any" name="{{ f }}" required />
            </label>
          {% endfor %}
          <label>Override threshold (optional)
            <input type="number" step="0.001" min="0" max="1" name="threshold" placeholder="auto" />
          </label>
          <button type="submit">Predict</button>
        </form>
      </details>

      <details>
        <summary>Batch Prediction (CSV)</summary>
        <form method="POST" action="/predict_csv" enctype="multipart/form-data">
          <label>CSV file with columns: <span class="code">{{ features|join(', ') }}</span>
            <input type="file" name="csv" accept=".csv" required />
          </label>
          <label>Override threshold (optional)
            <input type="number" step="0.001" min="0" max="1" name="threshold" placeholder="auto" />
          </label>
          <button type="submit">Predict CSV</button>
        </form>
      </details>

      <details>
        <summary>JSON API</summary>
        <p>POST <span class="code">/predict_json</span> with a JSON body like:</p>
<pre class="code">{
  "features": {"magnitude": 7.0, "cdi": 5, "mmi": 5, "sig": 820, "nst": 120,
               "dmin": 1.4, "gap": 20.0, "depth": 15.0, "latitude": -9.7, "longitude": 159.5},
  "threshold": 0.5  // optional
}</pre>
      </details>
    {% endif %}
  </main>
</body>
</html>
"""

def log1p_selected(X_df):
    # If input is a numpy array, convert to DataFrame with correct columns
    if isinstance(X_df, np.ndarray):
        X_df = pd.DataFrame(X_df, columns=FEATS)
    X_df = X_df.copy()
    for col in ("dmin", "gap", "depth"):
        if col in X_df.columns:
            vals = X_df[col].to_numpy()
            shift = 1 - np.nanmin(vals) if np.nanmin(vals) <= 0 else 0.0
            X_df[col] = np.log1p(vals + shift)
    return X_df


# ----------------------
# Plot helpers (return base64 PNG strings)
# ----------------------

def _fig_to_b64() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_pr_curve(y_true: np.ndarray, proba: np.ndarray, threshold: float | None = None) -> str:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    plt.figure()
    plt.plot(rec, prec, label="PR curve")
    if threshold is not None and len(thr) > 0:
        idx = np.argmin(np.abs(thr - threshold))
        plt.scatter(rec[idx], prec[idx])
        plt.text(rec[idx], prec[idx], f" thr={threshold:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall (Test)")
    plt.legend()
    return _fig_to_b64()


def plot_roc_curve(y_true: np.ndarray, proba: np.ndarray) -> str:
    fpr, tpr, _ = roc_curve(y_true, proba)
    from sklearn.metrics import auc
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC (Test)")
    plt.legend()
    return _fig_to_b64()


def plot_thresh_diag(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> str:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    plt.figure()
    plt.plot(thr, prec[:-1], label="Precision")
    plt.plot(thr, rec[:-1], label="Recall")
    plt.axvline(threshold, linestyle="--", label=f"thr={threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision/Recall vs Threshold (Train)")
    plt.legend()
    return _fig_to_b64()


def plot_calibration(y_true: np.ndarray, proba: np.ndarray) -> str:
    bins = np.linspace(0, 1, 11)
    inds = np.digitize(proba, bins) - 1
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    frac_pos = [np.mean(y_true[inds == i]) if np.any(inds == i) else np.nan for i in range(len(bin_centers))]
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    plt.plot(bin_centers, frac_pos, marker="o", label="Empirical")
    plt.xlabel("Predicted probability bin")
    plt.ylabel("Observed fraction positive")
    plt.title("Calibration (Reliability) — Test")
    plt.legend()
    return _fig_to_b64()


def plot_confusion(cm: np.ndarray) -> str:
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Pred 0", "Pred 1"]) ; plt.yticks(ticks, ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted") ; plt.ylabel("True")
    return _fig_to_b64()


def plot_permutation_importance(estimator: Pipeline, X: pd.DataFrame, y: np.ndarray, feature_names: List[str]) -> str:
    result = permutation_importance(estimator, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    order = np.argsort(result.importances_mean)
    plt.figure(figsize=(8, max(3, len(feature_names) * 0.35)))
    plt.barh(np.array(feature_names)[order], result.importances_mean[order])
    plt.xlabel("Mean decrease in score (perm)")
    plt.title("Permutation Importance (Test)")
    plt.tight_layout()
    return _fig_to_b64()


# ----------------------
# Persistence helpers
# ----------------------

def load_bundle() -> Dict | None:
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            print("Loaded bundle keys:", list(bundle.keys()))  # Add this line
            return bundle
        except Exception as e:
            print("Error loading bundle:", e)
            return None
    return None


# ----------------------
# Flask app
# ----------------------
app = Flask(__name__)

# In-memory holder for loaded bundle
BUNDLE: Dict | None = load_bundle()


@dataclass
class Report:
    best_cv_ap: str
    best_params: Dict
    test_pr_auc: str
    test_roc_auc: str
    precision: str
    recall: str
    f1: str
    threshold: str


@app.route("/", methods=["GET"])
def index():
    model_loaded = BUNDLE is not None
    return render_template_string(
        HTML,
        required_cols=FEATS + [TARGET],
        report=None,
        imgs=None,
        model_loaded=model_loaded,
        model_path=str(MODEL_PATH),
        features=FEATS,
    )


@app.route("/train", methods=["POST"])

# ---------- Prediction endpoints ----------

def _ensure_model_loaded():
    global BUNDLE
    if BUNDLE is None:
        BUNDLE = load_bundle()
    if BUNDLE is None:
        return None, ("No saved model found. Train first or place " + str(MODEL_PATH), 400)
    return BUNDLE, None


def _predict_proba_df(df: pd.DataFrame) -> np.ndarray:
    bundle, err = _ensure_model_loaded()
    if err:
        raise RuntimeError(err[0])
    calibrated: CalibratedClassifierCV = bundle["model"]
    # quick fitted check
    try:
        check_is_fitted(calibrated)
    except Exception:
        pass
    return calibrated.predict_proba(df[bundle["features"]])[:, 1]


@app.route("/predict", methods=["POST"])
def predict_form():
    bundle, err = _ensure_model_loaded()
    if err:
        return err
    try:
        row = {f: float(request.form[f]) for f in bundle["features"]}
        thr = request.form.get("threshold")
        thr = float(thr) if thr not in (None, "", "auto") else float(bundle["threshold"])
    except Exception as e:
        return f"Invalid inputs: {e}", 400

    X = pd.DataFrame([row])
    p = _predict_proba_df(X)[0]
    pred = int(p >= thr)
    return jsonify({
        "probability": float(p),
        "prediction": int(pred),
        "threshold": float(thr),
        "features": row,
    })


@app.route("/predict_json", methods=["POST"])
def predict_json():
    bundle, err = _ensure_model_loaded()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    feats = data.get("features", {})
    thr = data.get("threshold", "auto")
    try:
        row = {f: float(feats[f]) for f in bundle["features"]}
        thr = float(thr) if (isinstance(thr, (int, float)) or (isinstance(thr, str) and thr not in ("", "auto"))) else float(bundle["threshold"])
    except Exception as e:
        return jsonify({"error": f"Invalid inputs: {e}"}), 400

    X = pd.DataFrame([row])
    p = _predict_proba_df(X)[0]
    pred = int(p >= thr)
    return jsonify({
        "probability": float(p),
        "prediction": int(pred),
        "threshold": float(thr),
        "features": row,
    })


@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    bundle, err = _ensure_model_loaded()
    if err:
        return err
    file = request.files.get("csv")
    if not file:
        return "CSV file is required", 400
    thr_param = request.form.get("threshold", "auto")
    try:
        thr = float(thr_param) if thr_param not in (None, "", "auto") else float(bundle["threshold"])
    except Exception:
        return "Invalid threshold", 400

    df = pd.read_csv(file)
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        return f"Missing columns: {missing}", 400

    proba = _predict_proba_df(df)
    pred = (proba >= thr).astype(int)

    out = df.copy()
    out["probability"] = proba
    out["prediction"] = pred

    # Return CSV inline
    csv = out.to_csv(index=False)
    return (csv, 200, {"Content-Type": "text/csv"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
