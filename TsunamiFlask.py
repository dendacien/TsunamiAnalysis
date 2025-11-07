#!/usr/bin/env python3
"""
Flask app: Tsunami Prediction (LOAD-ONLY)
- This app does NOT train. It only loads a saved bundle produced by your training script
  and provides:
    • Prediction (single form / CSV batch )
    • Training/Test plots gallery (/plots)
    • Random test-row demo (/sample_test)

Expected artifact written by the trainer:
  artifacts/tsunami_model_bundle.pkl

Bundle schema (dict):
  {
    "pipeline":   sklearn.Pipeline,
    "calibrated": sklearn.calibration.CalibratedClassifierCV,  # preferred
    "features":   List[str],
    "threshold":  float,
    "best_params": dict,
    "best_cv_ap": float,
    "imgs":       {"pr","roc","cal","thresh","cm","imp"}  # base64 PNG strings
    "test_df":    pandas.DataFrame  # held-out test features + a 'tsunami' column
  }

Python 3.10+
scikit-learn==1.7.2
Flask>=3.0
matplotlib>=3.6
numpy, pandas, joblib
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

# non-interactive backend
import matplotlib
matplotlib.use("Agg")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted
import joblib

# ----------------------
# Config
# ----------------------
FEATS = [
    "magnitude","cdi","mmi","sig","nst","dmin","gap","depth","latitude","longitude"
]
TARGET = "tsunami"
ARTIFACT_DIR = os.path.dirname(__file__)
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
# Updated to reflect external training script's filename
MODEL_PATH = ARTIFACT_DIR / "tsunami_model_bundle.pkl"

# ----------------------
# HTML (no training UI)
# ----------------------
NAV_HTML = """
<nav style="margin-bottom:1.5rem;">
  <ul style="display:flex;gap:1rem;list-style:none;padding:0;">
  <h1>🌊 Tsunami Prediction<br />HGBT (scikit-learn)</h1>
    <li><a href="/">Home</a></li>
    <li><a href="/plots">Training Plots</a></li>
    <li><a href="/sample_test">Random Test Sample</a></li>
  </ul>
</nav>
"""

HTML = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tsunami Prediction (HGBT)</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css" />
  <style>
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }}
    img {{ width: 100%; height: auto; border-radius: 12px; }}
    .code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
    .muted {{ opacity: .7; }}
    form.inline-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: .6rem; }}
  </style>
</head>
<body class="container">
  <main>
    
    {NAV_HTML}

    {{% if not model_loaded %}}
      <article>
        <h3>No saved model found</h3>
        <p class="muted">Place <code>{{{{ model_path }}}}</code> here, produced by your training script, then refresh.</p>
        <p class="code">Expected keys: pipeline, calibrated, features, threshold, best_params, best_cv_ap, imgs, test_df</p>
      </article>
    {{% else %}}
      <article>
        <hgroup>
          <h3>Model bundle loaded</h3>
          <p class="muted"><code>{{{{ model_path }}}}</code></p>
        </hgroup>
        <p><strong>Saved threshold:</strong> {{{{ threshold }}}}</p>
        <p><strong>CV PR-AUC:</strong> {{{{ best_cv_ap }}}} | <strong>Best Params:</strong> <span class="code">{{{{ best_params }}}}</span></p>
      </article>

      <hr/>
      <article>
      <h2>Predict</h2>
      You can use the navigation link above to see a prediction using a random test sample from the saved test split.
      <br/>&nbsp;<br/>
      <details open>
        <summary>Single Prediction (Form)</summary>
        <form class="inline-grid" method="POST" action="/predict">
          {{% for f in features %}}
            <label>{{{{ f }}}}
              <input type="number" step="any" name="{{{{ f }}}}" required />
            </label>
          {{% endfor %}}
          <label>Override threshold
            <input type="number" step="0.001" min="0" max="1" name="threshold" placeholder="auto" />
          </label>
          <button type="submit">Predict</button>
        </form>
      </details>

      <details>
        <summary>Batch Prediction (CSV)</summary>
        <form method="POST" action="/predict_csv" enctype="multipart/form-data">
          <label>CSV columns: <span class="code">{{{{ features|join(', ') }}}}</span>
            <input type="file" name="csv" accept=".csv" required />
          </label>
          <label>Override threshold (optional)
            <input type="number" step="0.001" min="0" max="1" name="threshold" placeholder="auto" />
          </label>
          <button type="submit">Predict CSV</button>
        </form>
      </details>
        </article>
    {{% endif %}}
  </main>
</body>
</html>
"""

# ----------------------
# Load & helpers
# ----------------------

app = Flask(__name__)

# Load model bundle at startup
try:
  if MODEL_PATH.exists():
    BUNDLE = joblib.load(MODEL_PATH)
    # Normalize old/new schema keys for estimator access
    if "calibrated" not in BUNDLE and "model" in BUNDLE:
      BUNDLE["calibrated"] = BUNDLE["model"]
    print("Model bundle loaded at startup.")
  else:
    BUNDLE = None
    print(f"Model file not found at startup: {MODEL_PATH}")
except Exception as e:
  BUNDLE = None
  print(f"Error loading model bundle at startup: {e}")

def _ensure_model_loaded():
  global BUNDLE
  if BUNDLE is None:
    return None, ("No saved model found. Place artifacts/tsunami_model_bundle.pkl and refresh.", 400)
  return BUNDLE, None


def _predict_proba_df(df: pd.DataFrame) -> np.ndarray:
    bundle, err = _ensure_model_loaded()
    if err:
        raise RuntimeError(err[0])
    calibrated: CalibratedClassifierCV = bundle["calibrated"]
    try:
        check_is_fitted(calibrated)
    except Exception:
        pass
    return calibrated.predict_proba(df[bundle["features"]])[:, 1]



# ----------------------
# Routes (no training)
# ----------------------
@dataclass
class ModelMeta:
    best_cv_ap: str
    best_params: Dict
    threshold: str


@app.route("/", methods=["GET"])
def index():
  model_loaded = BUNDLE is not None
  meta = None
  if model_loaded:
    meta = ModelMeta(
      best_cv_ap=f"{BUNDLE.get('best_cv_ap', float('nan')):.3f}" if isinstance(BUNDLE.get('best_cv_ap', None), (int,float)) else str(BUNDLE.get('best_cv_ap')), 
      best_params=BUNDLE.get("best_params", {}),
      threshold=f"{BUNDLE.get('threshold', 0.5):.3f}",
    )
  return render_template_string(
    HTML,
    model_loaded=model_loaded,
    model_path=str(MODEL_PATH),
    features=(BUNDLE.get("features") if model_loaded else FEATS),
    threshold=(meta.threshold if model_loaded else "—"),
    best_cv_ap=(meta.best_cv_ap if model_loaded else "—"),
    best_params=(meta.best_params if model_loaded else {}),
  )


@app.route("/plots", methods=["GET"])
def show_plots():
    bundle, err = _ensure_model_loaded()
    if err:
        return err
    imgs = bundle.get("imgs")
    if not imgs:
        return "No plots available in the bundle. Re-run the training script to save them.", 400
    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Tsunami Prediction (HGBT — Load Only)</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css" />
    <style>
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }}
        img.plot-thumb {{ width: 100%; height: auto; border-radius: 12px; cursor: pointer; transition: box-shadow .2s; }}
        img.plot-thumb:hover {{ box-shadow: 0 0 0 4px #0099ff44; }}
        .modal-bg {{ display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.8); z-index:1000; align-items:center; justify-content:center; }}
        .modal-bg.active {{ display:flex; }}
        .modal-img {{ max-width:90vw; max-height:90vh; border-radius:16px; box-shadow:0 0 32px #000; background:#fff; }}
        .modal-close {{ position:absolute; top:2rem; right:2rem; font-size:2rem; color:#fff; background:none; border:none; cursor:pointer; }}
    </style>
    </head>
    <body class="container">
    <main>
    {NAV_HTML}
    <h2>Training & Test Plots</h2>
    <div class="grid">
      <figure><img class="plot-thumb" src="data:image/png;base64,{{{{ cmtrain }}}}" alt="Train Confusion Matrix" onclick="showModal(this.src)"></figure>
      <figure><img class="plot-thumb" src="data:image/png;base64,{{{{ cmtest }}}}" alt="Test Confusion Matrix" onclick="showModal(this.src)"></figure>
      <figure><img class="plot-thumb" src="data:image/png;base64,{{{{ pr }}}}" alt="Precision-Recall" onclick="showModal(this.src)"></figure>
      <figure><img class="plot-thumb" src="data:image/png;base64,{{{{ roc }}}}" alt="ROC" onclick="showModal(this.src)"></figure>
      <figure><img class="plot-thumb" src="data:image/png;base64,{{{{ cal }}}}" alt="Calibration" onclick="showModal(this.src)"></figure>
      <figure><img class="plot-thumb" src="data:image/png;base64,{{{{ thresh }}}}" alt="Threshold" onclick="showModal(this.src)"></figure>
      {{% if imp %}}
      <figure><img class="plot-thumb" src="data:image/png;base64,{{{{ imp }}}}" alt="Permutation Importance" onclick="showModal(this.src)"></figure>
      {{% endif %}}
    </div>
    <div id="modalBg" class="modal-bg" onclick="hideModal(event)">
      <button class="modal-close" onclick="hideModal(event)">&times;</button>
      <img id="modalImg" class="modal-img" src="" alt="Full Size Plot" />
    </div>
    <script>
      function showModal(src) {{
        document.getElementById('modalImg').src = src;
        document.getElementById('modalBg').classList.add('active');
      }}
      function hideModal(e) {{
        if (e.target.id === 'modalBg' || e.target.classList.contains('modal-close')) {{
          document.getElementById('modalBg').classList.remove('active');
          document.getElementById('modalImg').src = '';
        }}
      }}
    </script>
    <p><a href="/">Back</a></p>
    """
    return render_template_string(html, **imgs)


@app.route("/sample_test", methods=["GET"]) 
def sample_test():
    bundle, err = _ensure_model_loaded()
    if err:
        return err
    test_df: pd.DataFrame | None = bundle.get("test_df")
    if test_df is None or test_df.empty:
        return "No stored test split in the bundle. Re-run the training script to save it.", 400
    row = test_df.sample(n=1, random_state=np.random.randint(0, 1_000_000)).iloc[0]
    feats = {f: float(row[f]) for f in bundle["features"]}
    thr = float(bundle.get("threshold", 0.5))
    X = pd.DataFrame([feats])
    p = _predict_proba_df(X)[0]
    pred = int(p >= thr)
    truth = int(row.get(TARGET, -1))

    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Tsunami Prediction (HGBT — Load Only)</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css" />
    <style>
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }}
        img {{ width: 100%; height: auto; border-radius: 12px; }}
        .code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
        .muted {{ opacity: .7; }}
        form.inline-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: .6rem; }}
    </style>
    </head>
    <body class="container">
    <main>
    {NAV_HTML}
    <h2>Prediction on Random Test Sample</h2>
    <p>This sample is data drawn randomly from the held-out test split stored in the model bundle and run through the model for prediction.</p>
    <p><strong>Prediction:</strong> {{{{ pred }}}} (probability: {{{{ prob }}}}) | <strong>Grounded truth:</strong> {{{{ truth }}}}</p>
    <p>0 is no tsunami, 1 is tsunami.</p>
    <p></p>
    <p><strong>Saved threshold:</strong> {{{{ threshold }}}}</p>
    <table>
      <thead><tr><th>Feature</th><th>Value</th></tr></thead>
      <tbody>
        {{% for k,v in feats.items() %}}
          <tr><td>{{{{ k }}}}</td><td>{{{{ v }}}}</td></tr>
        {{% endfor %}}
      </tbody>
    </table>
    <p><a href="/">Back</a></p>
    """
    return render_template_string(html, threshold=thr, prob=f"{p:.3f}", pred=pred, truth=truth, feats=feats)


# ---------- Prediction endpoints ----------
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
    return jsonify({"probability": float(p), "prediction": int(pred), "threshold": float(thr), "features": row})


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
    return jsonify({"probability": float(p), "prediction": int(pred), "threshold": float(thr), "features": row})


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
    out = df.copy(); out["probability"] = proba; out["prediction"] = pred
    csv = out.to_csv(index=False)
    return (csv, 200, {"Content-Type": "text/csv"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
