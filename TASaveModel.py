import numpy as np, pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold, RandomizedSearchCV, PredefinedSplit
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    f1_score, precision_score, recall_score, confusion_matrix,
    roc_curve, auc
)
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import RetriveCV
import joblib
import base64
import io
import os
from typing import List

# -----------------------------
# Data preparation
# -----------------------------
# Retrieve data
df = RetriveCV.retrieve_earthquake_data()

# Target and features
TARGET = "tsunami" # Binary target: 1 if tsunami occurred, else 0
FEATS = ['magnitude','cdi','mmi','sig','nst','dmin','gap','depth','latitude','longitude']

# Create geographic groups by binning lat/lon
def make_geo_groups(df, lat_col='latitude', lon_col='longitude', deg=5):
    lat = df[lat_col].to_numpy()
    lon = df[lon_col].to_numpy()
    lat_bin = np.floor((lat + 90) / deg).astype(int)
    lon_bin = np.floor((lon + 180) / deg).astype(int)
    return (lat_bin * 1000 + lon_bin)  # single integer group id

# Small transforms for skewed positives
def log1p_selected(X_df):
    X_df = X_df.copy()
    for col in ("dmin","gap","depth"):
        if col in X_df:
            vals = X_df[col].to_numpy()
            shift = 1 - vals.min() if vals.min() <= 0 else 0.0
            X_df[col] = np.log1p(vals + shift)
    return X_df

log_tx = FunctionTransformer(log1p_selected, feature_names_out="one-to-one")

X = df[FEATS].copy()
y = df[TARGET].astype(int).to_numpy()

groups_all = make_geo_groups(df, deg=5)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups_all))

X_train, y_train = X.iloc[train_idx], y[train_idx]
X_test,  y_test  = X.iloc[test_idx],  y[test_idx]
groups_train = groups_all[train_idx]
groups_test  = groups_all[test_idx]

print("Train size:", len(train_idx), "Test size:", len(test_idx))

# -----------------------------
# Model tuning and training
# -----------------------------
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("log", log_tx),
    ("clf", HistGradientBoostingClassifier(random_state=42))
])
pipe.set_output(transform="pandas")

param_space = {
    "clf__learning_rate": [0.03, 0.06, 0.1],
    "clf__max_leaf_nodes": [15, 31, 63],
    "clf__max_depth": [None, 6],
    "clf__min_samples_leaf": [15, 30, 60],
    "clf__l2_regularization": [0.0, 0.5, 1.0],
}

# grouped CV on the *training* portion only
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_space,
    n_iter=24,
    scoring="average_precision",   # PR-AUC
    cv=sgkf,
    n_jobs=-1,
    refit=True,
    verbose=1,
    random_state=42,
)

# perform hyperparameter search
search.fit(X_train, y_train, groups=groups_train)
best_pipe = search.best_estimator_
print("Best CV PR-AUC:", round(search.best_score_, 3))
print("Best params:", search.best_params_)

# Build fold ids once from the same grouped splitter
fold_id = np.empty(len(X_train), dtype=int)
for i, (_, val_idx) in enumerate(sgkf.split(X_train, y_train, groups_train)):
    fold_id[val_idx] = i
predef = PredefinedSplit(fold_id)

# -----------------------------
# Calibration, evaluation, and reporting
# -----------------------------
calibrated = CalibratedClassifierCV(best_pipe, method="isotonic", cv=predef)
calibrated.fit(X_train, y_train)

proba_train = calibrated.predict_proba(X_train)[:, 1]
prec, rec, thr = precision_recall_curve(y_train, proba_train)
f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
threshold = thr[np.nanargmax(f1s)]

proba_test = calibrated.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= threshold).astype(int)
pred_train = (proba_train >= threshold).astype(int)

pr_auc  = average_precision_score(y_test, proba_test)
roc_auc = roc_auc_score(y_test, proba_test)
f1      = f1_score(y_test, pred_test)
prec_   = precision_score(y_test, pred_test, zero_division=0)
rec_    = recall_score(y_test, pred_test)
cm      = confusion_matrix(y_test, pred_test)
cm_train = confusion_matrix(y_train, pred_train)

print("TRAIN Confusion matrix [tn fp; fn tp]:\n", cm_train)
print("TEST Confusion matrix [tn fp; fn tp]:\n", cm)
print(f"TEST  PR-AUC: {pr_auc:.3f}")
print(f"TEST ROC-AUC: {roc_auc:.3f}")
print(f"TEST  Precision: {prec_:.3f}  Recall: {rec_:.3f}  F1: {f1:.3f}")
print("Using threshold:", round(threshold, 4))

# --- Build images for Flask “/plots” (base64 PNG strings) ---
def _fig_to_b64():
    import io, base64, matplotlib.pyplot as plt
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

imgs = {}

# Confusion matrix
cm = confusion_matrix(y_train, pred_train)
plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix (Train)"); plt.colorbar()
ticks = np.arange(2); plt.xticks(ticks, ["True","False"]); plt.yticks(ticks, ["True","False"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.xlabel("Predicted"); plt.ylabel("True")
imgs["cmtrain"] = _fig_to_b64()

# Confusion matrix
cm = confusion_matrix(y_test, pred_test)
plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix (Test)"); plt.colorbar()
ticks = np.arange(2); plt.xticks(ticks, ["True","False"]); plt.yticks(ticks, ["True","False"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.xlabel("Predicted"); plt.ylabel("True")
imgs["cmtest"] = _fig_to_b64()

# PR curve
prec, rec, thr = precision_recall_curve(y_test, proba_test)
plt.figure(); plt.plot(rec, prec, label="PR curve")
if len(thr) > 0:
    import numpy as np
    idx = np.argmin(np.abs(thr - threshold))
    plt.scatter(rec[idx], prec[idx]); plt.text(rec[idx], prec[idx], f" thr={threshold:.2f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall (Test)"); plt.legend()
imgs["pr"] = _fig_to_b64()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, proba_test)
plt.figure(); plt.plot(fpr, tpr, label=f"ROC AUC={auc(fpr, tpr):.3f}"); plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC (Test)"); plt.legend()
imgs["roc"] = _fig_to_b64()

# Threshold diagnostics (from train)
prec_tr, rec_tr, thr_tr = precision_recall_curve(y_train, proba_train)
plt.figure(); plt.plot(thr_tr, prec_tr[:-1], label="Precision"); plt.plot(thr_tr, rec_tr[:-1], label="Recall")
plt.axvline(threshold, linestyle="--", label=f"thr={threshold:.2f}")
plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title("Precision/Recall vs Threshold (Train)"); plt.legend()
imgs["thresh"] = _fig_to_b64()

# Simple calibration curve (test)
bins = np.linspace(0,1,11); inds = np.digitize(proba_test, bins) - 1; centers = 0.5*(bins[1:]+bins[:-1])
frac_pos = [np.mean(y_test[inds==i]) if np.any(inds==i) else np.nan for i in range(len(centers))]
plt.figure(); plt.plot([0,1],[0,1],"--",label="Perfect"); plt.plot(centers, frac_pos, "o-", label="Empirical")
plt.xlabel("Pred prob bin"); plt.ylabel("Observed frac positive"); plt.title("Calibration (Test)"); plt.legend()
imgs["cal"] = _fig_to_b64()



# Permutation importance (on test)
from sklearn.inspection import permutation_importance
result = permutation_importance(best_pipe, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
order = np.argsort(result.importances_mean)
plt.figure(figsize=(8, max(3, len(FEATS)*0.35)))
plt.barh(np.array(FEATS)[order], result.importances_mean[order])
plt.xlabel("Mean decrease in score (perm)"); plt.title("Permutation Importance (Test)"); plt.tight_layout()
imgs["imp"] = _fig_to_b64()

# --- Save bundle exactly as the Flask app expects ---
bundle = {
    "pipeline":   best_pipe,          # sklearn Pipeline
    "calibrated": calibrated,         # CalibratedClassifierCV (preferred by app)
    "features":   FEATS,              # list[str]
    "threshold":  float(threshold),   # chosen operating threshold
    "best_params": search.best_params_,
    "best_cv_ap": float(search.best_score_),
    "imgs":       imgs,               # base64 images for /plots
    "test_df":    pd.concat([X_test.reset_index(drop=True),
                             pd.Series(y_test, name=TARGET)], axis=1),
}

os.makedirs("artifacts", exist_ok=True)
joblib.dump(bundle, "artifacts/tsunami_model_bundle.pkl")
print("Saved -> artifacts/tsunami_model_bundle.pkl")

