import numpy as np
from sklearn.model_selection import (
    StratifiedGroupKFold, RandomizedSearchCV, PredefinedSplit
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
from tsunami_utils import log1p_selected

# -----------------------------
# Data preparation
# -----------------------------
# Retrieve data
df = RetriveCV.retrieve_earthquake_data()

# Target and features
TARGET = "tsunami" # Binary target: 1 if tsunami occurred, else 0
FEATS = ['magnitude','nst','dmin','depth','latitude','longitude']

# Create geographic groups by binning lat/lon
def make_geo_groups(df, lat_col='latitude', lon_col='longitude', deg=5):
    lat = df[lat_col].to_numpy()
    lon = df[lon_col].to_numpy()
    lat_bin = np.floor((lat + 90) / deg).astype(int)
    lon_bin = np.floor((lon + 180) / deg).astype(int)
    return (lat_bin * 1000 + lon_bin)  # single integer group id

log_tx = FunctionTransformer(log1p_selected, feature_names_out="one-to-one")

X = df[FEATS].copy()
y = df[TARGET].astype(int).to_numpy()

groups_all = make_geo_groups(df, deg=5)

def grouped_stratified_train_test_split(X, y, groups, test_size=0.2, random_state=42):
    n_splits = int(round(1.0 / test_size))
    n_splits = max(2, n_splits)  # at least 2 folds

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # take the first fold as test; the rest as train
    train_idx, test_idx = None, None
    for fold_id, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        train_idx, test_idx = tr, te
        break
    return train_idx, test_idx

train_idx, test_idx = grouped_stratified_train_test_split(
    X, y, groups_all, test_size=0.2, random_state=42
)

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

# Graphical reports

# --- Unified subplot grid ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Training confusion matrix
ax1 = axes[0, 0]
im1 = ax1.imshow(cm_train, interpolation="nearest")
ax1.set_title("Confusion Matrix (Train)")
fig.colorbar(im1, ax=ax1)
tick_marks = np.arange(2)
ax1.set_xticks(tick_marks)
ax1.set_xticklabels(["False", "True"])
ax1.set_yticks(tick_marks)
ax1.set_yticklabels(["False", "True"])
thresh1 = cm_train.max() / 2.0
for i in range(cm_train.shape[0]):
    for j in range(cm_train.shape[1]):
        ax1.text(j, i, format(cm_train[i, j], "d"),
                 ha="center", va="center")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("True")

# 2. Test confusion matrix
ax2 = axes[0, 1]
im2 = ax2.imshow(cm, interpolation="nearest")
ax2.set_title("Confusion Matrix (Test)")
fig.colorbar(im2, ax=ax2)
ax2.set_xticks(tick_marks)
ax2.set_xticklabels(["False", "True"])
ax2.set_yticks(tick_marks)
ax2.set_yticklabels(["False", "True"])
thresh6 = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax2.text(j, i, format(cm[i, j], "d"),
                 ha="center", va="center")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("True")

# 3. ROC curve (Test)
fpr, tpr, _ = roc_curve(y_test, proba_test)
roc_auc = auc(fpr, tpr)
ax3 = axes[0, 2]
ax3.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
ax3.plot([0,1], [0,1], linestyle="--")
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.set_title("ROC (Test)")
ax3.legend()

# 4. Threshold diagnostics (Train)
prec_tr, rec_tr, thr_tr = precision_recall_curve(y_train, proba_train)
ax4 = axes[1, 0]
ax4.plot(thr_tr, prec_tr[:-1], label="Precision")
ax4.plot(thr_tr, rec_tr[:-1], label="Recall")
ax4.axvline(threshold, linestyle="--", label=f"Chosen thr={threshold:.2f}")
ax4.set_xlabel("Threshold")
ax4.set_ylabel("Score")
ax4.set_title("Precision/Recall vs Threshold (Train)")
ax4.legend()

# 5. Calibration (reliability) curve (Test)
ax5 = axes[1, 1]
CalibrationDisplay.from_predictions(y_test, proba_test, ax=ax5)
ax5.set_title("Calibration (Reliability) – Test")

# 6. Precision–Recall curve (Test)
prec, rec, thr = precision_recall_curve(y_test, proba_test)
ap = np.trapezoid(rec, prec[::-1])
ax6 = axes[1, 2]
ax6.plot(rec, prec, label=f"PR curve (AP≈{np.trapezoid(prec, rec):.3f})")
if len(thr) > 0:
    idx = np.argmin(np.abs(thr - threshold))
    ax6.scatter(rec[idx], prec[idx], s=50)
    ax6.text(rec[idx], prec[idx], f" thr={threshold:.2f}", ha="left", va="bottom")
ax6.set_xlabel("Recall")
ax6.set_ylabel("Precision")
ax6.set_title("Precision–Recall (Test)")
ax6.legend()

plt.tight_layout()
plt.show()
