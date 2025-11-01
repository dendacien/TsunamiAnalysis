import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier
import RetriveCV

df = RetriveCV.retrieve_earthquake_data()

TARGET = "tsunami"
FEATS = ['magnitude','cdi','mmi','sig','nst','dmin','gap','depth','latitude','longitude']

X = df[FEATS].copy()
y = df[TARGET].astype(int).to_numpy()

# spatial groups to avoid location bias in CV
k = 30 # chosen from TAPickK.py
geo_group = KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(df[['latitude','longitude']])

# small transforms for skewed positives
def log1p_selected(X_df):
    X_df = X_df.copy()
    for col in ("dmin", "gap", "depth"):
        if col in X_df:
            vals = X_df[col].to_numpy()
            shift = 1 - vals.min() if vals.min() <= 0 else 0.0
            X_df[col] = np.log1p(vals + shift)
    return X_df

log_tx = FunctionTransformer(log1p_selected, feature_names_out="one-to-one")

# --- baselines ---
dummy = DummyClassifier(strategy="most_frequent")

logit_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("log", log_tx),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1))
])

hgb_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("log", log_tx),
    ("clf", HistGradientBoostingClassifier(random_state=42))
])

rf_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("log", log_tx),
    ("clf", RandomForestClassifier(n_jobs=-1, random_state=42))
])

# class weights for HGBT via sample_weight since it doesn't have class_weight=
classes = np.unique(y)
cw = compute_class_weight("balanced", classes=classes, y=y)
class_to_w = {c:w for c,w in zip(classes, cw)}
sample_weight = np.array([class_to_w[yi] for yi in y])

# ---spatial groups + stratification for CV ---
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fold_id = np.empty(len(X), dtype=int)
for i, (_, test_idx) in enumerate(cv.split(X, y, geo_group)):
    fold_id[test_idx] = i

predef = PredefinedSplit(fold_id)  

# --- quick baseline sanity ---
for name, est, use_sw in [
    ("Dummy", dummy, False),
    ("Logit", logit_pipe, False),
    ("RF", rf_pipe, False),
    ("HGBT", hgb_pipe, True),
]:
    if use_sw:
        est.fit(X, y, clf__sample_weight=sample_weight)
    else:
        est.fit(X, y)

# --- tuning spaces ---
hgb_space = {
    "clf__learning_rate": [0.03, 0.06, 0.1],
    "clf__max_leaf_nodes": [15, 31, 63],
    "clf__max_depth": [None, 6],
    "clf__min_samples_leaf": [15, 30, 60],
    "clf__l2_regularization": [0.0, 0.5, 1.0],
}

rf_space = {
    "clf__n_estimators": [400, 700, 1000],
    "clf__max_depth": [None, 8, 12],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", 0.4, 0.6],
    "clf__class_weight": [None, "balanced"],
}

# --- hyperparameter searches ---
hgb_search = RandomizedSearchCV(
    hgb_pipe, hgb_space, n_iter=24, scoring="average_precision",
    cv=cv, refit=True, random_state=42, n_jobs=-1, verbose=1
)
hgb_search.fit(X, y, groups=geo_group, clf__sample_weight=sample_weight)

rf_search = RandomizedSearchCV(
    rf_pipe, rf_space, n_iter=24, scoring="average_precision",
    cv=cv, refit=True, random_state=42, n_jobs=-1, verbose=1
)
rf_search.fit(X, y, groups=geo_group)

# --- select best model and calibrate ---
# pick the winner by CV PR-AUC
cands = [
    ("HGBT", hgb_search.best_estimator_, hgb_search.best_score_),
    ("RF", rf_search.best_estimator_, rf_search.best_score_),
]
cands.sort(key=lambda x: x[2], reverse=True)
best_name, best_est, best_cv_ap = cands[0]

# calibrate the best model with isotonic regression
calibrated = CalibratedClassifierCV(best_est, method="isotonic", cv=predef)
calibrated.fit(X, y)

# in-sample evaluation
proba = calibrated.predict_proba(X)[:, 1]
print("Best model:", best_name, "CV PR-AUC:", round(best_cv_ap, 3))
print("In-sample PR-AUC:", round(average_precision_score(y, proba), 3))
print("In-sample ROC-AUC:", round(roc_auc_score(y, proba), 3))
