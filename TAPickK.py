import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold
import RetriveCV

df = RetriveCV.retrieve_earthquake_data()

def choose_k_for_geo_groups(df, ks=(24, 30, 36, 42, 48), n_splits=5):
    best_k, best_score = None, -np.inf
    y = df["tsunami"].to_numpy()
    latlon = df[["latitude","longitude"]].to_numpy()
    for k in ks:
        groups = KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(latlon)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        pos_counts, neg_counts = [], []
        for tr, te in cv.split(df, y, groups):
            pos_counts.append(y[te].sum())
            neg_counts.append((1-y[te]).sum())
        # score: prefer folds with decent positives and low variance across folds
        score = (np.min(pos_counts)) - np.std(pos_counts)
        if score > best_score:
            best_k, best_score = k, score
    return best_k

print("Choosing k for geographic grouping...")
k_opt = choose_k_for_geo_groups(df)
print(f"Optimal k for geographic grouping: {k_opt}")

