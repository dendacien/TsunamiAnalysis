import numpy as np

def log1p_selected(X_df):
    X_df = X_df.copy()
    for col in ("dmin","gap","depth"):
        if col in X_df:
            vals = X_df[col].to_numpy()
            shift = 1 - vals.min() if vals.min() <= 0 else 0.0
            X_df[col] = np.log1p(vals + shift)
    return X_df
