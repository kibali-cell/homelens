"""Ready-made arrays for chapters 3-9 (classification, SVMs, trees, ensembles, PCA, clustering)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import make_premium, make_tiers, premium_threshold, split, tier_edges, xy
from .features import add_ratio_columns, simple_features_pipeline


@dataclass
class Task:
    train: pd.DataFrame           # raw training rows (with target)
    test: pd.DataFrame            # locked-away rows
    X_raw: pd.DataFrame           # raw predictors (train)
    y_value: pd.Series            # regression target
    y_premium: np.ndarray         # 1 if value >= 75th percentile of train
    y_tier: np.ndarray            # 0..3 quartile tier
    thr: float
    edges: np.ndarray
    X: np.ndarray                 # preprocessed predictors (train), dense float array
    feature_names: list
    pipe: object                  # the fitted preprocessing pipeline


def load_task() -> Task:
    train, test = split()
    X_raw, y_value = xy(train)
    thr, edges = premium_threshold(y_value), tier_edges(y_value)
    pipe = simple_features_pipeline()
    X = pipe.fit_transform(X_raw)
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    names = list(pipe.get_feature_names_out())
    return Task(train, test, X_raw, y_value, make_premium(y_value, thr), make_tiers(y_value, edges),
                thr, edges, X, names, pipe)
