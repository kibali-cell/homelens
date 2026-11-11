"""Run with:  pytest -q   (from the project root)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from homelens.data import add_income_cat, load_raw, make_premium, premium_threshold, split, xy
from homelens.features import ClusterSimilarity, build_preprocessing


def test_split_is_stratified_and_disjoint():
    train, test = split()
    assert len(train) + len(test) == 20640
    p_all = add_income_cat(load_raw()).value_counts(normalize=True).sort_index()
    p_te = add_income_cat(test).value_counts(normalize=True).sort_index()
    assert np.abs(p_all - p_te).max() < 0.005          # stratification keeps the income mix


def test_split_is_deterministic():
    a, _ = split(); b, _ = split()
    assert a.equals(b)


def test_preprocessing_has_no_nans_and_is_scaled():
    train, _ = split(); X, _ = xy(train)
    Xt = build_preprocessing().fit_transform(X)
    assert not np.isnan(Xt).any()
    assert np.allclose(Xt[:, :3].mean(0), 0, atol=1e-6)


def test_preprocessing_handles_unseen_category():
    train, _ = split(); X, _ = xy(train)
    prep = build_preprocessing().fit(X)
    new = X.head(3).copy(); new["ocean_proximity"] = "MARS"
    assert prep.transform(new).shape[0] == 3


def test_cluster_similarity_shape_and_range():
    train, _ = split()
    cs = ClusterSimilarity(n_clusters=7).fit(train[["latitude", "longitude"]])
    out = cs.transform(train[["latitude", "longitude"]].head(50))
    assert out.shape == (50, 7) and out.min() >= 0 and out.max() <= 1


def test_premium_label_uses_train_threshold_only():
    train, test = split()
    thr = premium_threshold(train["median_house_value"])
    assert abs(make_premium(train["median_house_value"], thr).mean() - 0.25) < 0.01
