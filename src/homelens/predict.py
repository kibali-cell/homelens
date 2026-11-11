"""Inference helper: load the shipped artefacts and score new districts.

Usage:  python -m homelens.predict            (scores three example districts)
"""
import joblib
import pandas as pd

from .config import ART_DIR

EXAMPLE = pd.DataFrame([
    dict(longitude=-122.23, latitude=37.88, housing_median_age=41, total_rooms=880, total_bedrooms=129,
         population=322, households=126, median_income=8.33, ocean_proximity="NEAR BAY"),
    dict(longitude=-117.10, latitude=32.75, housing_median_age=20, total_rooms=2500, total_bedrooms=None,
         population=1400, households=480, median_income=3.2, ocean_proximity="NEAR OCEAN"),
    dict(longitude=-119.80, latitude=36.75, housing_median_age=15, total_rooms=3100, total_bedrooms=620,
         population=1900, households=600, median_income=2.1, ocean_proximity="INLAND"),
])


def load():
    reg = joblib.load(ART_DIR / "homelens_regressor.joblib")
    clf = joblib.load(ART_DIR / "homelens_premium_classifier.joblib")
    return reg, clf


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Return predicted value and premium probability. Missing total_bedrooms is handled by the pipeline."""
    reg, clf = load()
    X = clf["preprocessing"].transform(df)
    X = X.toarray() if hasattr(X, "toarray") else X
    proba = clf["classifier"].predict_proba(X)[:, 1]
    return df.assign(predicted_value=reg.predict(df).round(-2), premium_probability=proba.round(3),
                     is_premium=proba >= clf["threshold"])


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    print(score(EXAMPLE)[["median_income", "ocean_proximity", "predicted_value", "premium_probability", "is_premium"]])
