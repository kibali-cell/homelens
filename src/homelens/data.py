"""Data access: loading, the stratified train/test split, and label construction."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, N_TIERS, PREMIUM_QUANTILE, RANDOM_STATE, TARGET, TEST_SIZE


def load_raw() -> pd.DataFrame:
    """Read the raw CSV exactly as delivered (no cleaning)."""
    return pd.read_csv(DATA_PATH)


def add_income_cat(df: pd.DataFrame) -> pd.Series:
    """Income category used ONLY to stratify the split (income is the strongest predictor)."""
    return pd.cut(df["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                  labels=[1, 2, 3, 4, 5])


def split(df: pd.DataFrame | None = None):
    """Stratified 80/20 split. The test set is locked away until the very last step."""
    df = load_raw() if df is None else df
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                   stratify=add_income_cat(df))
    return train.reset_index(drop=True), test.reset_index(drop=True)


def xy(df: pd.DataFrame):
    """Separate predictors from the regression target."""
    return df.drop(columns=[TARGET]), df[TARGET].copy()


def premium_threshold(train_values: pd.Series) -> float:
    """Price above which a district is 'premium'. Computed on TRAIN only (no leakage)."""
    return float(np.quantile(train_values, PREMIUM_QUANTILE))


def tier_edges(train_values: pd.Series) -> np.ndarray:
    """Quartile edges of the training target, used to build 4 price tiers."""
    return np.quantile(train_values, np.linspace(0, 1, N_TIERS + 1)[1:-1])


def make_premium(values: pd.Series, thr: float) -> np.ndarray:
    return (values.to_numpy() >= thr).astype(int)


def make_tiers(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values.to_numpy(), edges)   # 0=budget, 1=lower-mid, 2=upper-mid, 3=premium
