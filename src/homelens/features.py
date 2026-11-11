"""Feature engineering and preprocessing pipelines (scikit-learn compatible)."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .config import RANDOM_STATE


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """Fit k-means on (lat, lon); output RBF similarity of each row to each cluster centre.

    This is a *custom transformer*: it learns something in fit() (the cluster centres)
    and applies it in transform(). Because it lives inside the Pipeline it is re-fit on
    the training fold only during cross-validation, so it cannot leak information.
    """

    def __init__(self, n_clusters=10, gamma=1.0, random_state=RANDOM_STATE):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        check_is_fitted(self, "kmeans_")
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"geo_cluster_{i}_sim" for i in range(self.n_clusters)]


# ---- ratio features -----------------------------------------------------------------
def _column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def _ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(_column_ratio, feature_names_out=_ratio_name),
        StandardScaler(),
    )


def log_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler(),
    )


def build_preprocessing(n_clusters=10, gamma=1.0, use_clusters=True) -> ColumnTransformer:
    """Full preprocessing used by the Chapter 2 models.

    * rooms/house, bedroom ratio, people/house  -> impute, ratio, scale
    * heavy-tailed counts + income              -> impute, log, scale
    * latitude/longitude                        -> k-means similarity features
    * ocean_proximity                           -> one-hot
    * everything else (housing_median_age)      -> impute, scale
    """
    transformers = [
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline(), ["total_bedrooms", "total_rooms", "population",
                                 "households", "median_income"]),
    ]
    if use_clusters:
        transformers.append(("geo", ClusterSimilarity(n_clusters, gamma),
                             ["latitude", "longitude"]))
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"),
                         make_column_selector(dtype_exclude="number")))
    default = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    return ColumnTransformer(transformers, remainder=default, verbose_feature_names_out=True)


def build_simple_preprocessing() -> ColumnTransformer:
    """Lightweight preprocessing for the 'algorithm' chapters (3-9): impute, ratios kept raw,
    one-hot the category, scale everything. No clustering features (keeps models transparent)."""
    num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    return ColumnTransformer(
        [("num", num, make_column_selector(dtype_include="number")),
         ("cat", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_exclude="number"))]
    )


def add_ratio_columns(df):
    """Pandas-level version of the ratio features, for the lightweight pipeline."""
    out = df.copy()
    out["rooms_per_house"] = out["total_rooms"] / out["households"]
    out["bedrooms_ratio"] = out["total_bedrooms"] / out["total_rooms"]
    out["people_per_house"] = out["population"] / out["households"]
    return out


def _ratio_columns_names(transformer, names_in):
    return list(names_in) + ["rooms_per_house", "bedrooms_ratio", "people_per_house"]


def simple_features_pipeline() -> Pipeline:
    return Pipeline([("ratios", FunctionTransformer(add_ratio_columns, feature_names_out=_ratio_columns_names)),
                     ("prep", build_simple_preprocessing())])
