"""STEP 3 - Chapter 2 (part 2): Pipelines, first models, cross-validation, fine-tuning, error analysis.
The TEST SET IS NOT TOUCHED here."""
# %% setup
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, time
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
import joblib
from homelens.data import split, xy
from homelens.features import build_preprocessing
from homelens.utils import set_style, savefig, record, rmse, PALETTE
from homelens.config import ART_DIR, RANDOM_STATE
set_style()
train, _ = split()
X, y = xy(train)

# %% preprocess_demo
prep = build_preprocessing()
Xt = prep.fit_transform(X)
print("raw shape:", X.shape, "-> transformed shape:", Xt.shape)
names = prep.get_feature_names_out()
print("\nFirst 8 of the", len(names), "output features:")
print(*names[:8], sep="\n")
print("\nAny NaN left after the pipeline?", bool(np.isnan(Xt).any()))
print("Mean/std of first 3 ratio columns (should be ~0/~1):",
      Xt[:, :3].mean(0).round(3), Xt[:, :3].std(0).round(3))

# %% cv_models
# 10-fold cross-validation RMSE for several candidate models, each wrapped with preprocessing.
def cv_rmse(model, folds=10):
    s = -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=folds, n_jobs=2)
    return s

models = {
    "Mean baseline": DummyRegressor(),
    "Linear regression": LinearRegression(),
    "Decision tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random forest (100 trees)": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=2),
}
rows = []
for name, est in models.items():
    t0 = time.time()
    pipe = make_pipeline(build_preprocessing(), est)
    folds = 5 if "forest" in name else 10
    s = cv_rmse(pipe, folds)
    train_fit = pipe.fit(X, y)
    tr = rmse(y, train_fit.predict(X))
    rows.append((name, tr, s.mean(), s.std(), time.time() - t0))
    print(f"{name:28s} train RMSE={tr:9.0f} | CV RMSE={s.mean():9.0f} (+/- {s.std():6.0f})  [{time.time()-t0:4.0f}s]")
cv_table = pd.DataFrame(rows, columns=["model", "train_rmse", "cv_rmse", "cv_std", "secs"])
record("ch2_cv_table", cv_table.round(0).to_dict("list"))

# %% finetune
# Randomised search over preprocessing AND model hyper-parameters at the same time.
full = make_pipeline(build_preprocessing(), RandomForestRegressor(n_estimators=60, random_state=RANDOM_STATE, n_jobs=2))
param_dist = {
    "columntransformer__geo__n_clusters": randint(5, 40),
    "randomforestregressor__max_features": randint(3, 16),
    "randomforestregressor__min_samples_leaf": randint(1, 6),
}
t0 = time.time()
search = RandomizedSearchCV(full, param_dist, n_iter=10, cv=3, scoring="neg_root_mean_squared_error",
                            random_state=RANDOM_STATE, n_jobs=1)
search.fit(X, y)
print(f"search took {time.time()-t0:.0f}s")
print("Best params:", search.best_params_)
print("Best CV RMSE: %.0f" % -search.best_score_)
res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
cols = ["param_columntransformer__geo__n_clusters", "param_randomforestregressor__max_features",
        "param_randomforestregressor__min_samples_leaf", "mean_test_score"]
tab = res[cols].head(6).copy()
tab.columns = ["n_clusters", "max_features", "min_samples_leaf", "CV_RMSE"]
tab["CV_RMSE"] = (-tab["CV_RMSE"]).round(0)
print(tab.to_string(index=False))
record("ch2_best_params", {k.split("__")[-1]: int(v) for k, v in search.best_params_.items()})
record("ch2_best_cv", -search.best_score_)

# %% importance
best = search.best_estimator_
rf = best.named_steps["randomforestregressor"]
feat = best.named_steps["columntransformer"].get_feature_names_out()
imp = pd.Series(rf.feature_importances_, index=feat).sort_values(ascending=False)
print("Top 12 feature importances (sum of all = %.2f):" % imp.sum())
print(imp.head(12).round(3).to_string())
fig, ax = plt.subplots(figsize=(7.5, 4.8))
imp.head(12)[::-1].plot.barh(ax=ax, color=PALETTE["blue"])
ax.set_title("Random forest: what drives the price prediction?"); ax.set_xlabel("importance")
savefig(fig, "fig06_importance_ch2")

# %% error_analysis
# Out-of-fold predictions: every district is predicted by a model that never saw it.
from sklearn.model_selection import cross_val_predict
oof = cross_val_predict(clone(best), X, y, cv=5, n_jobs=1)
err = pd.DataFrame({"actual": y, "pred": oof, "abs_err": np.abs(y - oof),
                    "ocean": X.ocean_proximity, "income": X.median_income, "lat": X.latitude, "lon": X.longitude})
print("OOF RMSE = %.0f | OOF MAE = %.0f" % (rmse(y, oof), err.abs_err.mean()))
print("\nError by ocean proximity (MAE, $):")
print(err.groupby("ocean").abs_err.agg(["count", "mean"]).round(0).to_string())
capped = err[err.actual >= 500001]
print("\nCapped districts (price = 500,001): n=%d, mean prediction=%.0f -> models cannot learn beyond the cap" %
      (len(capped), capped.pred.mean()))
record("capped_mean_pred", capped.pred.mean())
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(err.actual, err.pred, s=4, alpha=.25, color=PALETTE["blue"])
ax[0].plot([0, 5e5], [0, 5e5], color=PALETTE["red"], lw=1.5)
ax[0].set_xlabel("actual price ($)"); ax[0].set_ylabel("out-of-fold prediction ($)"); ax[0].set_title("Predicted vs. actual")
sc = ax[1].scatter(err.lon, err.lat, c=err.abs_err, s=5, cmap="magma_r", vmax=150000)
plt.colorbar(sc, ax=ax[1], label="absolute error ($)"); ax[1].set_title("Where is the model wrong?")
ax[1].set_xlabel("longitude"); ax[1].set_ylabel("latitude")
savefig(fig, "fig07_error_analysis")
joblib.dump(best, ART_DIR / "ch2_random_forest_tuned.joblib")
