"""STEP 11 - Integration: choose the final models, evaluate ONCE on the locked test set, ship artefacts."""
# %% setup
import sys; sys.path.insert(0, "src")
import json, time, numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from scipy import stats
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, precision_recall_curve, precision_score, r2_score, recall_score,
                             roc_auc_score, f1_score, confusion_matrix)
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import make_pipeline
from homelens.config import ART_DIR, RANDOM_STATE, TARGET
from homelens.data import make_premium, split, xy, premium_threshold
from homelens.features import build_preprocessing, simple_features_pipeline
from homelens.utils import set_style, savefig, record, rmse, PALETTE
set_style()
train, test = split()
X_train, y_train = xy(train); X_test, y_test = xy(test)
print("train:", X_train.shape, "| test:", X_test.shape, "(touched for the first time in the evaluation block below)")

# %% tune_regressor
# Final regressor = preprocessing (with geo-cluster features) + histogram gradient boosting, tuned by randomised search.
pipe = make_pipeline(build_preprocessing(n_clusters=25), HistGradientBoostingRegressor(early_stopping=True, random_state=RANDOM_STATE))
space = {
    "columntransformer__geo__n_clusters": randint(10, 45),
    "histgradientboostingregressor__learning_rate": loguniform(0.03, 0.3),
    "histgradientboostingregressor__max_iter": randint(200, 700),
    "histgradientboostingregressor__max_leaf_nodes": randint(15, 80),
    "histgradientboostingregressor__min_samples_leaf": randint(10, 60),
    "histgradientboostingregressor__l2_regularization": loguniform(1e-3, 10),
}
t0 = time.time()
search = RandomizedSearchCV(pipe, space, n_iter=25, cv=4, scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE, n_jobs=2)
search.fit(X_train, y_train)
print("Search (25 candidates x 4 folds) took %.0fs" % (time.time() - t0))
bp = {k.split("__")[-1]: (round(float(v), 4) if isinstance(v, float) else int(v)) for k, v in search.best_params_.items()}
print("Best params:", bp)
print("Best CV RMSE on the training set: $%.0f" % -search.best_score_)
record("final_cv_rmse", -search.best_score_); record("final_params", bp)
model = search.best_estimator_

# %% evaluate_test
# THE ONE-SHOT TEST EVALUATION. No more tuning after this block.
pred = model.predict(X_test)
test_rmse, test_mae, test_r2 = rmse(y_test, pred), mean_absolute_error(y_test, pred), r2_score(y_test, pred)
print("TEST RMSE = $%.0f | MAE = $%.0f | R^2 = %.3f" % (test_rmse, test_mae, test_r2))
sq = (pred - y_test.to_numpy()) ** 2
ci = np.sqrt(stats.t.interval(0.95, len(sq) - 1, loc=sq.mean(), scale=stats.sem(sq)))
print("95%% confidence interval for the generalisation RMSE: [$%.0f, $%.0f]" % tuple(ci))
gap = test_rmse - (-search.best_score_)
print("Gap between CV estimate and test RMSE: %+.0f (%.1f%%) -> small gap = trustworthy validation scheme" % (gap, 100 * gap / test_rmse))
dummy = rmse(y_test, np.full(len(y_test), y_train.mean()))
print("Baseline (predict the training mean) test RMSE = $%.0f -> our model removes %.0f%% of the error" % (dummy, 100 * (1 - test_rmse / dummy)))
for k, v in dict(test_rmse=test_rmse, test_mae=test_mae, test_r2=test_r2, test_ci_low=ci[0], test_ci_high=ci[1], test_baseline_rmse=dummy).items(): record(k, v)

# %% segment_analysis
seg = pd.DataFrame({"actual": y_test.to_numpy(), "pred": pred, "ocean": X_test.ocean_proximity.to_numpy(), "income": X_test.median_income.to_numpy()})
seg["abs_err"] = (seg.actual - seg.pred).abs(); seg["pct_err"] = 100 * seg.abs_err / seg.actual
seg["income_band"] = pd.cut(seg.income, [0, 2.5, 4, 6, 20], labels=["low", "mid", "upper", "high"])
seg["capped"] = seg.actual >= 500001
print("Error by ocean proximity:")
print(seg.groupby("ocean").agg(n=("abs_err", "size"), MAE=("abs_err", "mean"), MAPE_pct=("pct_err", "mean")).round(1).to_string())
print("\nError by income band:")
print(seg.groupby("income_band", observed=True).agg(n=("abs_err", "size"), MAE=("abs_err", "mean"), MAPE_pct=("pct_err", "mean")).round(1).to_string())
print("\nCapped districts (true value hidden above $500k): n=%d, MAE=%.0f vs %.0f for the rest" %
      (seg.capped.sum(), seg[seg.capped].abs_err.mean(), seg[~seg.capped].abs_err.mean()))
within = (seg.pct_err <= 20).mean() * 100
print("Share of test districts predicted within +/-20%% of the truth: %.1f%%" % within)
record("within20", within)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(seg.actual, seg.pred, s=4, alpha=.3, color=PALETTE["blue"]); ax[0].plot([0, 5e5], [0, 5e5], color=PALETTE["red"]); ax[0].set_xlabel("actual ($)"); ax[0].set_ylabel("predicted ($)"); ax[0].set_title("Test set: predicted vs actual")
res = seg.actual - seg.pred
ax[1].hist(res, bins=70, color=PALETTE["teal"]); ax[1].axvline(0, color="k"); ax[1].set_xlabel("residual (actual - predicted, $)"); ax[1].set_title("Residuals are centred, with fat tails")
ax[2].scatter(seg.pred, res, s=4, alpha=.3, color=PALETTE["purple"]); ax[2].axhline(0, color="k"); ax[2].set_xlabel("predicted ($)"); ax[2].set_ylabel("residual ($)"); ax[2].set_title("Residuals vs prediction (heteroscedastic)")
savefig(fig, "fig33_test_diagnostics")

# %% premium_classifier
# Final screening classifier. Threshold chosen on OUT-OF-FOLD training predictions, then frozen.
thr_value = premium_threshold(y_train)
Xs = simple_features_pipeline(); Xtr_s = Xs.fit_transform(X_train); Xte_s = Xs.transform(X_test)
Xtr_s = Xtr_s.toarray() if hasattr(Xtr_s, "toarray") else Xtr_s; Xte_s = Xte_s.toarray() if hasattr(Xte_s, "toarray") else Xte_s
ytr_c, yte_c = make_premium(y_train, thr_value), make_premium(y_test, thr_value)
clf = HistGradientBoostingClassifier(learning_rate=0.08, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True, random_state=RANDOM_STATE)
oof = cross_val_predict(clf, Xtr_s, ytr_c, cv=5, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(ytr_c, oof)
f1s = 2 * prec * rec / (prec + rec + 1e-12); t_best = thr[np.argmax(f1s[:-1])]
t_prec = thr[np.argmax(prec[:-1] >= 0.90)]
print("Premium threshold = $%.0f" % thr_value)
print("Out-of-fold: best-F1 operating point at probability %.3f ; 90%%-precision point at %.3f" % (t_best, t_prec))
clf.fit(Xtr_s, ytr_c); proba = clf.predict_proba(Xte_s)[:, 1]
for name, t in [("max-F1 threshold", t_best), ("90%-precision threshold", t_prec), ("default 0.5", 0.5)]:
    pr = proba >= t
    print("TEST @ %-24s precision %.3f | recall %.3f | F1 %.3f" % (name, precision_score(yte_c, pr), recall_score(yte_c, pr), f1_score(yte_c, pr)))
print("TEST ROC-AUC = %.4f" % roc_auc_score(yte_c, proba))
print("Confusion matrix @ max-F1 (rows actual, cols predicted):\n", confusion_matrix(yte_c, proba >= t_best))
record("clf_test_auc", roc_auc_score(yte_c, proba)); pr = proba >= t_best
record("clf_test_precision", precision_score(yte_c, pr)); record("clf_test_recall", recall_score(yte_c, pr)); record("clf_test_f1", f1_score(yte_c, pr))
record("clf_threshold_prob", float(t_best)); record("premium_value_threshold", thr_value)

# %% ship
model.fit(X_train, y_train)    # (already fitted by search; refit is a no-op safeguard that keeps CV out of the artefact)
joblib.dump(model, ART_DIR / "homelens_regressor.joblib")
joblib.dump({"preprocessing": Xs, "classifier": clf, "threshold": float(t_best), "value_threshold": thr_value}, ART_DIR / "homelens_premium_classifier.joblib")
card = {
    "name": "HomeLens district valuation v1.0", "data": "California 1990 census block groups (20,640 rows)",
    "intended_use": "Rank and screen census districts by typical house value; NOT for pricing an individual home.",
    "regressor": "ColumnTransformer(+k-means geo similarity) -> HistGradientBoostingRegressor", "best_params": bp,
    "metrics_test": {"rmse": round(test_rmse), "mae": round(test_mae), "r2": round(test_r2, 3), "rmse_95ci": [round(float(c)) for c in ci]},
    "classifier_test": {"auc": round(float(roc_auc_score(yte_c, proba)), 4), "precision": round(float(precision_score(yte_c, pr)), 3), "recall": round(float(recall_score(yte_c, pr)), 3)},
    "known_limitations": ["Prices capped at $500,001 in the source data", "1990 data - the market has changed", "District-level medians, not individual homes",
                          "Weaker in coastal / <1H OCEAN areas", "Will not extrapolate outside the observed feature ranges"],
    "random_state": RANDOM_STATE}
(ART_DIR / "model_card.json").write_text(json.dumps(card, indent=2))
print(json.dumps(card, indent=2)[:1500])
