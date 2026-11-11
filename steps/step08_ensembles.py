"""STEP 8 - Chapter 7: Ensemble Learning and Random Forests (voting, bagging, forests, boosting, stacking)."""
# %% setup
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, time
from scipy.stats import binom
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
                              StackingRegressor, VotingClassifier)
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score
from homelens.tasks import load_task
from homelens.utils import set_style, savefig, record, rmse, PALETTE
from homelens.config import RANDOM_STATE
set_style()
T = load_task()
X, y, yv = T.X, T.y_premium, T.y_value.to_numpy()
names = [n.split("__")[-1] for n in T.feature_names]
Xtr, Xva, ytr, yva, vtr, vva = train_test_split(X, y, yv, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

# %% law_of_large_numbers
# Why ensembles work: many weak, INDEPENDENT voters beat one. A coin that is right 51% of the time...
fig, ax = plt.subplots(figsize=(6.5, 3.8))
n = np.arange(1, 5001, 20)
for p, c in [(0.51, PALETTE["orange"]), (0.55, PALETTE["blue"]), (0.65, PALETTE["green"])]:
    ax.plot(n, 1 - binom.cdf(n // 2, n, p), label=f"each voter {int(p*100)}% accurate", color=c, lw=2)
ax.set_xlabel("number of independent voters"); ax.set_ylabel("P(majority is right)"); ax.legend(); ax.set_title("Wisdom of the crowd (needs independence!)")
savefig(fig, "fig22_wisdom_crowd")
for p in [0.51, 0.55, 0.65]:
    print("voters %d%% accurate: majority of 1001 is right with P = %.3f ; majority of 5001: %.3f" %
          (p * 100, 1 - binom.cdf(500, 1001, p), 1 - binom.cdf(2500, 5001, p)))

# %% voting
clfs = [("lr", LogisticRegression(max_iter=3000)), ("rf", RandomForestClassifier(100, random_state=RANDOM_STATE, n_jobs=2)),
        ("knn", KNeighborsClassifier(15))]
for n_, c in clfs:
    c.fit(Xtr, ytr); print("%-5s validation accuracy %.4f | F1 %.4f" % (n_, c.score(Xva, yva), f1_score(yva, c.predict(Xva))))
for vt in ["hard", "soft"]:
    v = VotingClassifier(clfs, voting=vt, n_jobs=1).fit(Xtr, ytr)
    print("VOTING (%s): validation accuracy %.4f | F1 %.4f" % (vt, v.score(Xva, yva), f1_score(yva, v.predict(Xva))))
    record(f"voting_{vt}_acc", v.score(Xva, yva))

# %% bagging_oob
bag = BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=5), n_estimators=200, max_samples=0.7, bootstrap=True,
                        oob_score=True, n_jobs=2, random_state=RANDOM_STATE).fit(Xtr, ytr)
single = DecisionTreeClassifier(min_samples_leaf=5, random_state=RANDOM_STATE).fit(Xtr, ytr)
print("Single tree  validation accuracy: %.4f" % single.score(Xva, yva))
print("Bagging (200 trees) OOB estimate: %.4f  | actual validation accuracy: %.4f" % (bag.oob_score_, bag.score(Xva, yva)))
print("=> the out-of-bag score predicts validation accuracy WITHOUT needing a validation set")
print("A bootstrap sample leaves out ~ %.1f%% of rows (theory: 1/e = 36.8%%)" % (100 * np.mean([len(set(range(10000)) - set(np.random.RandomState(i).randint(0, 10000, 10000))) / 10000 for i in range(5)])))
record("bag_oob", bag.oob_score_); record("bag_val", bag.score(Xva, yva)); record("single_tree_val", single.score(Xva, yva))

# %% random_forest
rf = RandomForestClassifier(300, min_samples_leaf=3, oob_score=True, n_jobs=2, random_state=RANDOM_STATE).fit(Xtr, ytr)
print("Random forest: OOB %.4f | validation accuracy %.4f | F1 %.4f" % (rf.oob_score_, rf.score(Xva, yva), f1_score(yva, rf.predict(Xva))))
imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
print(imp.head(8).round(3).to_string())
record("rf_clf_val", rf.score(Xva, yva))
fig, ax = plt.subplots(figsize=(6.8, 4.4)); imp.head(10)[::-1].plot.barh(ax=ax, color=PALETTE["blue"]); ax.set_title("Random-forest feature importance (classification)")
savefig(fig, "fig23_rf_importance")
# number of trees
accs = []
for k in [1, 3, 10, 30, 100, 300]:
    accs.append((k, RandomForestClassifier(k, min_samples_leaf=3, n_jobs=2, random_state=RANDOM_STATE).fit(Xtr, ytr).score(Xva, yva)))
print("Trees -> validation accuracy:", " | ".join(f"{k}:{a:.3f}" for k, a in accs))

# %% boosting
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=300, learning_rate=0.5, random_state=RANDOM_STATE).fit(Xtr, ytr)
print("AdaBoost (300 stumps depth 2): validation accuracy %.4f | F1 %.4f" % (ada.score(Xva, yva), f1_score(yva, ada.predict(Xva))))
# Gradient boosting regression with early stopping, compared across learning rates
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for lr_, c in [(0.02, PALETTE["orange"]), (0.1, PALETTE["blue"]), (0.5, PALETTE["red"])]:
    gb = GradientBoostingRegressor(n_estimators=400, learning_rate=lr_, max_depth=3, subsample=0.8, random_state=RANDOM_STATE).fit(Xtr, vtr)
    errs = [rmse(vva, p) for p in gb.staged_predict(Xva)]
    ax[0].plot(errs, label=f"learning_rate={lr_}", color=c, lw=2)
    print("learning_rate=%.2f: best validation RMSE %.0f at %d trees; at 400 trees: %.0f" % (lr_, min(errs), int(np.argmin(errs)) + 1, errs[-1]))
ax[0].set_ylim(40000, 90000); ax[0].set_xlabel("number of trees"); ax[0].set_ylabel("validation RMSE ($)"); ax[0].legend(); ax[0].set_title("Gradient boosting: shrinkage vs. number of trees")
# residual-fitting idea in 3 lines
r1 = vtr - np.full_like(vtr, vtr.mean(), dtype=float)
t1 = DecisionTreeRegressor(max_depth=3).fit(Xtr, r1); pred1 = vtr.mean() + t1.predict(Xtr)
t2 = DecisionTreeRegressor(max_depth=3).fit(Xtr, vtr - pred1); pred2 = pred1 + t2.predict(Xtr)
print("\nBy hand: train RMSE  mean-only %.0f -> +tree1 %.0f -> +tree2 %.0f  (each tree fits the previous residuals)" %
      (rmse(vtr, np.full_like(vtr, vtr.mean(), dtype=float)), rmse(vtr, pred1), rmse(vtr, pred2)))
gb_es = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.1, max_depth=3, subsample=0.8, n_iter_no_change=15,
                                  validation_fraction=0.15, random_state=RANDOM_STATE).fit(Xtr, vtr)
print("Early stopping picked n_estimators = %d (out of a 2000 budget)" % gb_es.n_estimators_)
record("gb_early_trees", gb_es.n_estimators_)
ax[1].bar(["mean", "+tree 1", "+tree 2"], [rmse(vtr, np.full_like(vtr, vtr.mean(), dtype=float)), rmse(vtr, pred1), rmse(vtr, pred2)], color=PALETTE["teal"])
ax[1].set_ylabel("train RMSE ($)"); ax[1].set_title("Boosting by hand: fit the residuals")
savefig(fig, "fig24_boosting")

# %% model_bakeoff
# 5-fold CV RMSE for every regression model on the SAME features. This is the model-selection table.
def cv(model, name, Xm=X, folds=5):
    t0 = time.time()
    s = -cross_val_score(model, Xm, yv, cv=folds, scoring="neg_root_mean_squared_error", n_jobs=2)
    return (name, s.mean(), s.std(), time.time() - t0)
rows = [
    cv(LinearRegression(), "Linear regression"),
    cv(Ridge(alpha=10), "Ridge"),
    cv(DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=RANDOM_STATE), "Decision tree (tuned)"),
    cv(RandomForestRegressor(150, min_samples_leaf=2, max_features=0.5, n_jobs=1, random_state=RANDOM_STATE), "Random forest"),
    cv(ExtraTreesRegressor(150, min_samples_leaf=2, max_features=0.7, n_jobs=1, random_state=RANDOM_STATE), "Extra-trees"),
    cv(GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=4, subsample=0.8, random_state=RANDOM_STATE), "Gradient boosting"),
    cv(HistGradientBoostingRegressor(max_iter=500, learning_rate=0.08, max_leaf_nodes=31, early_stopping=True, random_state=RANDOM_STATE), "Hist. gradient boosting"),
]
stack = StackingRegressor(
    [("rf", RandomForestRegressor(100, min_samples_leaf=2, max_features=0.5, n_jobs=1, random_state=RANDOM_STATE)),
     ("hgb", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=RANDOM_STATE)),
     ("et", ExtraTreesRegressor(100, min_samples_leaf=2, max_features=0.7, n_jobs=1, random_state=RANDOM_STATE))],
    final_estimator=Ridge(alpha=1.0), cv=4, n_jobs=2)
rows.append(cv(stack, "Stacking (RF+HGB+ET -> Ridge)", folds=3))
tab = pd.DataFrame(rows, columns=["model", "cv_rmse", "cv_std", "secs"]).round(0)
print(tab.to_string(index=False))
record("bakeoff", tab.to_dict("list"))
fig, ax = plt.subplots(figsize=(8, 4.2)); order = tab.sort_values("cv_rmse", ascending=False)
ax.barh(order.model, order.cv_rmse, xerr=order.cv_std, color=PALETTE["blue"], alpha=.9)
for i, v in enumerate(order.cv_rmse): ax.text(v + 800, i, f"${v:,.0f}", va="center", fontsize=8)
ax.set_xlabel("cross-validated RMSE ($, lower is better)"); ax.set_title("Model bake-off on the same 16 features"); ax.set_xlim(0, 95000)
savefig(fig, "fig25_bakeoff")
