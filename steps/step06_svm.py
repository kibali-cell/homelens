"""STEP 6 - Chapter 5: Support Vector Machines (large-margin classification, kernels, SVR)."""
# %% setup
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, time
from sklearn.svm import LinearSVC, SVC, SVR, LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from homelens.tasks import load_task
from homelens.utils import set_style, savefig, record, rmse, PALETTE
from homelens.config import RANDOM_STATE
set_style()
T = load_task()
X, y = T.X, T.y_premium
rng = np.random.RandomState(RANDOM_STATE)
perm = rng.permutation(len(X))
hold = perm[:4000]                 # an internal hold-out taken from TRAINING data (test set still locked)
pool = perm[4000:]
print("pool:", len(pool), "| internal hold-out:", len(hold))

# %% linear_svm
t0 = time.time()
lin = LinearSVC(C=1.0, max_iter=20000, random_state=RANDOM_STATE).fit(X[pool], y[pool])
t_svm = time.time() - t0
pr = lin.predict(X[hold])
print("LinearSVC   : accuracy %.3f  F1 %.3f  (fit %.2fs)" % (accuracy_score(y[hold], pr), f1_score(y[hold], pr), t_svm))
lr = LogisticRegression(max_iter=3000).fit(X[pool], y[pool]); pl = lr.predict(X[hold])
print("LogisticReg.: accuracy %.3f  F1 %.3f   <- a linear SVM and logistic regression are close cousins" % (accuracy_score(y[hold], pl), f1_score(y[hold], pl)))
print("Support vectors are NOT reported for LinearSVC; margin width = 2/||w|| = %.3f" % (2 / np.linalg.norm(lin.coef_)))

# %% scaling_matters
# SVMs are distance/margin based: unscaled features with very different ranges wreck them.
raw_cols = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
Xraw = T.X_raw[raw_cols].fillna(T.X_raw[raw_cols].median()).to_numpy()
for name, model in [("RBF-SVC, raw features   ", SVC(kernel="rbf", C=1, gamma="scale")),
                    ("RBF-SVC, scaled features", make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma="scale")))]:
    t0 = time.time(); model.fit(Xraw[pool[:3000]], y[pool[:3000]]); p_ = model.predict(Xraw[hold])
    print(f"{name}: accuracy {accuracy_score(y[hold], p_):.3f}  F1 {f1_score(y[hold], p_):.3f}  ({time.time()-t0:.1f}s)")

# %% boundaries
# Visualise the effect of the two RBF hyper-parameters on a 2-D slice (income, latitude).
f2 = [T.feature_names.index("num__median_income"), T.feature_names.index("num__latitude")]
X2 = np.clip(X[:, f2], -3, 5); sub = pool[:1500]
gx, gy = np.meshgrid(np.linspace(-3, 5, 220), np.linspace(-3, 5, 220))
fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
configs = [(0.1, 1), (1, 1), (50, 1), (1, 0.01), (1, 100), (50, 1000)]
for ax, (gamma, C) in zip(axes.ravel(), configs):
    m = SVC(kernel="rbf", gamma=gamma, C=C).fit(X2[sub], y[sub])
    Z = m.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contourf(gx, gy, Z > 0, alpha=.15, colors=[PALETTE["grey"], PALETTE["orange"]], levels=[-.5, .5, 1.5])
    ax.contour(gx, gy, Z, levels=[0], colors=[PALETTE["blue"]], linewidths=2)
    ax.scatter(X2[sub][y[sub] == 0, 0], X2[sub][y[sub] == 0, 1], s=5, color=PALETTE["grey"])
    ax.scatter(X2[sub][y[sub] == 1, 0], X2[sub][y[sub] == 1, 1], s=5, color=PALETTE["orange"])
    ax.set_xlim(-3, 5); ax.set_ylim(-3, 5)
    acc = m.score(X2[hold], y[hold])
    ax.set_title(f"gamma={gamma}, C={C}  | n_SV={m.n_support_.sum()} | hold-out acc {acc:.3f}", fontsize=9)
    print(f"gamma={gamma:<5} C={C:<5}: support vectors = {m.n_support_.sum():4d} of {len(sub)} | train acc {m.score(X2[sub], y[sub]):.3f} | hold-out acc {acc:.3f}")
fig.suptitle("RBF kernel: gamma controls how local the boundary is, C controls how strictly errors are punished", fontweight="bold")
savefig(fig, "fig17_svm_boundaries")

# %% grid_search
sub = pool[:4000]
grid = GridSearchCV(SVC(kernel="rbf"),
                    {"C": [0.3, 1, 3, 10, 30], "gamma": [0.01, 0.03, 0.1, 0.3]}, cv=3, scoring="f1", n_jobs=2)
t0 = time.time(); grid.fit(X[sub], y[sub]); print("Grid search over 20 combos x 3 folds took %.0fs" % (time.time() - t0))
print("best params:", grid.best_params_, "| CV F1 = %.3f" % grid.best_score_)
best = grid.best_estimator_
pb = best.predict(X[hold])
print("Hold-out (4000 unseen training rows): accuracy %.3f | F1 %.3f" % (accuracy_score(y[hold], pb), f1_score(y[hold], pb)))
print("Support vectors: %d of %d (%.0f%%)" % (best.n_support_.sum(), len(sub), 100 * best.n_support_.sum() / len(sub)))
record("svc_best", {k: float(v) for k, v in grid.best_params_.items()}); record("svc_holdout_f1", f1_score(y[hold], pb))
table = pd.DataFrame(grid.cv_results_)[["param_C", "param_gamma", "mean_test_score"]].pivot(index="param_C", columns="param_gamma", values="mean_test_score").round(3)
print("\nCV F1 by (C rows x gamma cols):\n" + table.to_string())

# %% time_scaling
rows = []
for n in [1000, 2000, 4000, 8000, 12000]:
    idx = pool[:n]
    t0 = time.time(); SVC(kernel="rbf", C=3, gamma=0.1).fit(X[idx], y[idx]); ts = time.time() - t0
    t0 = time.time(); LinearSVC(max_iter=20000).fit(X[idx], y[idx]); tl = time.time() - t0
    rows.append((n, round(ts, 2), round(tl, 3)))
tt = pd.DataFrame(rows, columns=["n_samples", "kernel_SVC_sec", "LinearSVC_sec"])
print(tt.to_string(index=False))
print("Kernel SVC: doubling n multiplies time by ~%.1fx (between O(m^2) and O(m^3)); LinearSVC by ~%.1fx" %
      (tt.kernel_SVC_sec.iloc[3] / tt.kernel_SVC_sec.iloc[2], tt.LinearSVC_sec.iloc[3] / max(tt.LinearSVC_sec.iloc[2], 1e-6)))
record("svc_time_table", tt.to_dict("list"))

# %% svr
y100 = T.y_value.to_numpy() / 1e5
sub = pool[:4000]
res = []
t0 = time.time(); l = LinearSVR(epsilon=0.1, C=1, max_iter=50000, random_state=RANDOM_STATE).fit(X[pool], y100[pool])
res.append(("LinearSVR (epsilon=0.1), 12.5k rows", 1e5 * rmse(y100[hold], l.predict(X[hold])), time.time() - t0))
for C, eps in [(1, 0.1), (3, 0.1), (10, 0.1), (3, 0.5)]:
    t0 = time.time(); s = SVR(kernel="rbf", C=C, epsilon=eps, gamma="scale").fit(X[sub], y100[sub])
    res.append((f"SVR rbf C={C} eps={eps}, 4k rows", 1e5 * rmse(y100[hold], s.predict(X[hold])), time.time() - t0))
    if (C, eps) == (3, 0.1): n_sv = len(s.support_)
r = pd.DataFrame(res, columns=["model", "hold-out RMSE ($)", "fit sec"]).round(1)
print(r.to_string(index=False))
print("\nSupport vectors in SVR(C=3, eps=0.1): %d of %d (a wider epsilon tube -> fewer SVs)" % (n_sv, len(sub)))
record("svr_best_rmse", float(r["hold-out RMSE ($)"].min()))
