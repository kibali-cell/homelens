"""STEP 7 - Chapter 6: Decision Trees (CART, Gini, regularisation, instability, regression trees)."""
# %% setup
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split, validation_curve
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from homelens.tasks import load_task
from homelens.utils import set_style, savefig, record, rmse, PALETTE
from homelens.config import RANDOM_STATE
set_style()
T = load_task()
X, y, names = T.X, T.y_premium, T.feature_names
short = [n.split("__")[-1] for n in names]
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

# %% shallow_tree
tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE).fit(Xtr, ytr)
print(export_text(tree, feature_names=short, decimals=2))
print("train acc %.3f | validation acc %.3f" % (tree.score(Xtr, ytr), tree.score(Xva, yva)))
fig, ax = plt.subplots(figsize=(15, 6.5))
plot_tree(tree, feature_names=short, class_names=["not premium", "premium"], filled=True, rounded=True, impurity=True, fontsize=8, ax=ax, proportion=False)
savefig(fig, "fig18_tree_depth3")

# %% gini_by_hand
# Verify the root node's Gini impurity and the CART split cost by hand.
t = tree.tree_
def node_stats(i):
    n = int(t.n_node_samples[i]); frac = t.value[i][0]; frac = frac / frac.sum()   # class proportions
    return n, frac, 1 - np.sum(frac ** 2)                                          # size, proportions, Gini
n_root, p, G_root = node_stats(0)
print("Root node: %d samples, class proportions %s" % (n_root, p.round(3)))
print("Gini(root) = 1 - (p0^2 + p1^2) = 1 - (%.3f^2 + %.3f^2) = %.4f" % (p[0], p[1], G_root))
l, r = t.children_left[0], t.children_right[0]
(nl, _, gl), (nr, _, gr) = node_stats(l), node_stats(r)
J = nl / n_root * gl + nr / n_root * gr
raw_inc = T.X_raw["median_income"]
thr_raw = raw_inc.mean() + t.threshold[0] * raw_inc.std(ddof=0)      # undo the StandardScaler
print("Split: %s <= %.3f (standardised)  =  income <= %.2f in the original units (x $10,000)" % (short[t.feature[0]], t.threshold[0], thr_raw))
print("  left : n=%5d  Gini=%.4f" % (nl, gl))
print("  right: n=%5d  Gini=%.4f" % (nr, gr))
print("CART cost J = (%d/%d)*%.4f + (%d/%d)*%.4f = %.4f  -> impurity drops from %.4f to %.4f" % (nl, n_root, gl, nr, n_root, gr, J, G_root, J))
record("gini_root", G_root); record("gini_after_split", J); record("root_feature", short[t.feature[0]]); record("root_thr_income", thr_raw)
print("\npredict_proba for one district:", tree.predict_proba(Xva[:1]).round(3), "(= class fractions in its leaf)")

# %% overfit_tree
full = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(Xtr, ytr)
print("Unconstrained tree: depth=%d, leaves=%d, train acc=%.3f, validation acc=%.3f" %
      (full.get_depth(), full.get_n_leaves(), full.score(Xtr, ytr), full.score(Xva, yva)))
record("full_tree_depth", full.get_depth()); record("full_tree_leaves", full.get_n_leaves())
record("full_tree_val", full.score(Xva, yva))

# %% regularise
depths = np.arange(1, 26)
tr_s, va_s = validation_curve(DecisionTreeClassifier(random_state=RANDOM_STATE), X, y, param_name="max_depth", param_range=depths, cv=5)
leaf_range = [1, 2, 5, 10, 20, 50, 100, 200, 400]
tr_l, va_l = validation_curve(DecisionTreeClassifier(random_state=RANDOM_STATE), X, y, param_name="min_samples_leaf", param_range=leaf_range, cv=5)
bd = depths[va_s.mean(1).argmax()]
print("Best max_depth = %d (CV accuracy %.3f)" % (bd, va_s.mean(1).max()))
bl = leaf_range[va_l.mean(1).argmax()]
print("Best min_samples_leaf = %d (CV accuracy %.3f)" % (bl, va_l.mean(1).max()))
print(pd.DataFrame({"min_samples_leaf": leaf_range, "train_acc": tr_l.mean(1).round(3), "cv_acc": va_l.mean(1).round(3)}).to_string(index=False))
record("tree_best_depth", int(bd)); record("tree_best_leaf", int(bl)); record("tree_best_acc", float(max(va_s.mean(1).max(), va_l.mean(1).max())))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(depths, tr_s.mean(1), "o-", label="train", color=PALETTE["blue"], ms=3); ax[0].plot(depths, va_s.mean(1), "o-", label="CV", color=PALETTE["orange"], ms=3)
ax[0].set_xlabel("max_depth"); ax[0].set_ylabel("accuracy"); ax[0].legend(); ax[0].set_title("Depth: overfitting starts after the peak")
ax[1].plot(leaf_range, tr_l.mean(1), "o-", label="train", color=PALETTE["blue"]); ax[1].plot(leaf_range, va_l.mean(1), "o-", label="CV", color=PALETTE["orange"])
ax[1].set_xscale("log"); ax[1].set_xlabel("min_samples_leaf"); ax[1].legend(); ax[1].set_title("Leaf size: bigger leaves = more regularisation")
savefig(fig, "fig19_tree_regularise")

# %% instability
# Trees are sensitive: (a) rotate the data, (b) remove ONE training example.
f2 = [names.index("num__median_income"), names.index("num__latitude")]
X2 = np.clip(X[:, f2], -3, 5)
gx, gy = np.meshgrid(np.linspace(-3, 5, 250), np.linspace(-3, 5, 250))
fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
def draw(a, model, data, title, grid):
    Z = model.predict(grid).reshape(gx.shape)
    a.contourf(gx, gy, Z, alpha=.2, colors=[PALETTE["grey"], PALETTE["orange"]], levels=[-.5, .5, 1.5])
    a.scatter(data[:, 0][::6], data[:, 1][::6], c=np.where(y[::6] == 1, PALETTE["orange"], PALETTE["grey"]), s=3)
    a.set_title(title, fontsize=10); a.set_xlim(-3, 5); a.set_ylim(-3, 5)
base = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE).fit(X2, y)
draw(ax[0], base, X2, "Original axes", np.c_[gx.ravel(), gy.ravel()])
ang = np.deg2rad(35); R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
rot = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE).fit(X2 @ R, y)
draw(ax[1], rot, X2, "Data rotated 35 deg -> different boundary", np.c_[gx.ravel(), gy.ravel()] @ R)
keep = np.ones(len(X2), bool); keep[np.argmax(X2[:, 0])] = False
one_out = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE).fit(X2[keep], y[keep])
draw(ax[2], one_out, X2, "One training row removed", np.c_[gx.ravel(), gy.ravel()])
savefig(fig, "fig20_tree_instability")
acc_o, acc_r = base.score(X2, y), rot.score(X2 @ R, y)
print("Accuracy original axes: %.3f | rotated axes: %.3f" % (acc_o, acc_r))
# PCA rotates the data so that axes line up with directions of variance - often stabilises trees
pca_X = PCA().fit_transform(X)
a1 = cross_val_score(DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE), X, y, cv=5).mean()
a2 = cross_val_score(DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE), pca_X, y, cv=5).mean()
print("Full 16-feature tree CV acc: raw axes %.3f vs PCA-rotated axes %.3f" % (a1, a2))

# %% regression_tree
inc = T.X_raw[["median_income"]].to_numpy(); yv = T.y_value.to_numpy()
g = np.linspace(inc.min(), 12, 500).reshape(-1, 1)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
for a, kw, ttl in zip(ax, [dict(max_depth=2), dict(max_depth=4), dict()], ["max_depth=2 (4 leaves)", "max_depth=4 (16 leaves)", "no limit (overfit)"]):
    m = DecisionTreeRegressor(random_state=RANDOM_STATE, **kw).fit(inc[:3000], yv[:3000])
    a.scatter(inc[:3000], yv[:3000], s=3, alpha=.25, color=PALETTE["grey"]); a.plot(g, m.predict(g), color=PALETTE["red"], lw=1.8)
    a.set_title(ttl + " | leaves=%d" % m.get_n_leaves(), fontsize=10); a.set_xlabel("median income")
ax[0].set_ylabel("predicted value ($)")
savefig(fig, "fig21_regression_trees")
m2 = DecisionTreeRegressor(max_depth=2, random_state=RANDOM_STATE).fit(inc, yv)
print(export_text(m2, feature_names=["median_income"], decimals=2))
cvr = -cross_val_score(DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE), X, yv, cv=5, scoring="neg_root_mean_squared_error")
print("Regression tree on all 16 features (depth 8): CV RMSE = %.0f" % cvr.mean())
record("tree_reg_rmse", cvr.mean())
