"""STEP 9 - Chapter 8: Dimensionality Reduction (curse of dimensionality, PCA, kernel PCA, t-SNE)."""
# %% setup
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, time
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from homelens.tasks import load_task
from homelens.utils import set_style, savefig, record, rmse, PALETTE
from homelens.config import RANDOM_STATE
set_style()
T = load_task()
X, y, yv = T.X, T.y_premium, T.y_value.to_numpy()
names = [n.split("__")[-1] for n in T.feature_names]
print("Working with", X.shape[1], "standardised features")

# %% curse
# In high dimensions, 'near' and 'far' stop meaning anything.
rng = np.random.RandomState(0)
rows = []
for d in [2, 10, 100, 1000, 10000]:
    P = rng.rand(500, d); q = rng.rand(1, d)
    dist = np.linalg.norm(P - q, axis=1)
    rows.append((d, dist.min(), dist.max(), dist.min() / dist.max()))
print(pd.DataFrame(rows, columns=["dimensions", "nearest", "farthest", "nearest/farthest"]).round(3).to_string(index=False))
print("-> as d grows the nearest neighbour is almost as far as the farthest one: distance-based methods degrade.")
# fraction of the unit hypercube within 0.001 of its surface
print("Share of a random point's volume 'near the border' (within 0.001) in d=1000: %.1f%%" % (100 * (1 - 0.998 ** 1000)))

# %% pca_by_hand
# PCA = eigen-decomposition of the covariance matrix = SVD of the centred data.
Xc = X - X.mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
var_ratio_manual = S ** 2 / np.sum(S ** 2)
pca = PCA().fit(X)
print("explained variance ratio (manual SVD):", var_ratio_manual[:5].round(4))
print("explained variance ratio (sklearn)   :", pca.explained_variance_ratio_[:5].round(4))
print("max |difference| = %.2e" % np.abs(var_ratio_manual - pca.explained_variance_ratio_).max())
cum = np.cumsum(pca.explained_variance_ratio_)
d95 = int(np.argmax(cum >= 0.95) + 1); d99 = int(np.argmax(cum >= 0.99) + 1)
print("Components needed for 95%% of the variance: %d | for 99%%: %d (of %d)" % (d95, d99, X.shape[1]))
record("pca_d95", d95); record("pca_d99", d99)
loadings = pd.DataFrame(pca.components_[:3].T, index=names, columns=["PC1", "PC2", "PC3"]).round(2)
print("\nWhat do the first components mean? (loadings)")
print(loadings.to_string())
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(range(1, len(cum) + 1), pca.explained_variance_ratio_, color=PALETTE["blue"], label="per component")
ax[0].plot(range(1, len(cum) + 1), cum, "o-", color=PALETTE["orange"], label="cumulative"); ax[0].axhline(.95, ls="--", color=PALETTE["grey"])
ax[0].set_xlabel("component"); ax[0].set_ylabel("explained variance ratio"); ax[0].legend(); ax[0].set_title("Scree plot")
Z = pca.transform(X)[:, :2]
sc = ax[1].scatter(Z[:, 0], Z[:, 1], c=yv, s=3, alpha=.4, cmap="viridis"); plt.colorbar(sc, ax=ax[1], label="price ($)")
ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2"); ax[1].set_title("Districts projected to 2-D")
savefig(fig, "fig26_pca")

# %% reconstruction
for k in [2, 4, 8, 11, 16]:
    p_ = PCA(n_components=k).fit(X); rec = p_.inverse_transform(p_.transform(X))
    print("k=%2d components: reconstruction RMSE (in std-dev units) = %.3f | variance kept %.1f%%" %
          (k, np.sqrt(np.mean((X - rec) ** 2)), 100 * p_.explained_variance_ratio_.sum()))
# incremental and randomised PCA give the same answer, but scale to huge data
ipca = IncrementalPCA(n_components=8, batch_size=2000).fit(X)
rpca = PCA(n_components=8, svd_solver="randomized", random_state=0).fit(X)
full8 = PCA(n_components=8).fit(X)
print("\nIncrementalPCA vs PCA: max |diff| in explained variance = %.2e" % np.abs(ipca.explained_variance_ratio_ - full8.explained_variance_ratio_).max())
print("Randomised PCA  vs PCA: max |diff| in explained variance = %.2e" % np.abs(rpca.explained_variance_ratio_ - full8.explained_variance_ratio_).max())

# %% pca_before_model
# Does reducing dimensions before modelling help? Evaluate with proper cross-validation (PCA inside the pipeline).
rows = []
for k in [3, 5, 8, 11, 16]:
    for name, est in [("Linear", LinearRegression()), ("Random forest", RandomForestRegressor(60, min_samples_leaf=2, max_features=0.6, n_jobs=2, random_state=RANDOM_STATE))]:
        pipe = make_pipeline(PCA(k, random_state=0), est)
        s = -cross_val_score(pipe, X, yv, cv=4, scoring="neg_root_mean_squared_error", n_jobs=1).mean()
        rows.append((k, name, s))
pc = pd.DataFrame(rows, columns=["n_components", "model", "cv_rmse"]).pivot(index="n_components", columns="model", values="cv_rmse").round(0)
raw_rf = -cross_val_score(RandomForestRegressor(60, min_samples_leaf=2, max_features=0.6, n_jobs=2, random_state=RANDOM_STATE), X, yv, cv=4, scoring="neg_root_mean_squared_error").mean()
print(pc.to_string())
print("\nRandom forest on the ORIGINAL 16 features: CV RMSE = %.0f" % raw_rf)
print("=> At k=16 PCA is only a rotation, yet the forest is already worse: axis-aligned splits lose the meaningful axes. Dropping components then removes signal too.")
record("pca_rf_raw", raw_rf); record("pca_rf_k16", float(pc.loc[16, "Random forest"])); record("pca_rf_k8", float(pc.loc[8, "Random forest"])); record("pca_lin_k16", float(pc.loc[16, "Linear"]))
fig, ax = plt.subplots(figsize=(6.5, 4)); pc.plot(marker="o", ax=ax, color=[PALETTE["orange"], PALETTE["blue"]]); ax.axhline(raw_rf, color=PALETTE["red"], ls="--", label="RF on raw features")
ax.set_ylabel("CV RMSE ($)"); ax.set_title("Accuracy after PCA with k components"); ax.legend()
savefig(fig, "fig27_pca_models")

# %% kernel_pca
# Supervised use: choose the kernel and its hyper-parameters with a grid search on the final classification score.
sub = np.random.RandomState(1).permutation(len(X))[:3000]
pipe = make_pipeline(KernelPCA(n_components=6, kernel="rbf"), LogisticRegression(max_iter=3000))
grid = GridSearchCV(pipe, {"kernelpca__gamma": [0.003, 0.01, 0.03, 0.1], "kernelpca__kernel": ["rbf", "sigmoid"]}, cv=3, n_jobs=2)
t0 = time.time(); grid.fit(X[sub], y[sub])
print("kernel-PCA + logistic regression grid (3000 rows) took %.0fs" % (time.time() - t0))
print("best:", grid.best_params_, "| CV accuracy %.3f" % grid.best_score_)
lin = cross_val_score(make_pipeline(PCA(6), LogisticRegression(max_iter=3000)), X[sub], y[sub], cv=3).mean()
print("Plain PCA(6) + logistic regression CV accuracy: %.3f" % lin)
record("kpca_acc", grid.best_score_); record("pca6_acc", lin)

# %% tsne
# t-SNE is for VISUALISATION only: it preserves neighbourhoods, not distances, and cannot transform new data.
sub = np.random.RandomState(2).permutation(len(X))[:3000]
t0 = time.time()
E = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", random_state=0).fit_transform(X[sub])
print("t-SNE on 3000 districts took %.0fs" % (time.time() - t0))
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
Zp = PCA(2).fit_transform(X[sub])
for a, Zs, ttl in [(ax[0], Zp, "PCA (linear)"), (ax[1], E, "t-SNE (non-linear)")]:
    sc = a.scatter(Zs[:, 0], Zs[:, 1], c=T.y_tier[sub], s=5, cmap="viridis"); a.set_title(ttl); a.set_xticks([]); a.set_yticks([])
cb = plt.colorbar(sc, ax=ax, ticks=[0, 1, 2, 3], shrink=.8); cb.ax.set_yticklabels(["budget", "lower-mid", "upper-mid", "premium"])
savefig(fig, "fig28_tsne")
from sklearn.neighbors import KNeighborsClassifier
for nm, Zs in [("PCA-2D", Zp), ("t-SNE-2D", E)]:
    print("%-9s kNN(15) price-tier accuracy in 2-D: %.3f" % (nm, cross_val_score(KNeighborsClassifier(15), Zs, T.y_tier[sub], cv=5).mean()))
