"""STEP 10 - Chapter 9: Unsupervised Learning (k-means, DBSCAN, Gaussian mixtures, semi-supervised, anomalies)."""
# %% setup
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, time
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from homelens.tasks import load_task
from homelens.features import build_preprocessing
from homelens.utils import set_style, savefig, record, rmse, PALETTE
from homelens.config import RANDOM_STATE
set_style()
T = load_task()
geo = T.X_raw[["longitude", "latitude"]].to_numpy()
X, y, yv = T.X, T.y_premium, T.y_value.to_numpy()

# %% choose_k
ks = range(2, 16)
inertia, sil = [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(geo)
    inertia.append(km.inertia_); sil.append(silhouette_score(geo, km.labels_, sample_size=4000, random_state=0))
print(pd.DataFrame({"k": list(ks), "inertia": np.round(inertia, 0), "silhouette": np.round(sil, 3)}).to_string(index=False))
bestk = list(ks)[int(np.argmax(sil))]
print("Highest silhouette at k =", bestk, "(there is no single 'right' k - it is a modelling choice)")
record("kmeans_best_k_sil", bestk)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(list(ks), inertia, "o-", color=PALETTE["blue"]); ax[0].set_xlabel("k"); ax[0].set_ylabel("inertia (within-cluster SSE)"); ax[0].set_title("Elbow method")
ax[1].plot(list(ks), sil, "o-", color=PALETTE["orange"]); ax[1].set_xlabel("k"); ax[1].set_ylabel("silhouette score"); ax[1].set_title("Silhouette method")
savefig(fig, "fig29_choose_k")

# %% kmeans_details
for init, n_init in [("random", 1), ("k-means++", 1), ("k-means++", 10)]:
    sc = [KMeans(8, init=init, n_init=n_init, random_state=s).fit(geo).inertia_ for s in range(8)]
    print("init=%-10s n_init=%-2d -> inertia over 8 seeds: best %.0f | worst %.0f | spread %.1f%%" %
          (init, n_init, min(sc), max(sc), 100 * (max(sc) / min(sc) - 1)))
t0 = time.time(); KMeans(8, n_init=3, random_state=0).fit(X); t_full = time.time() - t0
t0 = time.time(); mb = MiniBatchKMeans(8, n_init=3, batch_size=1024, random_state=0).fit(X); t_mb = time.time() - t0
print("KMeans %.2fs vs MiniBatchKMeans %.2fs on the 16-feature data" % (t_full, t_mb))
km8 = KMeans(8, n_init=10, random_state=RANDOM_STATE).fit(geo)
fig, ax = plt.subplots(figsize=(6, 5.2))
ax.scatter(geo[:, 0], geo[:, 1], c=km8.labels_, s=3, cmap="tab10"); ax.scatter(*km8.cluster_centers_.T, marker="X", s=140, c="k", edgecolor="w")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("k-means (k=8) market regions")
savefig(fig, "fig30_kmeans_map")
print("\nAverage price per k-means region:")
reg = pd.DataFrame({"region": km8.labels_, "price": yv}).groupby("region").price.agg(["count", "mean"]).round(0)
print(reg.to_string())
print("-> the clusters are geographic markets; pricing differs strongly between them.")

# %% clusters_as_features
# Use clustering as a feature-engineering step (Chapter 2's ClusterSimilarity) - does it help?
rf = RandomForestRegressor(80, min_samples_leaf=2, max_features=0.5, n_jobs=2, random_state=RANDOM_STATE)
res = {}
for lab, uc in [("without geo-cluster features", False), ("with 25 geo-cluster similarity features", True)]:
    pipe = make_pipeline(build_preprocessing(n_clusters=25, use_clusters=uc), rf)
    res[lab] = -cross_val_score(pipe, T.X_raw, yv, cv=4, scoring="neg_root_mean_squared_error", n_jobs=1).mean()
    print("%-42s CV RMSE = %.0f" % (lab, res[lab]))
imp = 100 * (1 - res["with 25 geo-cluster similarity features"] / res["without geo-cluster features"])
print("Improvement: %.1f%%" % imp); record("cluster_feat_gain_pct", imp)
record("rmse_no_geo", res["without geo-cluster features"]); record("rmse_geo", res["with 25 geo-cluster similarity features"])

# %% semi_supervised
# Imagine labelling districts is EXPENSIVE (an expert valuation). We can only afford 20 labels.
rng = np.random.RandomState(0)
n_lab = 20
def draw():                                             # random labelled set (re-draw if only one class shows up)
    while True:
        i = rng.choice(len(X), n_lab, replace=False)
        if 0 < y[i].sum() < n_lab: return i
rand_idx = draw()
km = KMeans(n_lab, n_init=5, random_state=0).fit(X)
dist = km.transform(X)
rep_idx = np.argmin(dist, axis=0)                      # the most representative district of each cluster
holdout = np.setdiff1d(np.arange(len(X)), np.r_[rand_idx, rep_idx])
def evaluate(idx, extra=None):
    m = LogisticRegression(max_iter=3000).fit(X[idx], y[idx]); return m.score(X[holdout], y[holdout])
a_rand = np.mean([LogisticRegression(max_iter=3000).fit(X[i], y[i]).score(X[holdout], y[holdout]) for i in [draw() for _ in range(20)]])
a_rep = evaluate(rep_idx)
# propagate each representative's label to the whole cluster, train on everything
prop = y[rep_idx][km.labels_]
a_prop = LogisticRegression(max_iter=3000).fit(X, prop).score(X[holdout], y[holdout])
# propagate only to the 20% of points closest to their centroid (cleaner)
close = dist[np.arange(len(X)), km.labels_] <= np.percentile(dist[np.arange(len(X)), km.labels_], 20)
a_prop20 = LogisticRegression(max_iter=3000).fit(X[close], prop[close]).score(X[holdout], y[holdout])
full = LogisticRegression(max_iter=3000).fit(X[holdout[:8000]], y[holdout[:8000]]).score(X[holdout[8000:]], y[holdout[8000:]])
print("Accuracy with only %d labels:" % n_lab)
print("  random 20 labelled districts (avg of 20 draws) : %.3f" % a_rand)
print("  20 cluster representatives                     : %.3f" % a_rep)
print("  label propagation to the whole cluster         : %.3f" % a_prop)
print("  propagation to the 20%% closest-to-centroid     : %.3f" % a_prop20)
print("  (reference) all labels, 8000 rows              : %.3f" % full)
purity = sum(max(y[km.labels_ == c].mean(), 1 - y[km.labels_ == c].mean()) * (km.labels_ == c).sum() for c in range(n_lab)) / len(y)
print("\nWhy the gain is small: cluster purity (share of districts matching their cluster's majority label) = %.3f vs %.3f for always-guess-majority." % (purity, 1 - y.mean()))
print("Cluster-then-label works when clusters align with classes (e.g. handwritten digits). Here price classes overlap heavily in feature space.")
record("semi_purity", purity)
record("semi", {"random": a_rand, "rep": a_rep, "prop": a_prop, "prop20": a_prop20, "full": full})

# %% dbscan
# Density-based: finds dense urban areas of ANY shape, labels sparse districts as noise (-1).
db = DBSCAN(eps=0.12, min_samples=40).fit(geo)
lab = db.labels_
n_cl = len(set(lab)) - (1 if -1 in lab else 0)
print("DBSCAN found %d dense regions; %.1f%% of districts labelled noise (-1)" % (n_cl, 100 * np.mean(lab == -1)))
sizes = pd.Series(lab[lab >= 0]).value_counts().head(5)
print("Largest regions (label: size):", {int(k): int(v) for k, v in sizes.items()})
fig, ax = plt.subplots(figsize=(6, 5.2))
ax.scatter(geo[lab == -1, 0], geo[lab == -1, 1], s=3, color=PALETTE["grey"], alpha=.5, label="noise")
ax.scatter(geo[lab >= 0, 0], geo[lab >= 0, 1], c=lab[lab >= 0], s=3, cmap="tab20"); ax.legend(markerscale=4)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("DBSCAN (eps=0.12, min_samples=40)")
savefig(fig, "fig31_dbscan")
ag = AgglomerativeClustering(n_clusters=8, linkage="ward").fit(geo[:4000])
print("Hierarchical (Ward) vs k-means agreement on 4000 points: adjusted Rand = %.3f" % adjusted_rand_score(ag.labels_, km8.labels_[:4000]))

# %% gmm_anomaly
# Gaussian mixture models: soft clustering + a density estimate we can use for anomaly detection.
Z = StandardScaler().fit_transform(np.c_[np.log(T.X_raw["median_income"]), np.log(T.X_raw["households"]), np.log(T.X_raw["population"] / T.X_raw["households"]),
                                         T.X_raw["housing_median_age"], np.log(yv)])
rows = []
for k in range(1, 9):
    g = GaussianMixture(k, n_init=3, random_state=0).fit(Z); rows.append((k, g.bic(Z), g.aic(Z)))
bic = pd.DataFrame(rows, columns=["k", "BIC", "AIC"]).round(0); print(bic.to_string(index=False))
kb = int(bic.loc[bic.BIC.idxmin(), "k"]); print("Lowest BIC at k =", kb)
gm = GaussianMixture(kb, n_init=5, random_state=0).fit(Z)
dens = gm.score_samples(Z)
thr = np.percentile(dens, 2)
anom = dens < thr
print("Flagged the 2%% lowest-density districts as anomalies: %d" % anom.sum())
show = T.X_raw.assign(price=yv, density=dens)[anom].sort_values("density").head(6)[["median_income", "households", "population", "housing_median_age", "price", "ocean_proximity"]]
print(show.round(1).to_string())
record("gmm_k", kb); record("n_anomalies", int(anom.sum()))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].plot(bic.k, bic.BIC, "o-", label="BIC", color=PALETTE["blue"]); ax[0].plot(bic.k, bic.AIC, "s-", label="AIC", color=PALETTE["orange"]); ax[0].legend()
ax[0].set_xlabel("number of components"); ax[0].set_ylabel("information criterion (lower = better)"); ax[0].set_title("Choosing the number of Gaussians")
ax[1].scatter(geo[~anom, 0], geo[~anom, 1], s=2, color=PALETTE["grey"], alpha=.35); ax[1].scatter(geo[anom, 0], geo[anom, 1], s=14, color=PALETTE["red"], label="anomalous districts")
ax[1].set_xlabel("longitude"); ax[1].set_ylabel("latitude"); ax[1].legend(); ax[1].set_title("Density-based anomaly detection (2% lowest)")
savefig(fig, "fig32_gmm")
