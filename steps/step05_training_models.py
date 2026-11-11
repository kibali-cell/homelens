"""STEP 5 - Chapter 4: Training Models. Open the black box: normal equation, gradient descent,
polynomial regression, learning curves, regularisation, early stopping, logistic & softmax regression."""
# %% setup
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, time
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge, SGDRegressor)
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import log_loss
from homelens.tasks import load_task
from homelens.utils import set_style, savefig, record, rmse, PALETTE
from homelens.config import RANDOM_STATE
set_style()
T = load_task()
cols = ["num__median_income", "num__housing_median_age", "num__bedrooms_ratio", "num__latitude"]
idx = [T.feature_names.index(c) for c in cols]
Xn = np.clip(T.X[:, idx], -4, 4)    # 4 standardised features, outliers clipped at +/-4 std (keeps SGD stable)
y100 = T.y_value.to_numpy() / 1e5   # price in units of $100,000 (keeps gradient numbers friendly)
Xb = np.c_[np.ones(len(Xn)), Xn]    # add the bias column x0 = 1
print("Design matrix with bias:", Xb.shape)

# %% normal_equation
# theta = (X^T X)^(-1) X^T y  -- the closed-form solution
theta_ne = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y100
theta_pinv = np.linalg.pinv(Xb) @ y100              # SVD-based, safe when X^T X is singular
lr = LinearRegression().fit(Xn, y100)
print("%-22s %s" % ("normal equation:", np.round(theta_ne, 4)))
print("%-22s %s" % ("pseudo-inverse (SVD):", np.round(theta_pinv, 4)))
print("%-22s %s" % ("sklearn LinearRegression:", np.round(np.r_[lr.intercept_, lr.coef_], 4)))
print("Max difference between the three:", np.abs(theta_ne - np.r_[lr.intercept_, lr.coef_]).max().round(10))
print("RMSE in $: %.0f" % (100000 * rmse(y100, Xb @ theta_ne)))
for c, w in zip(["bias"] + cols, theta_ne):
    print(f"   {c:26s} {w * 1e5:+10.0f} $ per +1 std-dev")

# %% batch_gd
# Batch gradient descent: theta <- theta - eta * (2/m) X^T (X theta - y)
def batch_gd(Xb, y, eta, n_iter=200):
    m = len(Xb); theta = np.zeros(Xb.shape[1]); hist = []
    for _ in range(n_iter):
        err = Xb @ theta - y
        hist.append(np.mean(err ** 2))
        theta = theta - eta * (2 / m) * Xb.T @ err
    return theta, np.array(hist)

etas = [0.002, 0.02, 0.2, 1.0]
fig, ax = plt.subplots(figsize=(6.8, 4.3))
for eta, c in zip(etas, [PALETTE["grey"], PALETTE["orange"], PALETTE["green"], PALETTE["red"]]):
    th, h = batch_gd(Xb, y100, eta)
    status = "DIVERGES" if (not np.isfinite(h[-1]) or h[-1] > h[0]) else "converges"
    print(f"eta = {eta:<6}: final MSE = {h[-1]:.4g}  |theta - theta_NE| = {np.linalg.norm(th - theta_ne):.4f}  -> {status}")
    ax.plot(np.clip(h, 0, 1e6), label=f"eta={eta}", color=c, lw=2)
ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("MSE (log scale)")
ax.set_title("Learning-rate effect on batch gradient descent"); ax.legend()
savefig(fig, "fig11_gd_learning_rate")
th_bgd, h = batch_gd(Xb, y100, 0.2, n_iter=1000)
print("\nAfter 1000 iterations at eta=0.2:  |theta - theta_NE| =", np.linalg.norm(th_bgd - theta_ne).round(8))

# %% sgd_minibatch
rng = np.random.RandomState(RANDOM_STATE)
def schedule(t, t0=5, t1=50): return t0 / (t + t1)

def sgd(Xb, y, epochs=30):
    m = len(Xb); theta = np.zeros(Xb.shape[1]); path = []
    for epoch in range(epochs):
        for i in range(m):
            k = rng.randint(m)
            g = 2 * Xb[k] * (Xb[k] @ theta - y[k])
            theta = theta - schedule(epoch * m + i) * g
        path.append(np.mean((Xb @ theta - y) ** 2))
    return theta, path

def minibatch(Xb, y, epochs=30, bs=64, eta=0.05):
    m = len(Xb); theta = np.zeros(Xb.shape[1]); path = []
    for epoch in range(epochs):
        order = rng.permutation(m)
        for s in range(0, m, bs):
            b = order[s:s + bs]
            theta = theta - eta * (2 / len(b)) * Xb[b].T @ (Xb[b] @ theta - y[b])
        path.append(np.mean((Xb @ theta - y) ** 2))
    return theta, path

t0 = time.time(); th_sgd, p_sgd = sgd(Xb, y100, epochs=10); t_sgd = time.time() - t0
t0 = time.time(); th_mb, p_mb = minibatch(Xb, y100); t_mb = time.time() - t0
mse_opt = np.mean((Xb @ theta_ne - y100) ** 2)
print("Optimal MSE (normal equation): %.4f" % mse_opt)
print("SGD  (10 epochs, decaying eta):  final MSE %.4f  |  %.1fs  | distance to optimum %.4f" % (p_sgd[-1], t_sgd, np.linalg.norm(th_sgd - theta_ne)))
print("Mini-batch (30 epochs, bs=64):   final MSE %.4f  |  %.1fs  | distance to optimum %.4f" % (p_mb[-1], t_mb, np.linalg.norm(th_mb - theta_ne)))
fig, ax = plt.subplots(figsize=(6.8, 4.0))
ax.plot(p_sgd, "o-", label="SGD (1 sample/step)", color=PALETTE["orange"]); ax.plot(p_mb, "s-", label="Mini-batch (64)", color=PALETTE["blue"])
ax.axhline(mse_opt, color=PALETTE["red"], ls="--", label="closed-form optimum"); ax.set_xlabel("epoch"); ax.set_ylabel("training MSE")
ax.set_title("Stochastic and mini-batch GD approach the same optimum"); ax.legend()
savefig(fig, "fig12_sgd_minibatch")

# %% learning_curves
# Same data, three models of different capacity. Watch the gap between the two curves.
f4 = ["num__median_income", "num__housing_median_age", "num__latitude", "num__longitude"]
X4 = np.clip(T.X[:, [T.feature_names.index(c) for c in f4]], -2.5, 2.5)
inc = X4[:, [0]]; yy = T.y_value.to_numpy()
setups = [("Underfit: line on income only", inc, 1), ("Better: line on 4 features", X4, 1), ("Overfit: degree-4 on 4 features", X4, 4)]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for a, (ttl, data, deg) in zip(ax, setups):
    model = make_pipeline(PolynomialFeatures(deg, include_bias=False), StandardScaler(), LinearRegression())
    sizes, tr, va = learning_curve(model, data, yy, train_sizes=np.linspace(0.02, 1, 12), cv=5,
                                   scoring="neg_root_mean_squared_error", shuffle=True, random_state=0)
    a.plot(sizes, -tr.mean(1), "o-", label="train", color=PALETTE["blue"]); a.plot(sizes, -va.mean(1), "o-", label="validation", color=PALETTE["orange"])
    a.set_title(ttl, fontsize=10); a.set_xlabel("training set size"); a.set_ylim(50000, 140000)
    print(f"{ttl:34s}: final train RMSE = {-tr.mean(1)[-1]:6.0f}, validation RMSE = {-va.mean(1)[-1]:6.0f}, gap = {(-va.mean(1)[-1]) - (-tr.mean(1)[-1]):5.0f} | validation at n={sizes[0]}: {-va.mean(1)[0]:.0f}")
ax[0].set_ylabel("RMSE ($)"); ax[0].legend()
savefig(fig, "fig13_learning_curves")

# %% regularisation
# A tiny training set (150 districts) + a flexible model (degree 4 -> 69 features) = guaranteed overfit.
rng_ = np.random.RandomState(0); perm = rng_.permutation(len(X4)); tr_i, va_i = perm[:150], perm[300:4300]
Xtr, Xva, ytr, yva = X4[tr_i], X4[va_i], yy[tr_i], yy[va_i]
def poly_model(est): return make_pipeline(PolynomialFeatures(4, include_bias=False), StandardScaler(), est)
n_feat = PolynomialFeatures(4, include_bias=False).fit(Xtr).n_output_features_
print("Degree-4 polynomial on 4 features ->", n_feat, "features, but only", len(tr_i), "training rows")
unreg = poly_model(LinearRegression()).fit(Xtr, ytr)
print("Unregularised: train RMSE=%.0f | validation RMSE=%.0f   <- severe overfitting" % (rmse(ytr, unreg.predict(Xtr)), rmse(yva, unreg.predict(Xva))))
res = {}
alphas = np.logspace(-1, 4, 16)
for name, mk in [("Ridge (L2)", lambda a: Ridge(alpha=a)), ("Lasso (L1)", lambda a: Lasso(alpha=a, max_iter=50000)),
                 ("Elastic net", lambda a: ElasticNet(alpha=a, l1_ratio=0.5, max_iter=50000))]:
    tr_e, va_e, nz = [], [], []
    for a in alphas:
        m = poly_model(mk(a)).fit(Xtr, ytr)
        tr_e.append(rmse(ytr, m.predict(Xtr))); va_e.append(rmse(yva, m.predict(Xva)))
        nz.append(int(np.sum(np.abs(m[-1].coef_) > 1e-8)))
    res[name] = (np.array(tr_e), np.array(va_e), nz)
    b = int(np.argmin(va_e))
    print(f"{name:12s}: best alpha={alphas[b]:8.1f} -> validation RMSE={va_e[b]:.0f} | non-zero coefficients: {nz[b]} of {n_feat}")
record("reg_unreg_val", rmse(yva, unreg.predict(Xva))); record("reg_ridge_val", float(res["Ridge (L2)"][1].min()))
record("reg_lasso_val", float(res["Lasso (L1)"][1].min())); record("lasso_nonzero", res["Lasso (L1)"][2][int(np.argmin(res["Lasso (L1)"][1]))])
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for (n, (tr_e, va_e, nz)), c in zip(res.items(), [PALETTE["blue"], PALETTE["green"], PALETTE["purple"]]):
    ax[0].plot(alphas, va_e, "o-", label=n, color=c, ms=4)
ax[0].set_xscale("log"); ax[0].set_ylim(60000, 200000); ax[0].set_xlabel("regularisation strength alpha (log)"); ax[0].set_ylabel("validation RMSE ($)"); ax[0].legend(); ax[0].set_title("Validation error vs regularisation")
ax[1].plot(alphas, res["Lasso (L1)"][2], "o-", color=PALETTE["green"]); ax[1].set_xscale("log")
ax[1].set_xlabel("alpha (log)"); ax[1].set_ylabel("# non-zero coefficients"); ax[1].set_title("Lasso performs feature selection")
savefig(fig, "fig14_regularisation")

# %% early_stopping
# Batch/stochastic GD on the SAME overfit-prone problem. Track validation error epoch by epoch.
pre = make_pipeline(PolynomialFeatures(4, include_bias=False), StandardScaler()).fit(Xtr)
Ptr, Pva = pre.transform(Xtr), pre.transform(Xva)
sgd_r = SGDRegressor(penalty=None, learning_rate="constant", eta0=0.001, max_iter=1, tol=None, warm_start=True, random_state=RANDOM_STATE)
va_hist, tr_hist = [], []
for epoch in range(400):
    sgd_r.fit(Ptr, ytr / 1e5)                      # target in $100k units keeps the step size sane
    va_hist.append(rmse(yva, 1e5 * sgd_r.predict(Pva))); tr_hist.append(rmse(ytr, 1e5 * sgd_r.predict(Ptr)))
best_epoch = int(np.argmin(va_hist)); best_val = va_hist[best_epoch]
print("Best validation RMSE %.0f reached at epoch %d (train RMSE there: %.0f)" % (best_val, best_epoch, tr_hist[best_epoch]))
print("After 400 epochs: validation RMSE %.0f (%+.1f%% worse than best), train RMSE %.0f" % (va_hist[-1], 100 * (va_hist[-1] / best_val - 1), tr_hist[-1]))
record("es_best_epoch", best_epoch); record("es_worse_pct", 100 * (va_hist[-1] / best_val - 1))
fig, ax = plt.subplots(figsize=(6.8, 4)); ax.plot(tr_hist, label="train", color=PALETTE["blue"]); ax.plot(va_hist, label="validation", color=PALETTE["orange"])
ax.axvline(best_epoch, color=PALETTE["red"], ls="--", label=f"stop here (epoch {best_epoch})"); ax.set_ylim(40000, 130000)
ax.set_xlabel("epoch"); ax.set_ylabel("RMSE ($)"); ax.set_title("Early stopping: stop when validation error bottoms out"); ax.legend()
savefig(fig, "fig15_early_stopping")

# %% logistic
# Logistic regression from scratch on 2 features: P(premium) = sigmoid(theta . x)
f2 = [T.feature_names.index("num__median_income"), T.feature_names.index("num__latitude")]
X2 = np.c_[np.ones(len(T.X)), T.X[:, f2]]; y2 = T.y_premium
sigmoid = lambda z: 1 / (1 + np.exp(-z))
theta = np.zeros(3)
for it in range(3000):
    p = sigmoid(X2 @ theta)
    theta -= 0.3 * X2.T @ (p - y2) / len(y2)         # gradient of the log-loss
ll_scratch = log_loss(y2, sigmoid(X2 @ theta))
sk = LogisticRegression(C=1e9, max_iter=1000).fit(X2[:, 1:], y2)
print("from scratch: theta =", theta.round(3), "| log-loss = %.4f" % ll_scratch)
print("scikit-learn: theta =", np.r_[sk.intercept_, sk.coef_[0]].round(3), "| log-loss = %.4f" % log_loss(y2, sk.predict_proba(X2[:, 1:])))
print("Accuracy of the 2-feature model: %.3f" % ((sk.predict(X2[:, 1:]) == y2).mean()))
inc_idx = T.feature_names.index("num__median_income")
print("A +1 std-dev rise in income multiplies the odds of being premium by x%.2f" % np.exp(theta[1]))
g = np.linspace(-2.5, 6, 200); l = np.linspace(-1.5, 1.8, 200); G, L = np.meshgrid(g, l)
Z = sk.predict_proba(np.c_[G.ravel(), L.ravel()])[:, 1].reshape(G.shape)
fig, ax = plt.subplots(figsize=(6.6, 4.8))
ax.scatter(X2[y2 == 0, 1], X2[y2 == 0, 2], s=3, alpha=.25, color=PALETTE["grey"], label="not premium")
ax.scatter(X2[y2 == 1, 1], X2[y2 == 1, 2], s=3, alpha=.35, color=PALETTE["orange"], label="premium")
cs = ax.contour(G, L, Z, levels=[0.15, 0.5, 0.85], colors=[PALETTE["blue"]], linewidths=[1, 2.5, 1]); ax.clabel(cs, fmt="%.2f")
ax.set_xlabel("median income (standardised)"); ax.set_ylabel("latitude (standardised)"); ax.set_xlim(-2.5, 6); ax.set_ylim(-1.5, 1.8)
ax.set_title("Logistic regression decision boundary (p = 0.5 is bold)"); ax.legend(markerscale=4, loc="upper right")
savefig(fig, "fig16_logistic_boundary")

# %% softmax
yt = T.y_tier
sm = LogisticRegression(C=10, max_iter=3000).fit(T.X, yt)         # multi-class -> softmax (multinomial) by default
print("classes:", sm.classes_, "| coefficient matrix shape (K classes x n features):", sm.coef_.shape)
z = sm.decision_function(T.X[:2])
manual = np.exp(z) / np.exp(z).sum(1, keepdims=True)
print("\nScores z for district 0:", z[0].round(2))
print("softmax by hand        :", manual[0].round(3))
print("sklearn predict_proba  :", sm.predict_proba(T.X[:1])[0].round(3), " (identical)")
print("true tier of district 0:", yt[0])
cv = cross_val_score(LogisticRegression(C=10, max_iter=3000), T.X, yt, cv=5).mean()
print("Softmax regression 5-fold accuracy: %.3f" % cv)
record("softmax_acc", cv)
