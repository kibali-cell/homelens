"""STEP 1 - Chapter 1: The Machine Learning Landscape.
Frame the problem, define the metric, build baselines, and SEE under/overfitting."""
# %% setup
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from homelens.data import split, xy
from homelens.utils import set_style, savefig, record, rmse, PALETTE
set_style()
train, test = split()
X_train, y_train = xy(train)
print("train rows:", len(train), "| test rows (locked away):", len(test))
print("target mean = $%.0f, std = $%.0f" % (y_train.mean(), y_train.std()))

# %% baseline
# A baseline is the dumbest model that could possibly work. Every real model must beat it.
dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
base_rmse = rmse(y_train, dummy.predict(X_train))
print("Baseline 1 - predict the average price for everyone: RMSE = $%.0f" % base_rmse)

lin = LinearRegression().fit(X_train[["median_income"]], y_train)
lin_rmse = rmse(y_train, lin.predict(X_train[["median_income"]]))
print("Baseline 2 - one-feature linear model on median_income: RMSE = $%.0f" % lin_rmse)
print("   learned: price = %.0f + %.0f * income" % (lin.intercept_, lin.coef_[0]))
record("baseline_mean_rmse", base_rmse); record("baseline_income_rmse", lin_rmse)

# %% bias_variance
# Take only 40 training districts (small data = easy to overfit) and fit polynomials of
# increasing degree. Compare error on the data the model saw vs. on unseen districts.
rng = np.random.RandomState(7)
small = train.sample(40, random_state=7)
# unseen districts, restricted to the income range the model was trained on (no extrapolation)
lo, hi = small["median_income"].min(), small["median_income"].max()
val = train.drop(small.index).query("@lo <= median_income <= @hi").sample(2000, random_state=7)
xs, ys = small[["median_income"]].to_numpy(), small["median_house_value"].to_numpy()
xv, yv = val[["median_income"]].to_numpy(), val["median_house_value"].to_numpy()

degrees = [1, 2, 3, 5, 8, 12, 16, 20]
rows = []
for d in degrees:
    m = make_pipeline(MinMaxScaler((-1, 1)), PolynomialFeatures(d, include_bias=False), LinearRegression())
    m.fit(xs, ys)
    rows.append((d, rmse(ys, m.predict(xs)), rmse(yv, m.predict(xv))))
tab = pd.DataFrame(rows, columns=["degree", "train_RMSE", "unseen_RMSE"]).round(0).astype({"degree": int})
print(tab.to_string(index=False))
best = tab.loc[tab.unseen_RMSE.idxmin(), "degree"]
print("\nLowest error on unseen data at degree", best)
record("bv_table", tab.to_dict("list"))

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
grid = np.linspace(xs.min(), xs.max(), 300).reshape(-1, 1)
for a, d, title in zip(ax[:2], [1, 16], ["Underfit (degree 1)", "Overfit (degree 16)"]):
    m = make_pipeline(MinMaxScaler((-1, 1)), PolynomialFeatures(d, include_bias=False), LinearRegression()).fit(xs, ys)
    a.scatter(xs, ys, s=18, color=PALETTE["grey"], label="40 training districts")
    a.plot(grid, m.predict(grid), color=PALETTE["red"], lw=2)
    a.set_ylim(0, 550000); a.set_title(title); a.set_xlabel("median income (x $10k)")
ax[0].set_ylabel("median house value ($)")
ax[2].plot(tab.degree, tab.train_RMSE, "o-", label="train error", color=PALETTE["blue"])
ax[2].plot(tab.degree, tab.unseen_RMSE, "o-", label="unseen error", color=PALETTE["orange"])
ax[2].set_yscale("log"); ax[2].set_xlabel("model complexity (polynomial degree)")
ax[2].set_ylabel("RMSE ($, log scale)"); ax[2].set_title("The bias-variance U-curve"); ax[2].legend()
savefig(fig, "fig01_bias_variance")

# %% data_size
# More data is often the cheapest cure for overfitting: re-fit the degree-12 model with growing n.
rows = []
pool = train.drop(small.index)
for n in [20, 40, 80, 300, 1000, 5000]:
    s = pool.sample(n, random_state=1)
    m = make_pipeline(MinMaxScaler((-1, 1)), PolynomialFeatures(12, include_bias=False), LinearRegression())
    m.fit(s[["median_income"]].to_numpy(), s["median_house_value"])
    rows.append((n, rmse(s["median_house_value"], m.predict(s[["median_income"]].to_numpy())),
                 rmse(yv, m.predict(xv))))
t2 = pd.DataFrame(rows, columns=["n_train", "train_RMSE", "unseen_RMSE"]).round(0).astype(int)
print(t2.to_string(index=False))
record("datasize_table", t2.to_dict("list"))
