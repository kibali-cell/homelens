"""STEP 2 - Chapter 2 (part 1): Look at the data, create a trustworthy test set, explore."""
# %% setup
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from homelens.data import load_raw, add_income_cat, split
from homelens.utils import set_style, savefig, record, PALETTE
set_style()
df = load_raw()

# %% quick_look
print(df.head(3).T.to_string())
print()
df.info()

# %% describe
pd.set_option("display.width", 130)
print(df.describe().round(1).T.to_string())
print("\nocean_proximity counts:")
print(df["ocean_proximity"].value_counts().to_string())

# %% missing_and_caps
miss = df.isna().sum()
print("Missing values per column:\n", miss[miss > 0].to_string())
print("\nShare of districts whose price sits exactly at the 500,001 cap: %.2f%%"
      % (100 * (df.median_house_value >= 500001).mean()))
print("Share of districts whose age sits at the 52-year cap:        %.2f%%"
      % (100 * (df.housing_median_age >= 52).mean()))
record("n_missing_bedrooms", int(miss["total_bedrooms"]))
record("pct_price_cap", float(100 * (df.median_house_value >= 500001).mean()))

# %% histograms
fig, axes = plt.subplots(3, 3, figsize=(12, 8.5))
cols = [c for c in df.columns if c != "ocean_proximity"]
for ax, c in zip(axes.ravel(), cols):
    ax.hist(df[c].dropna(), bins=50, color=PALETTE["blue"], alpha=.85)
    ax.set_title(c)
fig.suptitle("Distribution of every numeric attribute", fontweight="bold", y=1.0)
fig.tight_layout()
savefig(fig, "fig02_histograms")

# %% split_bias
# Compare a purely random split with a stratified split on income category.
df["income_cat"] = add_income_cat(df)
rand_train, rand_test = train_test_split(df, test_size=0.2, random_state=42)
strat_train, strat_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["income_cat"])
cmp_ = pd.DataFrame({
    "Overall %": df["income_cat"].value_counts(normalize=True).sort_index() * 100,
    "Stratified %": strat_test["income_cat"].value_counts(normalize=True).sort_index() * 100,
    "Random %": rand_test["income_cat"].value_counts(normalize=True).sort_index() * 100,
}).round(2)
cmp_["Random err %"] = ((cmp_["Random %"] / cmp_["Overall %"] - 1) * 100).round(2)
cmp_["Strat err %"] = ((cmp_["Stratified %"] / cmp_["Overall %"] - 1) * 100).round(2)
print(cmp_.to_string())
record("split_random_maxerr", float(cmp_["Random err %"].abs().max()))
record("split_strat_maxerr", float(cmp_["Strat err %"].abs().max()))

# %% lock_test
train, test = split()          # from here on: EXPLORE THE TRAINING SET ONLY
print("train:", train.shape, " test:", test.shape)
explore = train.copy()

# %% geo
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(explore.longitude, explore.latitude, s=4, alpha=.25, color=PALETTE["blue"])
ax[0].set_title("Where the districts are (density shows population centres)")
sc = ax[1].scatter(explore.longitude, explore.latitude, s=explore.population / 100, alpha=.45,
                   c=explore.median_house_value, cmap="viridis")
plt.colorbar(sc, ax=ax[1], label="median house value ($)")
ax[1].set_title("Bubble size = population, colour = price")
for a in ax: a.set_xlabel("longitude"); a.set_ylabel("latitude")
savefig(fig, "fig03_geo")

# %% correlations
num = explore.select_dtypes("number")
corr = num.corr()["median_house_value"].sort_values(ascending=False)
print("Linear correlation with median_house_value (raw attributes):")
print(corr.round(3).to_string())

# %% engineered
explore["rooms_per_house"] = explore.total_rooms / explore.households
explore["bedrooms_ratio"] = explore.total_bedrooms / explore.total_rooms
explore["people_per_house"] = explore.population / explore.households
corr2 = explore.select_dtypes("number").corr()["median_house_value"].sort_values(ascending=False)
print("After adding ratio features:")
print(corr2.round(3).to_string())
record("corr_income", float(corr["median_income"]))
record("corr_bedrooms_ratio", float(corr2["bedrooms_ratio"]))
record("corr_total_rooms", float(corr["total_rooms"]))

# %% scatter_matrix
from pandas.plotting import scatter_matrix
attrs = ["median_house_value", "median_income", "rooms_per_house", "housing_median_age"]
axs = scatter_matrix(explore[attrs].sample(3000, random_state=0), figsize=(9, 8), alpha=.25,
                     color=PALETTE["blue"], hist_kwds={"color": PALETTE["blue"], "bins": 40})
plt.suptitle("Scatter matrix of the most promising attributes", y=0.92, fontweight="bold")
savefig(plt.gcf(), "fig04_scatter_matrix")

# %% income_vs_price
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter(explore.median_income, explore.median_house_value, s=5, alpha=.2, color=PALETTE["blue"])
ax.axhline(500001, color=PALETTE["red"], ls="--", lw=1); ax.text(10, 505000, "price cap $500,001", color=PALETTE["red"], ha="right")
ax.set_xlabel("median income (x $10,000)"); ax.set_ylabel("median house value ($)")
ax.set_title("Strongest signal: income vs. price (note the cap artefact)")
savefig(fig, "fig05_income_price")
