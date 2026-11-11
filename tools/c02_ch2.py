from pdflib import *

S2, S3 = "step02_eda", "step03_pipeline_models"


def build():
    f = [H1("Chapter 2 · End-to-End Machine Learning Project"),
         P("**Project steps 2 and 3** — everything a real project does between 'we have a CSV' and 'we have a validated model': look at the data, build an untouchable test set, explore, preprocess with pipelines, "
           "train and cross-validate several models, fine-tune, and analyse errors. Code: `steps/step02_eda.py`, `steps/step03_pipeline_models.py` (and the final evaluation in Step 11, Chapter 10).")]

    f.append(H2("2.1  Concepts you must own"))
    f.append(H3("The workflow"))
    f += NL(["Look at the big picture and frame the problem (Chapter 1).", "Get the data.", "Explore and visualise it to gain insights (on the **training set only**).",
             "Prepare the data for the algorithms (cleaning, encoding, scaling, feature engineering) **in a pipeline**.", "Select and train models; compare with cross-validation.",
             "Fine-tune hyper-parameters; analyse the errors; evaluate once on the test set.", "Present, launch, monitor and maintain the system."])
    f.append(H3("Creating a test set the right way"))
    f.append(P("Set aside the test set **before** you explore. Your brain is a pattern detector: if you study the test data you will subconsciously pick models that suit it (*data snooping bias*). "
               "A split must be (a) **random**, (b) **stable**, i.e. the same rows land in the test set every run, which you get from a fixed seed or from hashing a row identifier, and (c) **representative**."))
    f.append(P("A purely random split of 20,640 rows is usually fine, but when a *single attribute dominates* the target (here median income, correlation ≈ 0.69), a random split can accidentally under- or over-represent some income ranges, biasing your estimate. "
               "**Stratified sampling** divides the population into homogeneous subgroups (*strata*) and samples the right proportion from each. We bin income into five categories solely to stratify; the category is never used as a feature."))
    f.append(H3("Exploring"))
    f += BL(["**Look** at `head()`, `info()` (types, non-null counts), `describe()` (scale, range, percentiles), and histograms of every attribute. Look for caps, heavy tails, skew, different scales and suspicious values.",
             "**Visualise geography** with a scatter plot (alpha for density, size = population, colour = target).",
             "**Correlations**: the standard (Pearson) correlation coefficient measures only *linear* relationships, in [−1, 1]. A correlation near 0 does not mean 'no relationship' (an inverted U gives 0). Plot the scatter matrix too.",
             "**Experiment with attribute combinations**: totals per district are less useful than ratios, such as rooms per household or the share of rooms that are bedrooms."])
    f.append(H3("Preparing the data: the toolbox"))
    f += table([
        ["Problem", "Tool", "Notes"],
        ["Missing values", "`SimpleImputer(strategy='median')` (also mean, most_frequent, constant); `KNNImputer`, `IterativeImputer`", "Fit on train only; stores the medians so the same values fill the test set. Median is robust to outliers"],
        ["Categorical text", "`OneHotEncoder` (nominal), `OrdinalEncoder` (ordered)", "One-hot yields a sparse matrix; use `handle_unknown='ignore'` for production safety. Avoid ordinal codes for unordered categories: the model will assume 'close'"],
        ["Different scales", "`MinMaxScaler` (→ [0,1]), `StandardScaler` (mean 0, std 1)", "Standardisation is less affected by outliers. Needed by gradient descent, SVMs, k-NN, PCA, regularised models. Trees do not need it"],
        ["Heavy right tail", "`np.log`, square root, **bucketising**, RBF similarity", "Gets features closer to a bell shape; otherwise most models see a few extreme points dominate"],
        ["Multimodal distributions", "Bucketise, or add RBF similarity to the modes", "Lets a linear model see 'near the peak' as a feature"],
        ["Transformed targets", "`TransformedTargetRegressor`", "If you log the label, you must invert predictions; this wrapper does it automatically"],
        ["Geographic structure", "`ClusterSimilarity` (k-means + RBF)", "A custom transformer in `features.py`: location becomes 'how close to each market centre'"]],
        widths=[1.4, 3, 3.4], font=7.6)
    f.append(P("**Transformer API.** Every transformer has `fit()` (learn from data), `transform()` (apply), and `fit_transform()`. A custom transformer needs only `fit` and `transform`; inheriting `BaseEstimator` gives `get_params/set_params`, and `TransformerMixin` gives `fit_transform`. "
               "Hyper-parameters belong in `__init__`, learned values end with an underscore (`kmeans_`), the classic scikit-learn convention."))
    f.append(P("**Pipelines.** A `Pipeline` chains steps so that `fit` runs each transformer's `fit_transform` in order and then the final estimator's `fit`. `ColumnTransformer` applies *different* pipelines to *different columns* and concatenates the results. "
               "The decisive benefit is **leak-free validation**: when you cross-validate a pipeline, every fold re-fits imputers, scalers and encoders on that fold's training part only. If you scale the whole dataset first and cross-validate afterwards, "
               "information from the validation fold has already leaked into the scaler."))
    f += callout("warn", "Data leakage: the silent killer", [
        "Leakage means information from outside the training set reaches the model during training. Typical sources: scaling/imputing before splitting, selecting features using all labels, building time-based features that look into the future, "
        "duplicates present in both train and test, and **target-derived features**. Symptom: validation scores that look too good and then collapse in production. **Defence: split first, then do everything inside a Pipeline.**"])
    f.append(H3("Selecting, validating and tuning"))
    f += BL(["**Cross-validation** gives a mean *and* a standard deviation of the score: a measure of how stable the estimate is. Scikit-learn expects 'higher is better' so RMSE appears as `neg_root_mean_squared_error`.",
             "A model with **training error ≈ 0** is almost certainly overfitting (a decision tree with no limits memorises the data). Only cross-validation reveals it.",
             "**Grid search** tries every combination of a grid; **randomised search** samples combinations from distributions: better for many hyper-parameters or continuous ones. Include *preprocessing* choices (number of clusters, imputation strategy) in the search; they are hyper-parameters too.",
             "**Analyse errors**: feature importances, error by subgroup, and a look at the worst districts. Ask 'what is the model systematically missing?'",
             "**Evaluate on the test set once.** Report a **confidence interval** for the generalisation error, e.g. a 95% t-interval for the RMSE derived from the squared errors.",
             "**Launch**: persist the full pipeline with `joblib` (so preprocessing travels with the model), expose it behind an API or batch job, write monitoring for input drift and performance, retrain on fresh data, and keep dated backups of data and models."])

    f.append(H2("2.2  Project step 2 — look at the data and protect the test set"))
    f += code(S2, "setup"); f += code(S2, "quick_look")
    f += output(S2, "quick_look", 30)
    f.append(P("`info()` tells us the dataset has 20,640 rows and 10 attributes, and that `total_bedrooms` has fewer non-null entries (20,433): **207 missing values**. The only non-numeric column is `ocean_proximity`. "
               "(In pandas 3 text columns display as `str`; scikit-learn treats them as categorical text for one-hot encoding.)"))
    f += code(S2, "describe"); f += output(S2, "describe")
    f += code(S2, "missing_and_caps"); f += output(S2, "missing_and_caps")
    f.append(P(f"Two important quirks are visible. **First**, the maximum `median_house_value` is exactly 500,001 and about **{R['pct_price_cap']:.1f}%** of districts sit at that value: the data is *censored* (the true value is higher but was recorded as the cap). "
               "A model can never learn to predict above the cap; we will quantify this later. **Second**, `housing_median_age` is capped at 52. The business must be told: if the client needs predictions above $500k, we need correct labels (collect them) or must drop capped districts and limit our claims."))
    f += fig("fig02_histograms", "**Figure 2.** Histograms of all numeric attributes. Note the spike at the price cap, the age cap at 52, the strong right-skew of rooms/bedrooms/population/households, and very different scales.")
    f.append(P("The skew means a few huge districts will dominate un-transformed features: we will log-transform them. The different scales (income 0–15, rooms in the tens of thousands) mean scaling is mandatory for gradient-based and distance-based models."))
    f.append(H3("A stratified split"))
    f += code(S2, "split_bias"); f += output(S2, "split_bias")
    f.append(P(f"The 'Random err %' column shows what happens to the test-set income mix under a purely random split: some income categories are over- or under-represented by up to **{R['split_random_maxerr']:.1f}%**. "
               f"With stratification the worst deviation is only **{R['split_strat_maxerr']:.2f}%**. Because income is the strongest predictor, a biased income mix would bias the performance estimate, so this matters. "
               "From now on `split()` in `data.py` returns the stratified 80/20 split, and the test part is not looked at again until Step 11."))
    f += code(S2, "lock_test")
    f += output(S2, "lock_test")
    f.append(H3("Visualising geography"))
    f += code(S2, "geo", stop_at="for a in ax")
    f += fig("fig03_geo", "**Figure 3.** Left: the density of districts reveals the Bay Area and Los Angeles/San Diego. Right: price is highest near the coast and in the big metros, but 'near the ocean' is not a perfect rule (northern coastal prices are lower), which is why we model location with clusters instead of a single rule.")
    f.append(H3("Correlations and attribute combinations"))
    f += code(S2, "correlations"); f += output(S2, "correlations")
    f += code(S2, "engineered"); f += output(S2, "engineered")
    f.append(P(f"Income dominates (r = {R['corr_income']:.2f}). Raw `total_rooms` is nearly useless (r = {R['corr_total_rooms']:.2f}) because a big district has many rooms whatever its price, but the engineered **bedrooms_ratio** reaches r = {R['corr_bedrooms_ratio']:.2f}: "
               "districts where bedrooms are a small share of the rooms tend to be wealthier. *Ratios express the real phenomenon better than totals.* That insight goes into the pipeline."))
    f += fig("fig04_scatter_matrix", "**Figure 4.** Scatter matrix of the most promising attributes (sampled for clarity).", 0.78)
    f += fig("fig05_income_price", "**Figure 5.** The income–price relationship is strong and roughly linear, with horizontal lines of points at the price cap (and weaker ones at round values like $350k, $450k): quirks that could mislead a model if not handled.", 0.62)

    f.append(H2("2.3  Project step 3 — pipelines, models, fine-tuning, error analysis"))
    f.append(H3("The preprocessing pipeline"))
    f.append(P("Here is the pipeline from `features.py`. Study its structure: it is the pattern you will reuse in every project."))
    f += snippet("""
def build_preprocessing(n_clusters=10, gamma=1.0, use_clusters=True) -> ColumnTransformer:
    transformers = [
        ("bedrooms",         ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house",  ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline(), ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
    ]
    if use_clusters:
        transformers.append(("geo", ClusterSimilarity(n_clusters, gamma), ["latitude", "longitude"]))
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_exclude="number")))
    default = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())   # for housing_median_age
    return ColumnTransformer(transformers, remainder=default)

def ratio_pipeline():   # impute median -> divide column 0 by column 1 -> standardise
    return make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(_column_ratio), StandardScaler())
def log_pipeline():     # impute median -> natural log -> standardise
    return make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log), StandardScaler())
""", "CODE · src/homelens/features.py (abridged)")
    f.append(P("Read it as a data-flow diagram. Three ratio features are computed; five heavy-tailed attributes are log-transformed; latitude/longitude become *similarity to each of n k-means market centres* by `ClusterSimilarity`; "
               "the category is one-hot encoded; and the one remaining numeric column goes through the default imputer + scaler. Every numeric path ends in standardisation, and every imputer uses the median."))
    f += snippet("""
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=42): ...
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)      # learn the market centres
        return self
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)   # exp(-gamma * distance^2)
""", "CODE · the custom transformer (abridged)")
    f.append(P("The RBF kernel turns distance *d* to a centre into a similarity exp(−γd²) between 0 (far) and 1 (at the centre). A linear model or a tree then sees 'how Bay-Area-like is this district?' as a smooth feature. "
               "Because it is *inside* the pipeline, the k-means is refit on each training fold, so there is no leakage."))
    f += code(S3, "setup", skip_lines=0); f += code(S3, "preprocess_demo"); f += output(S3, "preprocess_demo")
    f.append(P("Nine raw columns became **24** model-ready features, with no NaNs left, and the ratio columns have mean 0 and standard deviation 1, which is exactly what standardisation promises. "
               "`get_feature_names_out()` keeps the traceability: `bedrooms__ratio`, `log__median_income`, `geo__geo_cluster_16_sim`, `cat__ocean_proximity_INLAND`…"))
    f.append(H3("Training and cross-validating four models"))
    f += code(S3, "cv_models"); f += output(S3, "cv_models")
    f.append(P("This table is one of the most important in the whole project; read it slowly."))
    f += BL(["**Mean baseline** — CV RMSE $115,700: training and validation agree, and it is just the target's standard deviation.",
             "**Linear regression** — train $69,240, CV $71,698: errors are similar (no overfitting) but *high*: the model is **underfitting**. A line cannot capture the geography and the non-linear income effect.",
             "**Decision tree** — train RMSE is **0** (it memorised the 16,512 rows!) but CV RMSE is $67,431: a perfect example of overfitting. Without cross-validation you would have believed it was perfect.",
             "**Random forest** — train $17,612 versus CV $47,630: it still overfits (gap), but its CV error is by far the best because averaging many decorrelated trees removes variance (Chapter 7)."])
    f.append(H3("Fine-tuning with randomised search"))
    f += code(S3, "finetune"); f += output(S3, "finetune")
    f.append(P(f"The search explored the number of geographic clusters, the number of features each split may consider (`max_features`), and `min_samples_leaf`, in **one** pass, *because the preprocessing is part of the pipeline*. "
               f"The best configuration reached a CV RMSE of **${R['ch2_best_cv']:,.0f}** (from ${47630:,.0f}), with {R['ch2_best_params']['n_clusters']} clusters: more geographic resolution helped. "
               "Note the tuning used 3-fold CV and 60 trees for speed; the search is a tool to find a *good region*, not to prove the final number."))
    f += callout("tip", "Practical tuning advice", [
        "Start with wide ranges and a coarse search; narrow around the best region; use randomised search when you have more than 2–3 hyper-parameters. When the best value sits at the edge of your range, extend the range. "
        "Remember that tuned hyper-parameters are tuned to the CV folds: always keep a final untouched test set."])
    f.append(H3("What did the forest learn? Feature importance"))
    f += code(S3, "importance", stop_at="fig, ax"); f += output(S3, "importance")
    f += fig("fig06_importance_ch2", "**Figure 6.** Income (log) leads; being INLAND is second; the three engineered ratios follow, and several geographic-cluster similarities carry location information.", 0.62)
    f.append(P("Importances sum to 1 and measure how much each feature reduced impurity across the forest. Useful, but treat it as a *hint*, not proof of causation: correlated features share importance, and impurity-based importance favours features with many distinct values. "
               "It also suggests pruning candidates: features with ~0 importance can be dropped, simplifying the system."))
    f.append(H3("Error analysis"))
    f += code(S3, "error_analysis", stop_at="fig, ax"); f += output(S3, "error_analysis")
    f += fig("fig07_error_analysis", "**Figure 7.** Left: predictions versus truth hug the diagonal but flatten at the cap. Right: large errors cluster in dense metropolitan areas where prices vary hugely over short distances.")
    f.append(P(f"We used **out-of-fold predictions** (`cross_val_predict`): each district is predicted by a model that never saw it, so these errors are honest. Three findings: "
               f"(1) coastal and bay districts have larger absolute errors than inland ones, because prices there are higher and more variable; (2) the **capped** districts (true value ≥ $500,001) get a mean prediction of about ${R['capped_mean_pred']:,.0f}: the model cannot see beyond the cap; "
               "(3) the geographic error map says 'add finer location information or neighbourhood-level data'. These are the kinds of statements a stakeholder can act on, and they come from error analysis, not from a single number."))

    f.append(H2("2.4  Hands-on experiments"))
    f += NL(["In `step03`, replace `RandomForestRegressor` by `HistGradientBoostingRegressor` inside the same pipeline. How much does CV RMSE change?",
             "Remove the three ratio pipelines from `build_preprocessing`. Re-run the CV. What happens? (Which ratio mattered most?)",
             "Deliberately create leakage: scale all of `X` with `StandardScaler` *before* `cross_val_score`. Compare the score with the pipeline version; explain the difference (it will be small here, but explain why it is not zero).",
             "Change `OneHotEncoder` to `OrdinalEncoder`. Which models are hurt, and why?",
             "Use `TransformedTargetRegressor` with `np.log`/`np.exp` on the price. Does RMSE in dollars improve?"])

    f.append(H2("2.5  Challenge questions"))
    f += QA([
        ("Why must you fit the scaler on the training data only?", "The scaler's mean/std summarise the data it is fitted on. Fitting on all the data puts test-set statistics inside the training procedure (**leakage**), making validation optimistic. Fit on train, then only *transform* validation/test data with those stored values."),
        ("Explain stratified sampling and when it is needed.", "Divide the population into strata (homogeneous groups) and sample the correct proportion from each, so the sample matches the population on the stratifying attribute. Needed when one attribute is highly predictive and a random split could mis-represent it, especially with small datasets or rare classes."),
        ("A decision tree has RMSE 0 on training data. Is it a great model?", "Almost certainly not: it has memorised the training set (overfitting). Judge it by cross-validated or test error. Here CV RMSE is $67k, worse than the forest."),
        ("What is the difference between `fit_transform` on train and `transform` on test?", "`fit_transform` learns parameters (medians, means, category lists, cluster centres) and applies them. `transform` re-uses the *stored* parameters without learning, so new data is treated exactly like the training data."),
        ("Why log-transform skewed features?", "Heavy tails let a few extreme values dominate scaling, distances and gradient steps. A log pulls the tail in, making the distribution closer to symmetric so standard scaling and linear relationships work better. Alternatives: bucketising, or a power transform."),
        ("Grid search versus randomised search?", "Grid tries all combinations (cost grows multiplicatively with the number of hyper-parameters) and only values you list. Randomised samples a fixed budget from distributions: more efficient in high dimensions, explores continuous ranges, and the budget is directly controlled."),
        ("Why use `handle_unknown='ignore'` in the one-hot encoder?", "In production a category unseen in training may appear. With `ignore`, it becomes a row of zeros instead of raising an error that crashes the service."),
        ("Why does the median beat the mean for imputation here?", "The attributes are heavily skewed with outliers, and the mean is pulled by them; the median is a typical value and robust."),
        ("What does 'the data is censored at 500,001' imply for the model and for the business?", "Labels are lower bounds for those districts, so the model systematically under-predicts in the most expensive areas and cannot be trusted above the cap. Either obtain true values, drop capped rows and restrict the claimed range, or model the censoring explicitly (survival-style methods)."),
        ("Why persist the entire pipeline instead of only the model?", "At inference, raw data must go through *exactly* the same preprocessing as in training. A saved pipeline guarantees it, eliminating training/serving skew."),
        ("What would you monitor after deployment?", "Input distributions (drift), prediction distributions, error against delayed ground truth, data-quality checks (missing rate, caps, new categories), latency, and the age of the model with an automatic retraining trigger.")])
    f.append(PageBreak())
    return f
