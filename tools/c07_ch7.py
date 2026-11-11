from pdflib import *

S8 = "step08_ensembles"


def build():
    f = [H1("Chapter 7 · Ensemble Learning and Random Forests"),
         P("**Project step 8** — combine many models into one better model, then run the **model bake-off** that decides what HomeLens ships. Code: `steps/step08_ensembles.py`.")]
    f.append(H2("7.1  Concepts you must own"))
    f.append(P("An **ensemble** aggregates the predictions of several predictors (the *wisdom of the crowd*). The idea works because errors that are *independent* partly cancel: a thousand voters each correct 51% of the time give a correct majority about 74% of the time (and 92% for five thousand), if they are independent. "
               "The catch is **independence**: models trained on the same data make correlated errors. So ensemble methods work to *diversify* their members: different algorithms, different data samples, different features, or sequentially focused training."))
    f += table([
        ["Method", "How members differ", "Reduces", "Key facts"],
        ["**Voting**", "Different algorithms on the same data", "Variance, some bias", "*Hard*: majority class. *Soft*: average class probabilities (needs `predict_proba`; usually better since confident votes weigh more)"],
        ["**Bagging** (bootstrap aggregating)", "Same algorithm, each on a bootstrap sample (with replacement)", "**Variance**", "Parallelisable. Each tree sees ~63% of rows; the other ~37% (**out-of-bag, OOB**) give a free validation score (`oob_score=True`). *Pasting* = without replacement"],
        ["**Random forest**", "Bagged trees + only a random subset of features at each split (`max_features`)", "Variance, further decorrelates trees", "Strong default on tabular data. Feature importance by average impurity decrease. **Extra-Trees**: random thresholds too: faster, higher bias, lower variance"],
        ["**Boosting**", "Models trained *sequentially*; each fixes the previous one's errors", "**Bias** (and some variance)", "Cannot be parallelised across rounds. Tune `learning_rate` (shrinkage) with `n_estimators`; use early stopping"],
        ["**Stacking**", "Heterogeneous models + a **meta-learner** trained on their out-of-fold predictions", "Both", "A blender learns how much to trust each model. Needs out-of-fold predictions to avoid leakage"]],
        widths=[1.3, 2, 1.2, 3.3], font=7.5)
    f.append(H3("Boosting in detail"))
    f.append(P("**AdaBoost** trains a weak learner, increases the weights of the instances it got wrong, trains the next learner on the re-weighted data, and so on; the final prediction is a weighted vote (better learners weigh more). "
               "**Gradient Boosting** instead fits each new tree to the **residual errors** of the current ensemble: prediction = tree<sub>1</sub> + η·tree<sub>2</sub> + η·tree<sub>3</sub> + …, where η is the **learning rate**. "
               "For squared error the residuals are exactly the negative gradient of the loss, which is why it is *gradient* boosting. A **small η needs more trees but generalises better** (shrinkage as regularisation); `n_iter_no_change` stops adding trees when validation error stalls. "
               "**Histogram-based gradient boosting** (`HistGradientBoosting*`, the idea behind XGBoost, LightGBM and CatBoost) bins features into 255 buckets: dramatically faster, supports missing values natively. It is the usual first choice for tabular data."))
    f += callout("key", "Why bagging cuts variance but boosting cuts bias", "Averaging many *high-variance, low-bias* learners (deep trees) cancels their noise: variance falls roughly as 1/N for uncorrelated members. Boosting chains *low-variance, high-bias* learners (shallow trees) so each one corrects what the previous missed: bias falls, and with it overall error; but because it fits residuals it can **overfit if run too long** (bagging cannot overfit by adding trees).")
    f.append(H2("7.2  Project step 8 — hands on"))
    f += code(S8, "setup", skip_lines=0)
    f.append(H3("Why the crowd is wise (and when it is not)"))
    f += code(S8, "law_of_large_numbers"); f += output(S8, "law_of_large_numbers")
    f += fig("fig22_wisdom_crowd", "**Figure 22.** Probability that a majority of independent voters is right, as the number of voters grows. Weakly accurate voters (51%) need thousands; moderately accurate ones (65%) need only a few dozen.", 0.55)
    f.append(H3("Voting"))
    f += code(S8, "voting"); f += output(S8, "voting")
    f.append(P("Three different models (logistic regression, random forest, k-NN) each score 87–91%. The voting ensembles reach 89.3% (hard) and 89.6% (soft). Notice a subtle, important point: **an ensemble is not guaranteed to beat its best member**. Here the random forest alone (90.6%) beats the vote, because the other two voters are weaker and make correlated errors. "
               "Ensembling helps most when members are *about equally good and diverse*."))
    f.append(H3("Bagging and the free OOB estimate"))
    f += code(S8, "bagging_oob"); f += output(S8, "bagging_oob")
    f.append(P(f"A single tree scores 86.2% on validation. Bagging 200 trees lifts it to **{100 * R['bag_val']:.1f}%**: a four-point gain from averaging alone. The **OOB score ({100 * R['bag_oob']:.1f}%) predicted the validation accuracy almost exactly**, so with bagging you can skip a separate validation set. "
               "The check on the bootstrap shows that each sample leaves out ≈ 36.8% of rows (1/e): that is where OOB instances come from."))
    f.append(H3("Random forests and feature importance"))
    f += code(S8, "random_forest"); f += output(S8, "random_forest")
    f += fig("fig23_rf_importance", "**Figure 23.** Random-forest feature importance for the premium classifier.", 0.55)
    f.append(P(f"The random forest reaches **{100 * R['rf_clf_val']:.1f}%** validation accuracy (OOB 90.7%). Accuracy saturates around 100 trees (see the last printed line): *adding trees never overfits*, it just stops helping. "
               "Income leads the importances, but now location (longitude, latitude), `people_per_house` and `bedrooms_ratio` all matter: the forest uses interactions that the linear model could not."))
    f.append(H3("Boosting"))
    f += code(S8, "boosting"); f += output(S8, "boosting")
    f += fig("fig24_boosting", "**Figure 24.** Left: validation RMSE versus number of trees for three learning rates. A tiny η (0.02) is still improving at 400 trees; η = 0.5 learns fast but bottoms out at a worse error and then drifts up. Right: boosting by hand, each tree fits the previous residuals.")
    f.append(P("Study the trade-off on the left panel. **η = 0.1 with 400 trees achieves the best error (RMSE $47.5k)**; η = 0.5 reaches its best (49.1k) after only 164 trees and then slowly overfits; η = 0.02 would need many more than 400 trees. The 'by hand' experiment builds gradient boosting from scratch in four lines: "
               "predict the mean (RMSE $115.6k), fit a tree to the residuals (→ $75.0k), fit another to the new residuals (→ $68.5k). **Early stopping** (`n_iter_no_change=15`) picked 327 trees automatically from a 2,000-tree budget: always prefer it to guessing."))
    f.append(H3("The model bake-off"))
    f += code(S8, "model_bakeoff"); f += output(S8, "model_bakeoff")
    f += fig("fig25_bakeoff", "**Figure 25.** Cross-validated RMSE for eight regression models on identical features. Lower is better; error bars show the fold-to-fold standard deviation.", 0.75)
    f.append(P("This table is the **model-selection evidence**. Ranking: histogram gradient boosting ($46.1k) ≈ stacking ($46.3k) < gradient boosting ($47.3k) < random forest ($49.2k) ≈ extra-trees ($49.8k) ≪ a tuned single tree ($60.0k) ≪ linear/ridge ($68.4k). "
               "Three lessons. (1) **Ensembles of trees beat every single model by a wide margin** (46k vs 60k vs 68k). (2) **The differences among the top four are comparable to the fold standard deviation (~$0.9–1.2k)**: do not over-interpret rank differences within noise. "
               "(3) Stacking is 20× slower than histogram boosting for no measurable gain, so we choose the simplest high performer: **HistGradientBoostingRegressor** for the final HomeLens regressor (tuned in Step 11)."))
    f += callout("tip", "A practical rule for tabular data", "Start with a tuned gradient-boosted tree model and a random forest; compare with a regularised linear model as the sanity baseline; stack/blend only if you need the last 1–2%. Spend your effort on **features and validation** before algorithms.")
    f.append(H2("7.3  Hands-on experiments"))
    f += NL(["Set `max_features=1.0` in the random forest (plain bagging of trees). Compare CV RMSE with `max_features=0.5`: why does restricting features help?", "In gradient boosting try `max_depth` 1, 2, 3, 6 at a fixed η. Which depth is best and why does a depth-1 ensemble underfit?",
             "Build your own bagging ensemble of 100 `DecisionTreeRegressor`s with `np.random.choice` bootstrap samples and average their predictions: do you match `BaggingRegressor`?", "Use `StackingRegressor(passthrough=True)`. Does feeding the raw features to the meta-learner help?",
             "Create a deliberately correlated ensemble (10 random forests with the same seed). Show that the ensemble gains nothing: independence matters."])
    f.append(H2("7.4  Challenge questions"))
    f += QA([
        ("Why does averaging many models reduce error? When does it fail?", "If members' errors are not perfectly correlated, averaging cancels part of the noise (variance falls toward ρσ² + (1−ρ)σ²/N). It fails when errors are strongly correlated (same data, same algorithm, same bias) or when members share a systematic bias: averaging cannot remove bias."),
        ("What is the out-of-bag score?", "With bootstrap sampling, ~37% of rows are not drawn for a given tree. Evaluating each tree on its OOB rows (and aggregating across trees for each row) gives an unbiased validation estimate without a separate set."),
        ("What extra randomness does a random forest add to bagging and why?", "Each split considers only a random subset of the features. This de-correlates the trees (they cannot all use the same dominant feature), so averaging reduces variance more."),
        ("Contrast bagging and boosting.", "Bagging trains models in parallel on bootstrap samples and averages them (variance reduction). Boosting trains sequentially, each model concentrating on the errors of its predecessors (bias reduction); it can overfit and cannot be parallelised across rounds."),
        ("What does the learning rate do in gradient boosting?", "It scales each tree's contribution. Lower η = smaller steps = more trees needed but better generalisation (shrinkage). Choose η and n_estimators jointly, using early stopping."),
        ("Explain gradient boosting for regression in three sentences.", "Start with a constant prediction (the mean). Fit a small tree to the residuals (negative gradient of the squared error), add η times its predictions to the ensemble. Repeat; each tree corrects what the ensemble still gets wrong."),
        ("Hard versus soft voting?", "Hard: each classifier votes for a class, the majority wins. Soft: average predicted probabilities and take the argmax: usually better because it uses confidence, but needs well-calibrated probabilities."),
        ("What is stacking and what is the leakage risk?", "Train a meta-model on base models' predictions. The meta-model must be trained on **out-of-fold** predictions, otherwise it sees predictions on data the base models were trained on and learns to over-trust them."),
        ("Random forest versus Extra-Trees?", "Extra-Trees choose split thresholds at random instead of searching the best one: faster, a bit more bias and less variance; sometimes better, sometimes worse; cheap to try."),
        ("Why choose HistGradientBoosting as the default for this dataset?", "Best CV error in the bake-off, very fast (≈2 s), handles missing values natively, scales to large datasets, and has built-in early stopping. Stacking's gain was within noise."),
        ("Can a random forest overfit by adding more trees?", "No; more trees only reduce the variance of the average (error plateaus). Overfitting is controlled by tree depth/leaf size, not the number of trees (unlike boosting).")])
    f.append(PageBreak())
    return f
