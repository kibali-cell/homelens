from pdflib import *


def build():
    f = [H1("Appendix A · Master cheat sheets")]
    f.append(H2("A.1  Algorithm comparison (Part I)"))
    f += table([
        ["Algorithm", "Task", "Needs scaling", "Strengths", "Weaknesses", "Key hyper-parameters"],
        ["Linear / Ridge / Lasso / Elastic net", "Regr.", "Yes (esp. regularised)", "Fast, interpretable, convex, good baseline", "Linear only; sensitive to outliers/collinearity", "α, l1_ratio"],
        ["Logistic / softmax regression", "Class.", "Yes", "Probabilities, interpretable coefficients, convex", "Linear boundary", "C (=1/α), penalty"],
        ["SGD models", "Both", "Yes", "Huge/streaming data, online", "Needs η tuning, noisy", "loss, α, η schedule, epochs"],
        ["k-NN", "Both", "Yes", "No training, flexible", "Slow prediction, curse of dimensionality", "k, metric, weights"],
        ["SVM (linear / RBF)", "Both", "**Yes**", "Strong on small/medium data; kernel trick", "Poor scaling O(m²–m³); no probabilities", "C, γ, kernel, ε (SVR)"],
        ["Decision tree", "Both", "No", "Readable, mixed features, no preprocessing", "Overfits, unstable, no extrapolation", "max_depth, min_samples_leaf"],
        ["Random forest / Extra-Trees", "Both", "No", "Robust, strong default, OOB score, importances", "Bigger and slower; no extrapolation", "n_estimators, max_features, min_samples_leaf"],
        ["Gradient boosting / HistGB", "Both", "No", "Typically best on tabular data; HistGB handles NaN, fast", "More tuning; sequential; can overfit", "learning_rate, n_estimators (early stop), max_depth/leaf_nodes, l2"],
        ["Voting / stacking", "Both", "Per member", "Squeezes extra accuracy", "Complexity, small gains", "members, meta-learner"],
        ["PCA / Kernel PCA / t-SNE", "Reduc.", "Yes", "Compression, denoising, visualisation", "Variance ≠ relevance; t-SNE not for transform", "n_components, γ, perplexity"],
        ["k-means / Mini-batch", "Cluster", "Yes", "Fast, simple, `transform` gives features", "Needs k, spherical clusters", "k, n_init, init"],
        ["DBSCAN", "Cluster", "Yes", "Any shape, noise detection, no k", "Varying densities; no predict", "eps, min_samples"],
        ["Gaussian mixture", "Cluster / density", "Yes", "Soft clusters, density, anomalies", "Local optima, assumes Gaussians", "k, covariance_type (BIC)"]],
        widths=[1.5, 0.6, 0.8, 1.9, 1.9, 1.7], font=6.9)
    f.append(H2("A.2  Metrics cheat sheet"))
    f += table([
        ["Problem", "Metric", "Formula / meaning", "Use when"],
        ["Regression", "RMSE", "√mean((ŷ−y)²)", "Large errors costly; same units as target"],
        ["Regression", "MAE", "mean|ŷ−y|", "Outliers present; easy to explain"],
        ["Regression", "R²", "1 − SSE/SST", "Share of variance explained (relative to the mean)"],
        ["Classification", "Accuracy", "(TP+TN)/all", "Balanced classes only"],
        ["Classification", "Precision / Recall / F<sub>1</sub>", "TP/(TP+FP) · TP/(TP+FN) · harmonic mean", "Imbalance; costs of FP vs FN differ"],
        ["Classification", "ROC-AUC / PR-AUC (AP)", "Ranking quality over all thresholds", "ROC: balanced; PR: rare positives / false-positive focus"],
        ["Classification", "Log loss / cross-entropy", "−mean[y log p + (1−y) log(1−p)]", "Probability quality; training loss of logistic/softmax"],
        ["Clustering", "Inertia, silhouette, BIC/AIC, ARI", "See Chapter 9", "No labels: internal; labels available: ARI"]], widths=[1.1, 1.6, 2.7, 2.4], font=7.6)
    f.append(H2("A.3  Symptom → diagnosis → remedy"))
    f += table([
        ["Symptom", "Likely cause", "What to try"],
        ["Train error ≪ validation error", "Overfitting (variance)", "More data; regularise (α↑, C↓, γ↓, depth↓, min_samples_leaf↑); early stopping; ensemble; simpler features"],
        ["Train ≈ validation, both high", "Underfitting (bias)", "More expressive model; engineer features; α↓; train longer; boosting"],
        ["CV score great, test/production poor", "Leakage, sampling bias, drift, tuned-to-validation", "Pipelines; check duplicates/time order; fresh holdout; check representativeness"],
        ["Loss diverges / NaN", "Learning rate too high, unscaled/outlier features", "Lower η; scale; clip outliers; gradient clipping"],
        ["Loss decreases very slowly", "η too small or poor conditioning", "Increase η; scale features; momentum/Adam (Part II)"],
        ["High accuracy but useless model", "Class imbalance", "Precision/recall/PR-AUC; resample or weight; threshold tuning"],
        ["SVM predicts one class only", "Unscaled features or bad C/γ", "StandardScaler; grid-search C, γ on log scale"],
        ["Tree/forest great in CV, bad on new regions", "No extrapolation / distribution shift", "Check feature ranges; add linear component; widen training data"],
        ["k-means results change between runs", "Initialisation", "k-means++, n_init ≥ 10, fixed seed"],
        ["Training too slow", "Algorithm complexity", "Subsample; LinearSVC/SGD/HistGB; mini-batch; randomised search; fewer features (PCA)"]], widths=[2.1, 2.1, 3.6], font=7.6)
    f.append(H2("A.4  Formula sheet (know these cold)"))
    f += table([
        ["Name", "Formula"],
        ["Linear model / MSE", "ŷ = θ·x ; MSE(θ) = (1/m) Σ (θ·x<super>(i)</super> − y<super>(i)</super>)<super>2</super>"],
        ["Normal equation", "θ̂ = (XᵀX)<super>−1</super> Xᵀ y  (use the pseudo-inverse X<super>+</super> y in practice)"],
        ["Gradient (MSE) & update", "∇ = (2/m) Xᵀ(Xθ − y) ; θ ← θ − η ∇"],
        ["Ridge / Lasso cost", "MSE + α·½Σθ<sub>j</sub><super>2</super> ; MSE + α·Σ|θ<sub>j</sub>|"],
        ["Sigmoid / logistic loss / gradient", "σ(t) = 1/(1+e<super>−t</super>) ; J = −(1/m)Σ[y log p̂ + (1−y)log(1−p̂)] ; ∇ = (1/m) Xᵀ(σ(Xθ) − y)"],
        ["Softmax / cross-entropy", "p̂<sub>k</sub> = e<super>s<sub>k</sub></super>/Σ<sub>j</sub> e<super>s<sub>j</sub></super> ; J = −(1/m) Σ<sub>i</sub> Σ<sub>k</sub> y<sub>k</sub><super>(i)</super> log p̂<sub>k</sub><super>(i)</super>"],
        ["SVM soft margin / margin width", "min ½‖w‖² + CΣξ<sub>i</sub> ; width = 2/‖w‖ ; RBF K(a,b) = exp(−γ‖a−b‖²)"],
        ["Gini / CART cost", "G = 1 − Σp<sub>k</sub><super>2</super> ; J = (m<sub>l</sub>/m)G<sub>l</sub> + (m<sub>r</sub>/m)G<sub>r</sub>"],
        ["Precision, recall, F<sub>1</sub>", "TP/(TP+FP) ; TP/(TP+FN) ; 2PR/(P+R)"],
        ["PCA", "X = UΣVᵀ ; X<sub>d</sub> = X V<sub>d</sub> ; explained variance ratio<sub>j</sub> = σ<sub>j</sub><super>2</super>/Σσ<super>2</super>"],
        ["K-means inertia / silhouette", "Σ<sub>i</sub> ‖x<sub>i</sub> − μ<sub>c(i)</sub>‖² ; s = (b − a)/max(a, b)"],
        ["BIC / AIC", "p ln m − 2 ln L̂ ; 2p − 2 ln L̂"],
        ["Bias–variance", "E[(y − ŷ)²] = Bias² + Variance + irreducible noise"]], widths=[2.3, 5.5], font=7.8)
    f.append(PageBreak())

    f.append(H1("Appendix B · The mixed challenge round"))
    f.append(P("These cross-chapter scenarios are deliberately open-ended: they train *decisions*, not definitions. Cover the answer, respond aloud, then compare."))
    f += QA([
        ("A bank's fraud rate is 0.2%. A model reports 99.8% accuracy. Your verdict, and what do you measure instead?", "Meaningless: predicting 'no fraud' gets that. Use precision/recall (or PR-AUC), choose a threshold from the cost of a missed fraud versus a false block, and validate with stratified folds on out-of-time data."),
        ("You have 50 features, 300 rows, and your linear model overfits badly. List three actions in order of preference.", "(1) Regularise (Ridge/Lasso/Elastic net) with α by CV; (2) reduce features (Lasso selection, domain knowledge, PCA inside the pipeline); (3) get more data or use a high-bias model. Avoid flexible models (deep trees, RBF with large γ)."),
        ("You have 20 million rows and 50 numeric features. Which of linear SVM, kernel SVM, random forest, HistGB, k-NN would you test first, and why?", "HistGradientBoosting (fast, scales, strong on tabular), plus an SGD/linear baseline. Kernel SVM (O(m²+)) and k-NN (slow prediction, memory) are impractical; a random forest is feasible but larger and slower than HistGB."),
        ("Validation accuracy is 95% but after deployment it is 78%. Give five hypotheses and one test for each.", "Leakage (re-audit the pipeline/features); sampling bias or data mismatch (compare feature distributions train vs production); concept drift (compare performance over time); duplicates between train and validation (hash rows); training/serving skew (run the production pipeline on a validation sample and compare predictions)."),
        ("Why might standardising the target help a gradient-descent model but not a random forest?", "GD's step size and loss scale depend on the target's magnitude; standardising keeps gradients in a sane range. Trees split on feature thresholds and predict leaf means; they are invariant to monotone rescaling of the target."),
        ("Your learning curves show validation error still falling at the largest training size. What does that suggest?", "The model is data-limited (variance); more data will likely help. If the training error is also high, add capacity too."),
        ("How would you detect and fix overfitting caused by hyper-parameter tuning itself?", "A large gap between CV score and fresh test score. Use nested CV or keep a final test set untouched, limit the search budget, and prefer simpler hyper-parameter regions."),
        ("When would you prefer Lasso to Ridge and when Elastic net to both?", "Lasso when you believe few features matter and want sparsity; Ridge when many small effects/correlated features; Elastic net when features are correlated *and* you want some selection, or n < m."),
        ("Why is the k in k-means a different kind of decision from the k in k-fold CV?", "k-means' k is a *modelling* choice (number of clusters) with no ground truth; k-fold's k is an *evaluation* choice (bias/variance/cost of the estimate; typically 5 or 10)."),
        ("Explain bagging, boosting and stacking to a manager in one sentence each.", "Bagging: ask many independent experts trained on different samples and average. Boosting: train experts in sequence, each focusing on the previous ones' mistakes. Stacking: hire a referee who learns which expert to trust when."),
        ("A decision tree on a trend dataset predicts a flat line beyond the last observed point. Why, and what do you do?", "Leaves predict constants (means), so it cannot extrapolate. Use a linear/GAM component, model the detrended residual, or a model that extrapolates."),
        ("Why can the same feature be useless to a linear model and useful to a forest?", "Linear models need a monotone, additive relationship; trees can use thresholds and interactions (e.g. latitude only matters in combination with income and proximity)."),
        ("Name three ways your pipeline could leak and how you would test for each.", "Preprocessing fitted on all data (move into Pipeline; compare scores), target-derived features (audit feature sources; check importances for suspiciously strong features), duplicates/related rows across splits (group-aware splitting, hash check). A too-good-to-be-true validation score should always trigger an audit."),
        ("How do you choose between two models whose CV scores differ by less than one standard deviation?", "Treat them as equivalent on accuracy; choose on secondary criteria: simplicity, speed, interpretability, robustness, maintenance cost; or run a paired comparison across the same folds."),
        ("Why use `StratifiedShuffleSplit`/`stratify=` and what is `GroupKFold` for?", "Stratified: preserve class (or binned-target) proportions in each split. GroupKFold: keep all rows of a group (customer, patient, region) in the same fold to avoid leakage between related samples."),
        ("Derive why the gradient of the log loss simplifies to Xᵀ(σ(Xθ) − y)/m.", "With p = σ(z), dp/dz = p(1 − p). The loss derivative for one instance w.r.t. z is −y/p·p(1−p) + (1−y)/(1−p)·p(1−p) = −y(1−p) + (1−y)p = p − y. By the chain rule dz/dθ = x. Averaging gives (1/m)Xᵀ(p − y)."),
        ("What happens to ridge regression's solution as α → 0 and α → ∞?", "α → 0: the ordinary least-squares solution. α → ∞: all weights (except the bias) shrink to 0, the model predicts the mean (maximum bias)."),
        ("What is the effect of increasing `max_features` in a random forest?", "Trees become more similar and individually stronger: lower bias per tree but higher correlation between trees (less variance reduction). Tune it with CV; smaller values add diversity."),
        ("You must explain a boosted model's prediction to a regulator. What can you offer?", "Feature importances (permutation importance is more reliable than impurity-based), partial dependence plots, per-prediction attributions (SHAP), a simpler surrogate model (a shallow tree or linear model), and the model card with its limitations."),
        ("Why does Part II still care about everything in this guide?", "Neural networks are trained with gradient descent on a loss (softmax + cross-entropy for classification, MSE for regression), need scaling, overfit and need regularisation/early stopping, are validated with the same splits, and sit inside the same pipelines. New: automatic differentiation, layers, initialisation, optimisers, and GPUs.")])
    f.append(PageBreak())

    f.append(H1("Appendix C · Bridge to Part II (Neural Networks and Deep Learning)"))
    f.append(P("You are about to learn Keras and TensorFlow. The following table maps each Part II idea to the Part I idea you now understand. If the right-hand column feels solid, Part II will feel like a natural extension."))
    f += table([
        ["Part II idea", "Part I foundation (where you built it)"],
        ["A neuron computes σ(w·x + b)", "Logistic regression (Chapter 4: you coded it)"],
        ["Multi-layer perceptron, softmax output layer", "Softmax regression + cross-entropy (Chapter 4); stacking linear layers with non-linearities gives non-linear boundaries (kernel/polynomial ideas, Chapter 5)"],
        ["Training with `model.fit`: epochs and batches", "Mini-batch gradient descent, learning rate, epochs (Chapter 4, Figures 11–12)"],
        ["Backpropagation", "The gradient (1/m)Xᵀ(p − y), generalised layer by layer by the chain rule; computed automatically (autodiff)"],
        ["Optimisers: momentum, RMSProp, Adam", "Plain GD and the learning-rate lessons; you implemented plain GD and (in experiments) momentum"],
        ["Learning-rate schedules and finders", "SGD learning schedule (Chapter 4)"],
        ["Regularisation: ℓ<sub>1</sub>/ℓ<sub>2</sub>, dropout, early stopping", "Ridge/Lasso, early stopping (Chapter 4), bagging-like averaging (Chapter 7; dropout ≈ ensemble of subnetworks)"],
        ["Vanishing/exploding gradients, normalisation", "Feature scaling (Chapters 2, 4); unstable GD (Figure 11)"],
        ["Train/validation/test, callbacks, hyper-parameter tuning", "Chapters 1–2: validation discipline, randomised search"],
        ["Preprocessing layers, `tf.data`", "Pipelines and ColumnTransformer (Chapter 2)"],
        ["Autoencoders, GAN-like generative ideas", "PCA/reconstruction error (Chapter 8), Gaussian mixtures and density estimation (Chapter 9)"],
        ["Over-parameterised models that still generalise", "Bias–variance (Chapters 1, 4), regularisation, ensembling"]], widths=[2.8, 5], font=7.8)
    f.append(H2("Suggested next step: a bridge experiment"))
    f.append(P("Before opening Part II, run this 30-minute experiment to anchor the transition: using `sklearn.neural_network.MLPRegressor` (a small multi-layer perceptron) on the HomeLens preprocessed features, compare it against the HistGradientBoosting baseline under the same CV folds. "
               "Plot the validation curve against `hidden_layer_sizes` and `alpha`. You should *predict*, before running it, that the network will need scaled inputs, will be more sensitive to the learning rate than the boosted trees, and will probably not beat boosting on 16k tabular rows."))
    f.append(H2("Mastery checklist"))
    f += BL(["☐ I can explain train/validation/test sets, CV and data leakage without notes.", "☐ I can build a Pipeline + ColumnTransformer with a custom transformer from memory.", "☐ I can write the gradient-descent update and implement linear and logistic regression with NumPy.",
             "☐ I can read learning curves and say whether to get more data or a better model.", "☐ I can select a metric and threshold from a cost description, and defend precision/recall/AUC choices.", "☐ I can explain C and γ in SVMs and why scaling is mandatory.",
             "☐ I can compute Gini and a CART split by hand, and explain tree instability.", "☐ I can explain bagging, boosting and stacking and say which reduces bias or variance.", "☐ I can compute PCA via SVD and explain when reduction hurts.",
             "☐ I can choose k for k-means, explain DBSCAN and GMM/BIC, and use clustering for features and anomaly detection.", "☐ I can re-run this entire project, change one design choice, and predict the effect.", "☐ I scored well on the challenge questions (aim: at least 80% answered fluently)."])
    f.append(H2("A two-week plan"))
    f += table([["Days", "Focus", "Output"],
                ["1–2", "Chapters 1–2 + Steps 1–3; read the code line by line; do experiments", "Re-write `features.py` from memory"],
                ["3–4", "Chapters 3–4 + Steps 4–5", "Implement logistic regression + GD variants without looking"],
                ["5–6", "Chapters 5–7 + Steps 6–8", "Reproduce the bake-off with a different dataset (e.g. the diabetes dataset)"],
                ["7–8", "Chapters 8–9 + Steps 9–10", "Explain PCA and GMM to a friend; run the extensions"],
                ["9–10", "Step 11 + Chapter 10", "Write your own model card for a variant model"],
                ["11–12", "Challenge questions + mixed round", "Answer aloud, mark weak areas, revisit those chapters"],
                ["13–14", "Capstone (below) + Part II bridge experiment", "A new project on a new dataset using the same structure"]], widths=[0.7, 4, 3])
    f.append(H2("Capstone extensions"))
    f += BL(["**Out-of-time validation**: simulate drift by training on districts from some regions and testing on others; quantify the degradation.",
             "**Quantile regression**: use `HistGradientBoostingRegressor(loss='quantile', quantile=0.1/0.9)` to give prediction intervals per district; check their coverage on the test set.",
             "**Calibration**: plot a reliability curve of the premium classifier and apply `CalibratedClassifierCV`.",
             "**Permutation importance & partial dependence** (`sklearn.inspection`) to explain the final model.",
             "**Monitoring simulation**: add noise/shift to features and compute the population stability index (PSI) per feature to practice drift detection.",
             "**New dataset**: apply the entire workflow (framing → EDA → pipeline → baselines → models → tuning → test → card) to another tabular dataset."])
    f.append(PageBreak())

    f.append(H1("Appendix D · Glossary"))
    terms = [("Baseline", "Simple reference model used to judge whether a complex model adds value."), ("Bias (statistical)", "Error from wrong/simplistic assumptions; causes underfitting."), ("Bootstrap", "Sampling with replacement to create a resampled training set."),
             ("Cross-validation", "Rotating validation folds to estimate generalisation error from the training data."), ("Data leakage", "Information from validation/test (or the future/the target) reaching training."), ("Epoch", "One pass through the full training set."),
             ("Feature engineering", "Creating/transforming features to expose structure to the model."), ("Generalisation error", "Expected error on new, unseen data."), ("Gradient descent", "Iterative optimisation moving parameters against the cost gradient."),
             ("Hyper-parameter", "A setting of the learning algorithm, set before training."), ("Hinge loss", "max(0, 1 − t·s), used by SVMs."), ("Inertia", "Sum of squared distances of points to their cluster centroid."),
             ("Kernel trick", "Computing inner products in a high-dimensional feature space without constructing it."), ("Learning curve", "Training and validation error as a function of training-set size."), ("Log loss", "Cross-entropy for binary classification."),
             ("Margin", "Width of the street between classes in an SVM."), ("OOB (out-of-bag)", "Training instances left out of a bootstrap sample, used for validation."), ("Overfitting", "Model fits training noise: low training error, high validation error."),
             ("Pipeline", "A chain of transformations and an estimator fitted as one object."), ("Precision / Recall", "Share of positive predictions that are right / share of actual positives found."), ("Regularisation", "Constraining a model to reduce overfitting."),
             ("Residual", "Actual minus predicted value."), ("Stratified sampling", "Sampling that preserves group proportions."), ("Support vector", "Training instance on/inside the SVM margin that defines the boundary."),
             ("Underfitting", "Model too simple to capture the pattern: high error on both train and validation."), ("Variance (statistical)", "Sensitivity of the model to the specific training sample; causes overfitting.")]
    f += table([["Term", "Meaning"]] + [[f"**{a}**", b] for a, b in terms], widths=[1.8, 6], font=7.8)
    f.append(H2("Closing note"))
    f.append(P("If you have *run* the code, predicted the outputs, broken things on purpose and answered the questions aloud, you do not merely recognise Part I: you can use it, debug it and defend it. That is exactly the preparation Part II assumes. "
               "Keep the project as a template: **frame → split → explore → pipeline → baseline → compare → tune → evaluate once → explain → ship → monitor.** Every project in Part II, and beyond, follows the same arc."))
    return f
