from pdflib import *

STEP = "step01_framing"


def build():
    f = [H1("Chapter 1 · The Machine Learning Landscape"),
         P("**Project step 1** — frame the HomeLens problem, choose a metric, build baselines, and *watch* underfitting and overfitting happen. Code: `steps/step01_framing.py`.")]

    f.append(H2("1.1  Concepts you must own"))
    f.append(H3("What machine learning is"))
    f.append(P("Machine learning is the practice of getting a computer to improve at a task *from data* instead of from hand-written rules. Tom Mitchell's definition is the one to memorise: "
               "**a program learns from experience E with respect to task T and performance measure P if its performance on T, as measured by P, improves with experience E.** "
               "For HomeLens: T = estimate a district's typical house value; E = the 16,512 training districts with known values; P = RMSE in dollars. "
               "If you cannot name T, E and P for a project, you do not yet have a machine-learning problem."))
    f.append(P("ML shines where hand-coded rules are too long, too brittle or unknown (spam filtering, image recognition), where the environment changes (retrain instead of rewrite), "
               "and where it can reveal structure in data humans have not noticed (*data mining*). It is the wrong tool when a few exact rules solve the problem, or when you have no relevant data."))

    f.append(H3("A taxonomy of ML systems"))
    f.append(P("Systems are classified along three independent axes. Being able to place any project on all three is the first step of framing it."))
    f += table([
        ["Axis", "Options", "HomeLens example"],
        ["**Supervision** (what the training data contains)", "**Supervised** (labels given: regression, classification) · **Unsupervised** (no labels: clustering, anomaly detection, dimensionality reduction, association rules) · **Semi-supervised** (few labels, many unlabelled) · **Self-supervised** (labels generated from the data itself) · **Reinforcement** (an agent acts, receives rewards)", "Valuation and screening = supervised; segmentation = unsupervised; Chapter 9 shows a semi-supervised experiment"],
        ["**Learning mode** (does it learn incrementally?)", "**Batch / offline** (train on all data, then deploy; retrain periodically) · **Online** (learn from a stream in small steps; tune the *learning rate*; also used for data too big for memory = out-of-core)", "Batch: the market changes slowly, so retrain on a schedule"],
        ["**Generalisation strategy**", "**Instance-based** (memorise examples, compare new ones with a similarity measure, e.g. k-NN) · **Model-based** (fit a parametric model and predict with it)", "Model-based (linear models, trees, boosting)"]],
        widths=[1.6, 4, 2.2])
    f.append(P("**Regression** predicts a number; **classification** predicts a class. Many algorithms have both versions (trees, SVMs, forests), and some have a misleading name (*logistic regression* is a classifier)."))

    f.append(H3("The main challenges: \"bad data\" and \"bad algorithm\""))
    f += table([
        ["Challenge", "Symptom", "Remedy"],
        ["Insufficient training data", "Learning curves still falling; high variance", "Collect more; use simpler/regularised models; data augmentation. Banko & Brill showed that with enough data very different algorithms converge, but data is not always cheap"],
        ["Non-representative data / **sampling bias**", "Good validation score, bad production score", "Make the sample resemble the population; use stratified sampling; check who is missing (the classic example is a poll that only reaches people with landlines)"],
        ["Poor-quality data", "Outliers, noise, missing values, wrong units", "Clean, impute, clip or discard; know *why* values are missing (HomeLens: 207 missing bedroom counts)"],
        ["Irrelevant features", "Noise columns reduce accuracy", "**Feature engineering**: selection, extraction (e.g. rooms per household), creating new features"],
        ["**Overfitting**", "Tiny training error, large validation error", "Simplify the model, constrain it (**regularisation**), get more data, reduce noise"],
        ["**Underfitting**", "Large error on both sets", "Use a more powerful model, better features, less regularisation"]],
        widths=[1.7, 2.1, 4], font=7.8)
    f += callout("key", "Overfitting in one sentence", [
        "A model overfits when it learns the *noise and quirks of the training sample* instead of the signal. Complex models (high degree polynomials, deep unrestricted trees, huge networks) have enough freedom to memorise. "
        "**Regularisation** is any constraint that reduces that freedom; **hyper-parameters** (like regularisation strength) are settings of the *learning algorithm*, chosen before training and never learned by it."])

    f.append(H3("Testing and validating"))
    f.append(P("The only way to know how a model will behave on new cases is to try it on new cases. Split the data: the **training set** fits the parameters; the **test set** estimates the *generalisation error* once, at the end. "
               "If you pick hyper-parameters by repeatedly measuring on the test set, you will have tuned to it, and your estimate will be optimistic. So introduce a third set:"))
    f += table([
        ["Set", "Used to", "How often"],
        ["Training", "Fit the model's parameters", "Constantly"],
        ["Validation (dev) set, or cross-validation folds", "Compare models, choose hyper-parameters", "Many times"],
        ["Test", "Report the final generalisation error", "**Once**, at the very end"]], widths=[2.3, 3, 1.5])
    f.append(P("With a small dataset a single validation set is noisy, so use **k-fold cross-validation**: split the training data in *k* folds, train on *k − 1*, validate on the remaining fold, rotate, and average. "
               "It costs *k* times more compute but uses the data efficiently. After choosing the best settings, retrain on the full training set."))
    f.append(P("**Data mismatch.** If your training data is plentiful but unlike production data (e.g. web photos vs phone-app photos), a validation set drawn from the training source misleads you. "
               "Make the validation/test sets from *production-like* data, and carve out a **train-dev set** from the training source: if the model does well on train-dev but badly on validation, the problem is *data mismatch*, not overfitting."))
    f.append(P("**No Free Lunch theorem.** No model is universally best. Wolpert showed that averaged over all possible problems every algorithm performs the same. In practice you make *reasonable assumptions* about your data (linear? smooth? local?) and test a few suitable models. This is why Part I teaches many algorithms and a rigorous way to compare them."))

    f.append(H2("1.2  Framing HomeLens: the checklist every project starts with"))
    f += table([
        ["Question", "HomeLens answer"],
        ["What is the business objective? How will the model be used?", "Help analysts rank/compare districts; help brokers screen for premium areas. It is a decision-support tool, **not** an appraisal of individual houses."],
        ["What does the current solution look like?", "Manual spreadsheet estimates (our human baseline; typically off by large margins in unfamiliar areas)."],
        ["Supervised / unsupervised / RL?", "Supervised (labels: median_house_value). Plus unsupervised extras."],
        ["Regression or classification?", "Regression (value); classification (premium or not); clustering (segments)."],
        ["Batch or online?", "Batch; plain retraining suffices. Data is small enough for memory."],
        ["Performance measure?", "RMSE (penalises big errors, matches the typical regression objective) and MAE (robust, easier to explain). Both in dollars."],
        ["Assumptions to verify?", "Is the downstream user expecting *prices* or *categories*? If categories only (cheap/mid/expensive), classification would have been enough. We checked: both are needed."]],
        widths=[2.5, 5.3])
    f.append(P("**Choosing RMSE vs MAE.** RMSE is the Euclidean (ℓ<sub>2</sub>) norm of the error vector: RMSE = √( (1/m) Σ (ŷ<super>(i)</super> − y<super>(i)</super>)<super>2</super> ). "
               "Squaring makes it sensitive to large errors, which is good when big misses are costly and the error distribution is roughly bell-shaped. MAE is the ℓ<sub>1</sub> norm: (1/m) Σ |ŷ<super>(i)</super> − y<super>(i)</super>|. "
               "It is more robust to outliers. A larger norm index focuses on large values: RMSE ≥ MAE always, and they are close only if errors are similar in size."))

    f.append(H2("1.3  Project step 1 — hands on"))
    f.append(P("We load the data, split it, and make two baselines. A **baseline** is the simplest defensible model; it sets the bar. If a sophisticated model does not clearly beat 'predict the average', something is wrong."))
    f += code(STEP, "setup")
    f += output(STEP, "setup")
    f += code(STEP, "baseline")
    f += output(STEP, "baseline")
    f.append(P(f"Predicting the training mean everywhere yields an RMSE of about **${R['baseline_mean_rmse']:,.0f}**. This number is simply the standard deviation of the target: a constant predictor's error is the spread of the data. "
               f"One feature, `median_income`, in a straight line already cuts it to **${R['baseline_income_rmse']:,.0f}**. Every model from here on must be compared with these two numbers."))
    f.append(H3("Seeing under- and overfitting"))
    f.append(P("To see the bias/variance trade-off clearly we deliberately make learning hard: only **40** training districts, one feature (income), and polynomials of increasing degree. "
               "A degree-*d* polynomial has *d* + 1 parameters, so higher degree means more flexibility. We measure error on the 40 points it saw (training) and on 2,000 districts it did not see (unseen)."))
    f += code(STEP, "bias_variance", stop_at="fig, ax")
    f += output(STEP, "bias_variance")
    f += fig("fig01_bias_variance", "**Figure 1.** Left: a straight line underfits (high bias). Middle: a degree-16 polynomial threads the 40 training points but swings wildly in between (high variance). Right: training error keeps falling with complexity while error on unseen data explodes: the signature of overfitting.")
    f.append(P("Read the table in three steps. (1) Training RMSE falls monotonically with degree: more flexibility always fits the training points better. "
               "(2) Unseen RMSE is best at the simplest degrees and then *explodes* by orders of magnitude. (3) The gap between the two columns *is* the overfitting. "
               "Notice that the best model here is the simple one: with noisy data and few points, the extra flexibility only buys noise."))
    f.append(H3("More data is the best regulariser"))
    f += code(STEP, "data_size")
    f += output(STEP, "data_size")
    f.append(P("Same degree-12 model, growing training sets. With 20 points the model is useless; with 300 the training and unseen errors are almost the same; with 5,000 they have essentially converged. "
               "A model that overfits on a small sample can be perfectly fine on a large one. This is why the first question when you see overfitting is *can I get more data?*, and why complexity should always be judged relative to the amount of data."))
    f += callout("warn", "Common mistakes this step guards against", [
        "• **Judging a model by its training error.** It can always be made small. Only error on unseen data counts.  • **No baseline.** An RMSE of $60,000 means nothing until you know the mean-baseline is $115,000.  "
        "• **Letting validation data leak into training.** Here the validation districts are removed from the training sample before fitting."])

    f.append(H2("1.4  Hands-on experiments (do these)"))
    f += NL(["Change `PolynomialFeatures` degree values and the training sample size (40 → 400). At what sample size does the degree-8 model stop overfitting? What is the rule of thumb you infer?",
             "Replace `MinMaxScaler` with nothing. What goes wrong numerically and why? (Hint: x<super>16</super>.)",
             "Add `Ridge(alpha=1)` instead of `LinearRegression` to the degree-16 model. Does the unseen error recover? (You will study why in Chapter 4.)",
             "Compute the MAE of both baselines. Which is larger, RMSE or MAE, and what does the ratio tell you about the error distribution?"])

    f.append(H2("1.5  Challenge questions"))
    f += QA([
        ("What is the difference between a parameter and a hyper-parameter?", "A **parameter** is learned from data by the training algorithm (a regression coefficient, a tree's split thresholds). A **hyper-parameter** is a setting of the algorithm chosen *before* training (learning rate, tree depth, regularisation strength, k in k-means) and tuned on validation data."),
        ("Your model has 99% training accuracy and 70% validation accuracy. Diagnose it and list four remedies.", "**Overfitting** (high variance). Remedies: get more training data; simplify the model (fewer features/parameters, lower tree depth); add or increase regularisation; use early stopping, dropout or ensembling; reduce noise (remove outliers, fix labels)."),
        ("Both training and validation errors are high and similar. What now?", "**Underfitting** (high bias). Use a more expressive model, engineer better features, reduce regularisation, train longer. Collecting more data will *not* help much if the model cannot represent the pattern."),
        ("Why a separate validation set and a test set?", "Choosing hyper-parameters on the test set makes it part of the training process, so its error is an optimistic estimate. The validation set absorbs the repeated comparisons; the test set stays pristine for one final, unbiased estimate."),
        ("What is sampling bias, and give an example.", "The sample is systematically unlike the population. Example: pooling customer reviews only from users who bothered to review gives a sentiment model that fails on silent users. Fix by representative or stratified sampling and by checking who is missing."),
        ("Why can a dumb baseline matter more than a clever model?", "It gives the model a meaning. A 91% accuracy on data where 90% of cases belong to one class is a failure. Baselines (mean predictor, majority class, last-period value, a simple linear model) expose such illusions and quantify the value the complex model adds."),
        ("Name the three kinds of 'learning mode/generalisation' axes and place k-NN on them.", "Supervision × batch/online × instance/model-based. k-NN is supervised, usually batch, and **instance-based** (it stores training points and compares new points to them)."),
        ("When would you prefer MAE over RMSE?", "When the data has many outliers or when errors should be penalised linearly (e.g. delivery-time errors where an hour late is exactly twice as bad as thirty minutes). RMSE is preferred when large errors are disproportionately costly and outliers are rare."),
        ("What is a train-dev set and what problem does it diagnose?", "A held-out slice of the *training distribution*. If the model does well on train-dev but badly on the (production-like) validation set, the culprit is **data mismatch**, not overfitting."),
        ("State the No Free Lunch theorem and its practical consequence.", "Averaged over all possible data-generating processes, no algorithm beats any other. Practically, there is no universally best model: make assumptions about the data and compare a few suitable candidates with proper validation.")])
    f.append(PageBreak())
    return f
