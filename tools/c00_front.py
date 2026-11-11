from pdflib import *
from reportlab.platypus import NextPageTemplate


def cover():
    f = []
    f.append(SP(70))
    f.append(Paragraph('<font name="Sans-B" size="11" color="#d9822b">PROJECT-BASED STUDY GUIDE</font>', S["body"]))
    f.append(SP(6))
    f.append(Paragraph('<font name="Sans-B" size="34" color="#1f4e79">The Fundamentals<br/>of Machine Learning</font>', ParagraphStyle("t", leading=40)))
    f.append(SP(10))
    f.append(Paragraph('<font name="Sans" size="15" color="#374151">Part I of <i>Hands-On Machine Learning with Scikit-Learn, Keras &amp; TensorFlow</i>,<br/>learned by building one realistic, end-to-end project: <b>HomeLens</b></font>', ParagraphStyle("t2", leading=21)))
    f.append(SP(34))
    rows = [["Book chapters covered", "1 · The ML Landscape   2 · End-to-End Project   3 · Classification   4 · Training Models   5 · Support Vector Machines   6 · Decision Trees   7 · Ensembles &amp; Random Forests   8 · Dimensionality Reduction   9 · Unsupervised Learning"],
            ["Project", "HomeLens: valuation, screening and market segmentation of 20,640 California census districts"],
            ["What you get", "A complete, tested Python project (11 runnable steps), every output and figure in this guide produced by that code, plus a challenge bank of 100+ questions"],
            ["Tool versions used", "Python 3.13 · scikit-learn 1.9 · pandas 3.0 · NumPy 2.5 · Matplotlib 3.11"]]
    f += table([[f"**{a}**", b] for a, b in rows], widths=[1.3, 4], header=False, zebra=False, font=8.6)
    f.append(SP(30))
    f.append(Paragraph('<font size="8.5" color="#6b7280">This guide is an independent study companion. It explains the concepts of the book in its own words and uses original code. '
                       'Read it next to the book: the book gives you the author\'s full treatment; this guide makes you apply every idea until you can defend it.</font>', S["body"]))
    f.append(NextPageTemplate("main"))
    f.append(PageBreak())
    return f


def how_to_use():
    f = [H1("How to use this guide"), P(
        "Part I of the book is the foundation for everything that follows. Neural networks (Part II) are *one more family of models* trained with the same ideas you meet here: "
        "a loss function, gradient descent, regularisation, a validation set, a pipeline that prevents leakage. If those ideas are solid, deep learning is an extension; if they are shaky, deep learning feels like magic and fails for reasons you cannot diagnose."),
        P("This guide therefore does not walk through the chapters one by one in isolation. It makes you **build one project** in which every chapter solves a real problem that appears naturally, in the same order a professional would meet it. "
          "You will see each idea *fail or succeed on real data*, which is how understanding becomes durable.")]
    f.append(H2("The three-pass method"))
    f += table([
        ["Pass", "What you do", "Goal"],
        ["1 · Read and run", "Read each chapter's concept section, then read the code blocks and outputs. Run `./run_all.sh` or run each step with `python tools/run_step.py steps/stepXX_*.py`.", "Recognise every idea and see it work."],
        ["2 · Predict and break", "Before reading each OUTPUT block, cover it and predict the result. Then change one thing (a hyper-parameter, a feature, the split) and predict again. Use the *Hands-on experiments* at the end of each chapter.", "Build intuition for cause and effect."],
        ["3 · Defend", "Answer the *Challenge questions* aloud without notes. Then do the final *Defend your model* review in Chapter 10 as if a sceptical colleague were interrogating you.", "Be able to explain, derive and justify."]],
        widths=[1.1, 4, 2])
    f += callout("key", "The five abilities you are training", [
        "**Explain** — say what an algorithm does and why, in plain words. **Derive** — write the key equation and say what each symbol means. "
        "**Implement** — code the core of it from scratch at least once (Chapter 4 does this). **Debug** — recognise under/overfitting, leakage and bad metrics from symptoms. "
        "**Decide** — choose an algorithm, metric and validation scheme for a new problem and justify it."])
    f.append(H2("What \"ready for Part II\" means"))
    f += BL(["You can say, without hesitation, what the **training**, **validation** and **test** sets are for, and why touching the test set early invalidates it.",
             "You can write the gradient-descent update rule, explain the effect of the learning rate, and explain why features must be scaled.",
             "You can explain the **bias/variance trade-off** and name at least four levers that move a model along it.",
             "You can choose between precision, recall, F1, ROC-AUC and PR-AUC for a given business cost, and set a decision threshold.",
             "You can explain why **ensembles** (bagging, boosting, stacking) work, and why a single decision tree is unstable.",
             "You can describe **PCA** geometrically and algebraically (SVD), and say when dimensionality reduction helps and when it hurts.",
             "You can wrap everything in a **Pipeline** so cross-validation is honest. In Part II this becomes `tf.data` + Keras models, but the discipline is identical."])
    f.append(PageBreak())
    return f


def project_brief():
    f = [H1("The HomeLens project"),
         P("You have just joined **HomeLens**, a (fictional) real-estate analytics start-up. The sales team currently estimates neighbourhood values by hand from spreadsheets. "
           "They want a data product that can, for any census district in California:")]
    f += NL(["**Estimate the typical house value** (a regression model) so analysts can rank and compare districts.",
             "**Flag premium districts** (a classifier) so brokers can quickly screen where high-value listings are likely, tuned for a precision the business can trust.",
             "**Discover market segments and unusual districts** (unsupervised learning) so that marketing can target regions and data engineers can find suspicious records."])
    f.append(P("This is a deliberately realistic brief. It is **not** a toy: the data has missing values, hard caps, heavy tails, a categorical field, geographic structure and noisy labels. "
               "It is the California housing dataset used throughout Part I of the book (1990 census block groups, 20,640 rows)."))
    f.append(H2("Success criteria agreed with the stakeholders"))
    f += table([["Deliverable", "Business question", "Metric", "Acceptance bar"],
                ["Valuation model", "How much is a typical house worth here?", "RMSE and MAE in dollars; share within ±20%", "Must clearly beat the naive baseline (predict the mean); error estimate must be trustworthy"],
                ["Premium screener", "Is this district in the top price quartile?", "Precision (cost of a false lead) and recall; ROC-AUC", "Precision ≥ 0.85 with the best possible recall"],
                ["Segmentation / anomaly detector", "Which markets exist? Which rows look wrong?", "Silhouette, BIC, business sense", "Interpretable regions; flagged rows reviewable by a human"]],
                widths=[1.2, 1.7, 1.7, 2.2])
    f.append(H2("The data dictionary"))
    f += table([["Attribute", "Meaning", "Type / quirk"],
                ["longitude, latitude", "Location of the district", "Numeric; strong geographic structure"],
                ["housing_median_age", "Median age of houses (years)", "Capped at 52"],
                ["total_rooms, total_bedrooms", "Totals over the whole district", "Heavy right tail; 207 missing bedroom values"],
                ["population, households", "People and households in the district", "Heavy right tail"],
                ["median_income", "Median household income (units of $10,000)", "Strongest predictor; capped near 15"],
                ["ocean_proximity", "Category: <1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, ISLAND", "Categorical; ISLAND has only 5 rows"],
                ["median_house_value", "**Target**: median house value ($)", "Capped at $500,001: about 4.7% of rows"]], widths=[1.6, 2.6, 2.6])
    f.append(H2("Repository layout"))
    f += snippet("""
homelens/
├── data/housing.csv                  # raw data, never edited
├── src/homelens/                     # the reusable library (imported by every step)
│   ├── config.py                     # paths, RANDOM_STATE, constants
│   ├── data.py                       # load, stratified split, label construction
│   ├── features.py                   # custom transformer + preprocessing pipelines
│   ├── tasks.py                      # ready-made arrays for chapters 3-9
│   ├── utils.py                      # plotting style, RMSE, results registry
│   └── predict.py                    # inference on new districts
├── steps/                            # ONE script per learning step (runnable)
│   ├── step01_framing.py             #  Ch1  landscape, baselines, under/overfitting
│   ├── step02_eda.py                 #  Ch2  look at the data, split, explore
│   ├── step03_pipeline_models.py     #  Ch2  pipelines, CV, fine-tuning, error analysis
│   ├── step04_classification.py      #  Ch3  metrics, thresholds, multiclass
│   ├── step05_training_models.py     #  Ch4  GD from scratch, regularisation, softmax
│   ├── step06_svm.py                 #  Ch5  kernels, C and gamma, SVR
│   ├── step07_trees.py               #  Ch6  CART, Gini, instability
│   ├── step08_ensembles.py           #  Ch7  voting, bagging, forests, boosting, stacking
│   ├── step09_dimensionality.py      #  Ch8  PCA, kernel PCA, t-SNE
│   ├── step10_unsupervised.py        #  Ch9  k-means, DBSCAN, GMM, semi-supervised
│   └── step11_final_model.py         #  all  final model, test evaluation, shipping
├── tests/test_homelens.py            # automated checks of the data/feature code
├── artifacts/                        # saved models + model_card.json
├── reports/figures, reports/outputs  # every figure / printed output in this guide
├── tools/                            # step runner and PDF builder
├── requirements.txt   run_all.sh
""", "PROJECT TREE")
    f.append(H2("Professional conventions used everywhere"))
    f += BL(["**The test set is locked.** It is split off once, stratified, with a fixed seed, and is first used in Step 11. Everything before that uses cross-validation or an internal hold-out taken from the *training* set.",
             "**Everything that learns lives inside a Pipeline.** Imputers, scalers, encoders, even k-means feature generators are fitted on training folds only. This is how leakage is avoided.",
             "**Reproducibility.** One `RANDOM_STATE`, one `config.py`, a `requirements.txt`, and a runner that regenerates every number in this guide.",
             "**Baselines before models.** A model is only good relative to a dumb alternative.",
             "**Tests and a model card.** Data code is unit-tested; the shipped model has a documented purpose, metrics and limitations."])
    f += callout("tip", "How to set up and run", [
        "`pip install -r requirements.txt`, then from the project folder run `./run_all.sh` (about 10 minutes on two CPU cores) or run a single step: `python tools/run_step.py steps/step05_training_models.py`. "
        "The runner executes the script block by block (blocks are separated by `# %% name` comments) and stores each block's printed output; steps are also ordinary Python scripts. "
        "After Step 11, `PYTHONPATH=src python -m homelens.predict` scores three example districts, and `pytest -q` runs the tests."])
    f.append(PageBreak())
    return f


def roadmap():
    f = [H1("Roadmap: book chapter → project step"),
         P("Use this table as your map. The *proof* column is what you should be able to demonstrate before moving on.")]
    f += table([
        ["Book chapter", "Project step", "What the step teaches", "Proof you have learned it"],
        ["1 · ML landscape", "Step 1", "Problem framing, metric choice, baselines, under/overfitting, why data quantity matters", "You can predict the shape of a train-vs-validation curve and say which lever to pull"],
        ["2 · End-to-end project", "Steps 2 and 3 (+ 11)", "EDA, stratified split, imputation, scaling, encoding, custom transformers, pipelines, CV, randomised search, error analysis, one-shot test evaluation", "You can build a leak-free pipeline from memory"],
        ["3 · Classification", "Step 4", "Accuracy trap, confusion matrix, precision/recall, thresholds, ROC vs PR, multiclass, multilabel", "You can pick a metric and a threshold from a business cost"],
        ["4 · Training models", "Step 5", "Normal equation, batch/stochastic/mini-batch GD, learning curves, Ridge/Lasso/ElasticNet, early stopping, logistic and softmax regression", "You can implement GD and logistic regression with NumPy"],
        ["5 · SVMs", "Step 6", "Large margin, soft margin (C), kernel trick, gamma, SVR, scaling and complexity", "You can predict what C and gamma do to a boundary"],
        ["6 · Decision trees", "Step 7", "CART, Gini, regularisation, instability, regression trees", "You can compute a Gini split by hand"],
        ["7 · Ensembles", "Step 8", "Voting, bagging/OOB, random forests, boosting, stacking; model bake-off", "You can explain why each ensemble works and its failure mode"],
        ["8 · Dimensionality reduction", "Step 9", "Curse of dimensionality, PCA (SVD), kernel PCA, t-SNE", "You can compute PCA with `svd` and interpret loadings"],
        ["9 · Unsupervised learning", "Step 10", "k-means, choosing k, DBSCAN, GMM, BIC, anomaly detection, semi-supervised", "You can choose and validate a clustering"],
        ["Integration", "Step 11", "Final tuned model, honest test evaluation, segment analysis, shipping artefacts", "You can defend the model to a sceptic"]],
        widths=[1.2, 0.9, 3.3, 2.5], font=7.8)
    f.append(PageBreak())
    return f
