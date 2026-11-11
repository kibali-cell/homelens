from pdflib import *

S4 = "step04_classification"


def build():
    f = [H1("Chapter 3 · Classification"),
         P("**Project step 4** — build the HomeLens *premium-district screener*: a district is **premium** if its median value is in the top 25% of the training prices (about $263,900 or more). "
           "You will meet the accuracy trap, the confusion matrix, precision/recall, thresholds, ROC and PR curves, multiclass strategies and error analysis. Code: `steps/step04_classification.py`.")]
    f.append(H2("3.1  Concepts you must own"))
    f.append(H3("Why accuracy is not enough"))
    f.append(P("Accuracy is the fraction of correct predictions. When one class dominates it is useless: a classifier that always says 'not premium' is 75% accurate here and finds *nothing*. "
               "The same trap appears in fraud, disease, defects, churn. You need metrics that look at each class's errors separately."))
    f.append(H3("The confusion matrix and its metrics"))
    f += table([["", "Predicted negative", "Predicted positive"], ["**Actual negative**", "TN (true negative)", "FP (false positive) — a false alarm, *type I error*"], ["**Actual positive**", "FN (false negative) — a miss, *type II error*", "TP (true positive)"]], widths=[1.5, 2.2, 3.2])
    f += table([
        ["Metric", "Formula", "Answers the question", "Optimise when"],
        ["**Precision**", "TP / (TP + FP)", "When it says positive, how often is it right?", "False alarms are costly (a broker wastes a day on a bad lead)"],
        ["**Recall** (sensitivity, TPR)", "TP / (TP + FN)", "Of all real positives, how many did it find?", "Misses are costly (missed fraud, missed disease)"],
        ["**F<sub>1</sub>**", "2·P·R / (P + R) = TP / (TP + (FN + FP)/2)", "Harmonic mean: high only if *both* are high", "You want a single balance number"],
        ["**Specificity** (TNR)", "TN / (TN + FP)", "How many negatives were correctly rejected?", "Paired with recall in ROC analysis"],
        ["**FPR**", "FP / (FP + TN) = 1 − specificity", "x-axis of the ROC curve", "—"]], widths=[1.3, 1.9, 2.6, 2.4], font=7.6)
    f.append(P("The harmonic mean punishes imbalance: precision 1.0 with recall 0.1 gives F<sub>1</sub> ≈ 0.18, not 0.55. Use F<sub>β</sub> to weight recall β times more than precision."))
    f.append(H3("The precision/recall trade-off and the threshold"))
    f.append(P("Most classifiers compute a **score** (a decision function value or a probability) and compare it with a **threshold**. Raise the threshold: fewer positives, higher precision, lower recall. Lower it: the reverse. "
               "No threshold gives both. Precision can *bounce* when the threshold moves, but recall only decreases when the threshold rises. The business sets the threshold from the relative costs of FP and FN. Tools: "
               "`decision_function` or `predict_proba`, then `precision_recall_curve`. Never state 'we reached 99% precision' without the recall at that point."))
    f.append(H3("ROC curve, AUC and when to prefer PR"))
    f.append(P("The **ROC curve** plots recall (TPR) against the false-positive rate as the threshold varies. A random classifier is the diagonal; a perfect one hugs the top-left corner. "
               "**AUC** is the area under the curve (0.5 = random, 1.0 = perfect), and can be read as *the probability that a random positive is scored higher than a random negative*. "
               "Prefer the **precision–recall curve** (and average precision) when the positive class is rare or when you care more about false positives than false negatives, since ROC can look optimistic when negatives are plentiful."))
    f.append(H3("Multiclass strategies"))
    f += table([["Strategy", "How it works", "# binary models for K classes", "Notes"],
                ["Native multiclass", "Algorithm handles K classes directly", "1", "Logistic (softmax), random forests, naive Bayes, k-NN, trees"],
                ["**One-versus-the-rest** (OvR / OvA)", "One detector per class (class vs the rest); predict the class with the highest score", "K", "Default for most binary-only learners such as SGD"],
                ["**One-versus-one** (OvO)", "One classifier per *pair* of classes; the class winning most duels is chosen", "K(K−1)/2", "Each trains on a subset only; preferred for SVMs, which scale badly with training size"]],
                widths=[1.5, 2.6, 1.3, 2.4], font=7.6)
    f.append(P("**Error analysis.** Look at the *normalised* confusion matrix (each row divided by its class size) and ask what kinds of mistakes dominate. "
               "Then improve with targeted features, more data for confused classes or pre-processing. **Multilabel** classification outputs several binary labels per instance; **multioutput** generalises to several outputs, each possibly multiclass (e.g. de-noising an image pixel by pixel)."))

    f.append(H2("3.2  Project step 4 — hands on"))
    f.append(P("`load_task()` (in `tasks.py`) returns the training rows with a fitted lightweight pipeline: 16 standardised features, plus the labels *premium* (0/1) and *tier* (0–3 quartiles). The premium threshold is the 75th percentile of the **training** prices."))
    f += code(S4, "setup", skip_lines=0); f += output(S4, "setup")
    f.append(H3("The accuracy trap"))
    f += code(S4, "accuracy_trap"); f += output(S4, "accuracy_trap")
    f.append(P("The 'always not premium' classifier gets **75%** accuracy because premium is 25% of the data. This is the number any real classifier must put in perspective."))
    f.append(H3("A first classifier and its confusion matrix"))
    f += code(S4, "sgd_confusion"); f += output(S4, "sgd_confusion")
    f.append(P(f"We trained an `SGDClassifier` (a linear model trained by stochastic gradient descent, hinge loss) and used `cross_val_predict` so every prediction is out-of-fold. "
               f"Reading the matrix: 2,675 premium districts were found (TP), 1,457 were missed (FN), 661 non-premium districts were wrongly flagged (FP). "
               f"**Precision {R['sgd_precision']:.2f}** says 80% of flagged districts are truly premium; **recall {R['sgd_recall']:.2f}** says we found only 65% of all premium districts; F<sub>1</sub> = {R['sgd_f1']:.2f}. "
               "Accuracy (0.87) hides the fact that a third of the premium districts are missed."))
    f.append(H3("Choosing a threshold for the business requirement"))
    f.append(P("The brief demanded **precision ≥ 0.85–0.90**. We obtain scores with `method='decision_function'` and search the precision–recall curve for the lowest threshold that achieves 90% precision."))
    f += code(S4, "pr_curve", stop_at="fig, ax"); f += output(S4, "pr_curve")
    f += fig("fig08_pr_curve", "**Figure 8.** Left: precision rises and recall falls as the threshold grows (dashed line: the 90%-precision threshold). Right: the precision–recall curve with the chosen operating point.")
    f.append(P(f"Hitting 90% precision with this linear model costs us recall: it falls to **{R['sgd_p90_recall']:.2f}**. The model finds fewer than half of premium districts at that precision. That is the *trade-off made explicit*, and it is the number to show the stakeholders. "
               "We can improve both only with a *better model*."))
    f.append(H3("Comparing models: ROC and average precision"))
    f += code(S4, "roc_compare"); f += output(S4, "roc_compare")
    f += fig("fig09_roc", "**Figure 9.** ROC curves for the linear SGD classifier and a random forest. The forest dominates over the whole range.", 0.55)
    f.append(P(f"The forest's AUC is **{R['auc_rf']:.3f}** versus **{R['auc_sgd']:.3f}** for the linear model, and its average precision (the PR-curve area) is far higher (0.90 vs 0.82). At the default threshold it is already better on *both* precision ({R['rf_precision']:.2f}) and recall ({R['rf_recall']:.2f}). "
               "The non-linear interactions (income × location) matter. Remember: the interpretation 'a random positive outranks a random negative 96% of the time' is the intuition behind AUC."))
    f += callout("think", "ROC or PR here?", "The premium class is 25% of the data (moderately rare) and the business cares about false leads, so **PR curves and average precision** are the primary evidence. We still report ROC-AUC because it is the standard and it is threshold-independent.")
    f.append(H3("Multiclass: four price tiers"))
    f += code(S4, "multiclass"); f += output(S4, "multiclass")
    f.append(P("Now predict the quartile tier (budget, lower-mid, upper-mid, premium). `SGDClassifier` is binary, so scikit-learn automatically uses **OvR** (4 detectors); wrapping it in `OneVsOneClassifier` trains **6** (= 4·3/2) pairwise classifiers, which improved accuracy from 0.59 to 0.63 because each pair is easier to separate linearly. "
               f"The random forest handles multiclass natively and reaches **{R['tier_acc_rf']:.2f}** accuracy (chance would be 0.25)."))
    f += code(S4, "error_analysis"); f += output(S4, "error_analysis")
    f += fig("fig10_confusion_tiers", "**Figure 10.** Confusion matrices for the 4-tier forest: counts (left) and row-normalised (right). The diagonal dominates; almost all errors are one tier away.", 0.82)
    f.append(P(f"**{R['adjacent_err_pct']:.0f}%** of all mistakes land in an *adjacent* tier. That is good news for an ordinal target (tiers have an order): most errors are mild. "
               "It also suggests the problem is really a *regression* one cut into artificial bins, which is why a regression model with an error bar is more informative for borderline districts."))
    f.append(H3("Multilabel classification"))
    f += code(S4, "multilabel"); f += output(S4, "multilabel")
    f.append(P("A `KNeighborsClassifier` natively supports multilabel targets, so one model predicts both *is_premium* and *is_coastal*. For multilabel metrics, `average='macro'` gives each label equal weight, while `'weighted'` weights by label frequency. "
               "(Coastal is easy to predict from longitude/latitude and the proximity category, which is trivially related to it: F<sub>1</sub> 0.99. That label is almost leaked from a feature, a useful reminder to check whether a label is derivable from the inputs.)"))
    f += callout("warn", "Common mistakes in classification", [
        "• Reporting accuracy on imbalanced data. • Picking the threshold on the test set. • Using `predict` (0.5 threshold) when the cost of errors is asymmetric. • Reading ROC-AUC as 'quality' for a rare class without checking PR. "
        "• Forgetting that **cross-validated predictions** (not training predictions) are needed for any curve you use to choose a threshold. • Using stratified folds? `StratifiedKFold` keeps class ratios in each fold (the default for classifiers in `cross_val_*`)."])
    f.append(H2("3.3  Hands-on experiments"))
    f += NL(["Change the premium quantile from 0.75 to 0.95 (very rare class). Re-run: how do accuracy, precision, recall and the PR curve change? Which metric becomes the most informative?",
             "Add `class_weight='balanced'` to the forest. What happens to precision and recall at the default threshold?",
             "Compute `fbeta_score` with β = 0.5 and β = 2 at the 90%-precision threshold and say which stakeholder each favours.",
             "Plot the ROC curve of the *tier* classifier using OvR probabilities (`roc_auc_score(..., multi_class='ovr')`).",
             "Find the 20 districts for which the forest is most confidently wrong. What do they have in common?"])
    f.append(H2("3.4  Challenge questions"))
    f += QA([
        ("Define precision and recall in words and formulas. Which one would you maximise for cancer screening, and for a spam filter?", "Precision = TP/(TP+FP): share of alarms that are real. Recall = TP/(TP+FN): share of real cases found. Cancer screening → high recall (missing a case is worse; follow-up tests weed out false alarms). Spam filter → high precision (blocking a real email is worse than letting spam through)."),
        ("Why can't you have both 100% precision and 100% recall just by moving the threshold?", "The threshold only trades one for the other along a curve defined by the model's scores. Only a better model (better-separated scores) shifts the whole curve."),
        ("Your classifier has 98% accuracy and 0% recall on the positive class. What happened?", "The positives are rare (≈2%) and the model always predicts negative. Accuracy is misleading; evaluate with recall/precision/PR-AUC, rebalance, or change the threshold/loss."),
        ("What does AUC = 0.5 mean? AUC = 0.9?", "0.5: no discriminative power (random ranking). 0.9: a randomly chosen positive is scored above a randomly chosen negative 90% of the time."),
        ("Explain OvR versus OvO; which is cheaper for 10 classes and an SVM?", "OvR trains 10 classifiers on the *full* data; OvO trains 45 on pairs, each seeing only two classes' data. Because SVM training cost grows super-linearly with sample size, OvO's smaller subsets can be cheaper overall; scikit-learn uses OvO for SVMs."),
        ("How do you choose the decision threshold in practice?", "Estimate cost per FP and per FN (or a minimum precision requirement), use out-of-fold scores to trace the PR (or cost) curve, pick the threshold that minimises expected cost or meets the requirement, freeze it, and only then evaluate once on the test set."),
        ("Why does `cross_val_predict` matter for threshold selection?", "Scores on training data are overly confident (the model has seen the labels). Out-of-fold scores resemble what the model will output on new data, so the chosen threshold generalises."),
        ("What is the difference between multilabel and multiclass?", "Multiclass: exactly one of K classes per instance. Multilabel: any subset of K labels (each yes/no) per instance, e.g. all people recognised in a photo."),
        ("A confusion matrix shows large off-diagonal counts between two classes. What would you do?", "Inspect the examples, add features that separate them, gather more data for them, tweak preprocessing, or consider merging classes or using a pairwise (OvO) specialist. Error analysis guides the intervention."),
        ("Why might PR curves be better than ROC when positives are rare?", "ROC's x-axis is FP/(all negatives): with many negatives, even many false positives give a small FPR so the curve looks great. Precision uses FP relative to predicted positives, so it reveals the practical burden of false alarms.")])
    f.append(PageBreak())
    return f
