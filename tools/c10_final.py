from pdflib import *

S11 = "step11_final_model"


def build():
    f = [H1("Chapter 10 · Integration: the final model, the one-shot test, and shipping"),
         P("**Project step 11** — everything now comes together. You tune the winning algorithm, open the locked test set **once**, analyse where the model is weak, and ship a documented, tested artefact. This is the part clients and reviewers actually judge. Code: `steps/step11_final_model.py`, `src/homelens/predict.py`, `tests/test_homelens.py`.")]
    f.append(H2("10.1  The decision trail"))
    f += table([["Decision", "Evidence", "Chapter"],
                ["Use RMSE/MAE in dollars; baseline = mean predictor ($115.7k)", "Framing and baselines", "1"],
                ["Stratified split on income category; test set locked", "Random split deviates up to 5% in income mix; stratified 0.5%", "2"],
                ["Log, ratio and cluster-similarity features inside a Pipeline", "Ratios correlate better; clusters cut RMSE by 9.7%", "2, 9"],
                ["Gradient-boosted trees over linear/SVM/single tree", "CV RMSE $46.1k vs $68.4k (linear) vs $60.0k (tree) vs $49.2k (forest)", "4–7"],
                ["Histogram GB over stacking", "Equal within noise; 20× faster; handles missing values", "7"],
                ["No PCA", "PCA lowered CV accuracy/RMSE", "8"],
                ["Classifier threshold from out-of-fold PR curve", "Never from the test set", "3"]], widths=[3.3, 3.5, 0.7], font=7.8)
    f.append(H2("10.2  Tuning the final regressor"))
    f += code(S11, "setup", skip_lines=0); f += output(S11, "setup")
    f += code(S11, "tune_regressor"); f += output(S11, "tune_regressor")
    f.append(P(f"A randomised search of 25 candidates × 4 folds jointly tuned the *preprocessing* (number of geographic clusters) and the booster (learning rate, number of iterations, leaf count, minimum leaf size, ℓ<sub>2</sub> regularisation), "
               f"using log-uniform distributions for scale-type parameters. Best CV RMSE: **${R['final_cv_rmse']:,.0f}**. That figure is our **expectation** for new data, with a fold-to-fold uncertainty around ±$1k. We now stop tuning."))
    f.append(H2("10.3  The one-shot test evaluation"))
    f += code(S11, "evaluate_test"); f += output(S11, "evaluate_test")
    f.append(P(f"On the 4,128 locked districts: **RMSE ${R['test_rmse']:,.0f}**, MAE ${R['test_mae']:,.0f}, R² {R['test_r2']:.3f}. The 95% confidence interval for the generalisation RMSE is **[${R['test_ci_low']:,.0f}, ${R['test_ci_high']:,.0f}]**, computed from the squared test errors with a t-interval and then square-rooted. "
               f"The mean-predictor gets ${R['test_baseline_rmse']:,.0f}, so the model removes **63% of the baseline error**. The test RMSE is 5% *lower* than the CV estimate: well within the interval, and the direction is plausible (the final model trains on 100% of the training data versus 75% in each CV fold). "
               "**What if the test error had been much worse than CV?** That would signal leakage, distribution shift or over-tuning to the folds. You would investigate; you would **not** go back and tune on the test set (that would burn it)."))
    f += callout("key", "Why the test set is sacred", "Every time you look at test results and then change something, information about the test set flows into your decisions. After a few cycles the test score is just another validation score and the estimate is optimistic. If you must iterate after seeing it, get *new* data.")
    f.append(H2("10.4  Where is the model weak? Segment analysis"))
    f += code(S11, "segment_analysis", stop_at="fig, ax"); f += output(S11, "segment_analysis")
    f += fig("fig33_test_diagnostics", "**Figure 33.** Test-set diagnostics. Left: predicted vs actual. Middle: residual histogram (centred at 0, fat tails). Right: residuals versus predictions (variance grows with price: heteroscedastic).")
    f.append(P(f"An aggregate RMSE hides important structure. **{R['within20']:.0f}%** of districts are predicted within ±20% of the truth. *Absolute* error is larger for expensive/coastal districts (MAE $29–36k) than inland (MAE $20k), but *percentage* error is lowest for high-income districts (10.7% versus 19.1% for low-income): "
               "cheap districts are hard in relative terms because a $20k miss on a $90k district is large. The **capped** districts are predicted with an average error of $53k (twice the rest): the model cannot see beyond $500k. "
               "The three ISLAND districts have a mean error of $112k: 5 rows in the entire dataset, so the model cannot learn that category. These findings go in the model card as known limitations."))
    f.append(H2("10.5  The premium screener"))
    f += code(S11, "premium_classifier"); f += output(S11, "premium_classifier")
    f.append(P(f"The classifier is a histogram gradient-boosting classifier. The threshold is chosen on **out-of-fold training predictions**: the max-F<sub>1</sub> point (probability {R['clf_threshold_prob']:.2f}) and the point that gives 90% precision. On the test set the max-F<sub>1</sub> operating point reaches **precision {R['clf_test_precision']:.2f}, recall {R['clf_test_recall']:.2f}** (ROC-AUC {R['clf_test_auc']:.3f}); "
               "the 90%-precision threshold gives 0.93 precision and 0.77 recall. Compare this to the linear SGD classifier from Chapter 3 (precision 0.80 / recall 0.65 at its default threshold): the acceptance bar of *precision ≥ 0.85* is now met with recall of 0.84, rather than the 0.47 we would have had with the linear model at 90% precision. "
               "The business can choose between the operating points knowing exactly what each costs."))
    f.append(H2("10.6  Shipping: artefacts, model card, inference, tests"))
    f += code(S11, "ship"); f += output(S11, "ship", 22)
    f.append(P("We persist the entire fitted **pipeline** (preprocessing + model) with `joblib`: raw data in, prediction out, with no possibility of training/serving skew. A **model card** (`artifacts/model_card.json`) documents purpose, data, metrics, limitations and the seed. "
               "`predict.py` demonstrates inference on three new districts (including one with a missing bedroom count, which the pipeline imputes):"))
    f += snippet("""
$ PYTHONPATH=src python -m homelens.predict
   median_income ocean_proximity  predicted_value  premium_probability  is_premium
0           8.33        NEAR BAY         444400.0                0.984        True
1           3.20      NEAR OCEAN         127700.0                0.006       False
2           2.10          INLAND          61200.0                0.002       False
""", "OUTPUT of predict.py on three new districts")
    f += snippet("""
tests/test_homelens.py  (run: pytest -q)
  test_split_is_stratified_and_disjoint ........ income mix of the test set within 0.5% of the full data
  test_split_is_deterministic .................. same seed -> identical split
  test_preprocessing_has_no_nans_and_is_scaled . imputation + scaling behave
  test_preprocessing_handles_unseen_category ... "MARS" does not crash inference
  test_cluster_similarity_shape_and_range ...... custom transformer contract
  test_premium_label_uses_train_threshold_only . 25% of train labelled premium
""", "AUTOMATED TESTS")
    f += callout("tip", "Deployment and maintenance checklist", [
        "• Wrap `predict.score` behind an API or a scheduled batch job; validate inputs (ranges, missing columns, caps). • Log inputs and predictions. • Monitor: input drift (e.g. PSI per feature), prediction drift, error against delayed ground truth, share of missing/capped values. "
        "• Retrain on a schedule or when drift/performance triggers fire; keep dated backups of data, code and model so you can roll back. • Re-run the unit tests and a **shadow comparison** with the previous model before promoting a new one."])
    f.append(H2("10.7  Defend your model: the sceptic's interview"))
    f.append(P("Answer these aloud as if a hostile reviewer were asking. They draw on every chapter. Model answers follow."))
    f += QA([
        ("Why should I believe your reported RMSE of about $41.8k?", "Because the test set was split once with stratification and a fixed seed, never used for any decision, and the CV estimate ($44.0k) agrees with it within the 95% CI [$40.0k, $43.6k]. All preprocessing sits inside a pipeline, so no information leaks from validation folds."),
        ("Your model has R² = 0.87. Is it good?", "R² is relative to the mean baseline; what matters is business error: median relative error ~13–19% depending on income band, 77% of districts within ±20%. For *ranking and screening* that is useful; for pricing an individual home it is not (and the model card says so)."),
        ("Why not a neural network?", "On 16k rows of tabular data, gradient-boosted trees usually match or beat neural nets with far less tuning and compute. We will revisit neural nets in Part II and benchmark them on the same split."),
        ("Your classifier's recall is 0.84. What do the missed 16% look like?", "Mostly borderline districts just under/over the threshold (errors are overwhelmingly adjacent-tier) and capped/high-variance coastal areas. Examining the false negatives is the next error-analysis step."),
        ("How do you know the model will work next year?", "I don't: this is 1990 data. A deployed version needs monitoring for drift, periodic retraining on fresh data, and an out-of-time test set (train on earlier years, test on later ones)."),
        ("A colleague suggests scaling the full dataset once and then cross-validating, to 'save time'. Respond.", "That leaks validation statistics into training (the scaler sees the validation fold). The effect may be small for scaling but large for target-related steps (feature selection, target encoding). A Pipeline costs nothing and removes the question."),
        ("What would you do if the business needs predictions above $500k?", "Obtain uncapped labels, or restrict claims to the uncapped range and flag any district predicted near the cap. A censored-target model is another option."),
        ("Which single change would most improve the model?", "Better location information (neighbourhood-level data, sale prices of individual homes): error maps show large residuals in dense metros where prices vary over short distances; and uncapped labels.")])
    f.append(PageBreak())
    return f
