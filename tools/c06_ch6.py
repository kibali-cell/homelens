from pdflib import *

S7 = "step07_trees"


def build():
    f = [H1("Chapter 6 · Decision Trees"),
         P("**Project step 7** — grow trees on HomeLens, compute a Gini split by hand, tune regularisation, and *see* their instability. Trees are the building block of forests and boosting (Chapter 7), the strongest models in this guide. Code: `steps/step07_trees.py`.")]
    f.append(H2("6.1  Concepts you must own"))
    f.append(H3("What a tree is"))
    f.append(P("A decision tree is a flowchart of yes/no questions on single features. Each **internal node** tests one feature against a threshold; each **leaf** gives a prediction. "
               "Trees need **little preprocessing** (no scaling, no centring; they handle mixed feature types) and are **white-box**: you can read the rules. That is why they are popular for explanations and why they are the base of the best tabular learners."))
    f.append(P("`predict_proba` for a classification tree returns the **fraction of each class among the training instances in the leaf**; the predicted class is the most frequent one. For regression the leaf returns the **mean target** of its instances."))
    f.append(H3("CART: how a tree is grown"))
    f.append(P("Scikit-learn uses **CART** (Classification And Regression Trees), a greedy, recursive algorithm. At each node it searches *every feature* and *every threshold* for the split (k, t<sub>k</sub>) that minimises the cost "
               "**J(k, t<sub>k</sub>) = (m<sub>left</sub>/m) G<sub>left</sub> + (m<sub>right</sub>/m) G<sub>right</sub>**, where G measures impurity. It then recurses on both halves until it reaches `max_depth`, cannot find a split that reduces impurity, or violates a stopping rule."))
    f += table([["Impurity", "Formula", "Notes"],
                ["**Gini** (default)", "G = 1 − Σ<sub>k</sub> p<sub>k</sub><super>2</super>", "0 when a node is pure; slightly faster; tends to isolate the most frequent class"],
                ["**Entropy**", "H = −Σ<sub>k</sub> p<sub>k</sub> log<sub>2</sub> p<sub>k</sub>", "0 when pure; slightly more balanced trees; results usually nearly identical"],
                ["**MSE** (regression)", "MSE<sub>node</sub> = (1/m<sub>node</sub>) Σ (ŷ<sub>node</sub> − y<super>(i)</super>)<super>2</super>", "Leaf predicts the mean; the same J formula with MSE instead of G"]], widths=[1.4, 3, 3.4])
    f += callout("key", "CART is greedy", "It picks the best split *now* and never reconsiders. It does not guarantee the globally optimal tree (finding it is NP-complete), but it is a reasonable solution. **Cost:** prediction O(log<sub>2</sub> m) (fast); training O(n × m log m) because every feature is considered at every node.")
    f.append(H3("Regularising a tree"))
    f.append(P("A tree with no constraints is **non-parametric**: it grows as complex as the data demands, fitting the training set perfectly (overfitting). Constrain it with hyper-parameters (increase `min_*` or reduce `max_*`):"))
    f += table([["Hyper-parameter", "Effect"],
                ["`max_depth`", "Maximum number of levels. The simplest, most common lever"],
                ["`min_samples_split` / `min_samples_leaf`", "Minimum instances to split a node / to exist in a leaf. Larger = smoother, more regularised"],
                ["`max_leaf_nodes`, `max_features`", "Cap the leaves; limit the features considered at each split (adds randomness, used by forests)"],
                ["`min_impurity_decrease`, `ccp_alpha`", "Only split if impurity drops enough; cost-complexity **pruning** after growing"]], widths=[2.6, 5.2])
    f.append(H3("Weaknesses: orthogonal boundaries and instability"))
    f += BL(["**Axis-aligned splits.** Every split is perpendicular to a feature axis, so a diagonal boundary needs a staircase of many splits. Rotating the data (e.g. with PCA) can change the tree's quality dramatically.",
             "**High variance.** Small changes in the training data (removing one instance, rotating the axes, a different random seed when ties occur) can produce a very different tree. This instability is exactly what *ensembles of trees* fix by averaging.",
             "**Extrapolation.** Trees predict constants within regions: they cannot extrapolate a trend beyond the training range, unlike linear models.",
             "**Overfitting** if unconstrained, and sensitivity to class imbalance (balance or weight the classes)."])

    f.append(H2("6.2  Project step 7 — hands on"))
    f += code(S7, "setup", skip_lines=0)
    f.append(H3("Reading a small tree"))
    f += code(S7, "shallow_tree", stop_at="fig, ax"); f += output(S7, "shallow_tree", 30)
    f += fig("fig18_tree_depth3", "**Figure 18.** The depth-3 tree plotted by `plot_tree`: each node shows its rule, Gini impurity, sample count and class counts; colours encode the majority class.")
    f.append(P("Thresholds are in *standardised* units because the data passed through a `StandardScaler` (trees would be equally happy with raw values). The first rule says: *if median income ≤ 0.91 standard deviations above the mean → go left*. "
               "Trace one district through the tree in the figure, and then through the text export: you should be able to read any path as an *if-then rule*. A depth-3 tree already achieves 84% validation accuracy using only a handful of rules."))
    f.append(H3("Verifying Gini and the CART cost by hand"))
    f += code(S7, "gini_by_hand"); f += output(S7, "gini_by_hand")
    f.append(P(f"The root has 12,384 training rows, 25% premium: Gini = 1 − (0.75² + 0.25²) = **0.3752**. CART tried every feature and threshold and found that splitting on **{R['root_feature']} at ≈ {R['root_thr_income']:.2f}** (units of $10,000, i.e. $56,100) "
               f"gives a weighted child impurity of **{R['gini_after_split']:.4f}**: the largest decrease available. The right child (1,753 rows, income above the threshold) is still impure (0.314) because not every high-income district is premium: location matters. "
               "This calculation is the *whole* algorithm repeated recursively. `predict_proba` for a district in a leaf with 94.6% non-premium returns [0.946, 0.054]: leaf fractions, not magic."))
    f.append(H3("An unconstrained tree overfits"))
    f += code(S7, "overfit_tree"); f += output(S7, "overfit_tree")
    f.append(P(f"With no limits the tree reaches **depth {R['full_tree_depth']} and {R['full_tree_leaves']:,} leaves**, 100% training accuracy but only {R['full_tree_val']:.3f} validation accuracy. Almost one leaf per 12 training rows: it has memorised the noise."))
    f.append(H3("Regularisation curves"))
    f += code(S7, "regularise", stop_at="fig, ax"); f += output(S7, "regularise")
    f += fig("fig19_tree_regularise", "**Figure 19.** 5-fold CV accuracy versus `max_depth` (left) and `min_samples_leaf` (right). Training accuracy keeps rising; CV accuracy peaks and then falls.")
    f.append(P(f"The sweet spot is **max_depth ≈ {R['tree_best_depth']}** or **min_samples_leaf ≈ {R['tree_best_leaf']}** (CV accuracy {R['tree_best_acc']:.3f}, versus 0.859 for the unconstrained tree). Again the U-shaped validation curve: *this is the same bias/variance picture, now for trees*. "
               "Notice that `min_samples_leaf` can express smoothness more directly than `max_depth`: it prevents tiny leaves wherever they appear, not just at one depth."))
    f.append(H3("Instability, demonstrated"))
    f += code(S7, "instability")
    f += output(S7, "instability")
    f += fig("fig20_tree_instability", "**Figure 20.** Left: a depth-6 tree on the original 2-D axes. Middle: the *same data rotated 35°*: a different staircase boundary and a lower score. Right: remove a single training row and the boundary changes.")
    f.append(P("The same tree algorithm gives visibly different decision regions after (a) a rotation and (b) deleting a single training row. Rotating the whole 16-feature dataset with PCA lowered the CV accuracy (0.878 → 0.861) here: PCA is not a free lunch for trees. "
               "*This variance is the motivation for Chapter 7*: averaging hundreds of such unstable trees yields a stable, powerful model."))
    f.append(H3("Regression trees"))
    f += code(S7, "regression_tree"); f += output(S7, "regression_tree")
    f += fig("fig21_regression_trees", "**Figure 21.** Regression trees on income only: a depth-2 tree is 4 horizontal steps; depth 4 follows the trend; an unconstrained tree chases individual points.")
    f.append(P("The depth-2 tree is readable: income ≤ 3.07 → $135,890; 3.07–5.09 → $209,790; 5.09–6.82 → $292,300; above 6.82 → $421,590. Each leaf predicts the **mean price of its districts**, so the prediction is a staircase and **cannot extrapolate**. "
               f"Using all 16 features at depth 8, the regression tree reaches a CV RMSE of ${R['tree_reg_rmse']:,.0f}: better than linear regression ($68k) but far behind the forest (~$49k), which averages many such trees."))
    f += callout("warn", "Common mistakes with trees", [
        "• Trusting training accuracy of an unconstrained tree. • Interpreting a deep tree as 'explainable'; 1,000 leaves are not. • Using impurity-based feature importances as causal truth. • Expecting trees to extrapolate or to benefit from scaling. • Forgetting `random_state`: ties in split quality can make results vary run to run."])
    f.append(H2("6.3  Hands-on experiments"))
    f += NL(["Use `criterion='entropy'` and compare accuracy and tree shape with Gini. Are the root splits the same?", "Use `ccp_alpha` (cost-complexity pruning): plot accuracy against alpha from `cost_complexity_pruning_path`.",
             "Fit `DecisionTreeClassifier(class_weight='balanced')`. How do precision and recall of the premium class change?", "Bootstrap 20 trees on resampled training sets, and compare their root splits. How often does the root change? (Preview of bagging.)",
             "Plot a regression tree's prediction beyond the training range of income. What do you observe?"])
    f.append(H2("6.4  Challenge questions"))
    f += QA([
        ("Compute the Gini impurity of a node with 30 positives and 10 negatives.", "p = 0.75, 0.25 → G = 1 − (0.5625 + 0.0625) = 0.375."),
        ("How does CART choose a split?", "For each feature and each candidate threshold it computes the weighted average impurity of the two resulting children, J = (m_l/m)G_l + (m_r/m)G_r, and picks the (feature, threshold) pair with the smallest J. Then it recurses (greedy, top-down)."),
        ("Why are decision trees called 'non-parametric'? Why does that mean they overfit easily?", "The number of parameters is not fixed before training; it grows with the data. Without constraints the structure adapts until it fits the training set perfectly. Regularise via depth/leaf-size limits or pruning."),
        ("Do trees need feature scaling? Why or why not?", "No. Splits compare a single feature with a threshold; monotone rescaling does not change which instances go left or right."),
        ("List three ways to regularise a decision tree.", "Limit `max_depth`; raise `min_samples_leaf`/`min_samples_split`; limit `max_leaf_nodes`/`max_features`; prune with `ccp_alpha`."),
        ("Why are trees unstable and what fixes that?", "Greedy splits depend on small differences in impurity, so small data perturbations change early splits and thus the whole subtree. Averaging many trees (bagging, random forests) reduces the variance."),
        ("Gini or entropy?", "Usually little difference. Gini is a bit cheaper; entropy can give slightly more balanced trees. Choose by cross-validation if it matters."),
        ("How does a regression tree make predictions and what shape does the prediction function have?", "Each leaf predicts the mean target of its training instances; the prediction function is piecewise-constant (a staircase), which cannot extrapolate beyond the training range."),
        ("A tree has 100% training accuracy and 80% test accuracy. What would you do?", "It overfits. Constrain it (e.g. `min_samples_leaf`≥10, `max_depth`), choose the values by cross-validation, or move to a random forest/gradient boosting to cut variance."),
        ("Why can rotating the data with PCA help or hurt a decision tree?", "Trees split along feature axes; if the discriminating direction is diagonal, rotating so it aligns with an axis simplifies the tree (help). If the original axes were meaningful, PCA mixes them and makes splits harder (hurt), as in our experiment.")])
    f.append(PageBreak())
    return f
