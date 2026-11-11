from pdflib import *

S6 = "step06_svm"


def build():
    f = [H1("Chapter 5 · Support Vector Machines"),
         P("**Project step 6** — apply large-margin classifiers and regressors to HomeLens. You will see what C and γ really do to a decision boundary, why scaling is non-negotiable, and why kernel SVMs do not scale to big data. Code: `steps/step06_svm.py`.")]
    f.append(H2("5.1  Concepts you must own"))
    f.append(H3("Large-margin classification"))
    f.append(P("For two linearly separable classes many lines separate them. An SVM chooses the one that stays **as far as possible from the closest training instances of both classes**: it fits the widest possible \"street\" between the classes. "
               "The instances that sit on the edge of the street are the **support vectors**; they alone determine the boundary: adding instances *off* the street changes nothing. "
               "The model is w·x + b: predict the positive class if it is ≥ 0. The street's width is **2/‖w‖**, so *maximising the margin = minimising ‖w‖*."))
    f.append(H3("Hard and soft margin"))
    f.append(P("A **hard margin** demands that every instance is outside the street and on the correct side. It works only if the data is linearly separable and is hypersensitive to outliers. "
               "The **soft margin** allows *margin violations* with slack variables ξ<sup>(i)</sup> ≥ 0 and balances the two goals in one objective: **minimise ½‖w‖² + C Σ ξ<sup>(i)</sup>**."))
    f += table([["Hyper-parameter", "Small value", "Large value"],
                ["**C** (penalty on violations)", "Wide street, many violations tolerated, strong regularisation → may **underfit**", "Narrow street, few violations tolerated, weak regularisation → may **overfit**"],
                ["**γ** (RBF kernel width: 1/(2σ²))", "Each instance has a wide influence, boundary smooth → may **underfit**", "Each instance has a narrow bell, boundary wiggly and local → may **overfit**"]], widths=[2, 3, 3])
    f.append(P("Equivalent view: the soft-margin cost is the **hinge loss** max(0, 1 − t·(w·x + b)) with t = ±1, plus an ℓ<sub>2</sub> penalty. This is why a linear SVM can also be trained by SGD (`SGDClassifier(loss='hinge')`), and why it is a cousin of logistic regression (log loss instead of hinge). "
               "Unlike logistic regression an SVM outputs a *score* (distance to the boundary), not a probability; `SVC(probability=True)` adds calibration via an expensive internal cross-validation."))
    f += callout("warn", "SVMs are sensitive to feature scales", "The margin is measured with a Euclidean distance. A feature with a huge range dominates the geometry and the street becomes narrow in the important directions. **Always standardise** (`StandardScaler`) before an SVM. We prove this experimentally below.")
    f.append(H3("Non-linear SVMs and the kernel trick"))
    f.append(P("Two ways to handle non-linear data: (1) add polynomial features and fit a linear SVM; (2) use a **kernel**. The **kernel trick** is the key insight: the SVM's dual formulation only needs *dot products* between instances, and a kernel function K(a, b) computes the dot product that the instances *would have* in a higher-dimensional feature space without ever building that space. "
               "So you get the effect of thousands or infinite features at the cost of computing K."))
    f += table([["Kernel", "K(a, b)", "When"],
                ["Linear", "aᵀb", "First thing to try, especially with many features or lots of data (use `LinearSVC`, much faster)"],
                ["Polynomial", "(γ aᵀb + r)<super>d</super>", "When you believe in polynomial interactions; d and r are hyper-parameters"],
                ["**Gaussian RBF**", "exp(−γ ‖a − b‖²)", "The default non-linear choice: acts like a similarity to every training instance"],
                ["Sigmoid, string kernels…", "—", "Special cases (text, sequences)"]], widths=[1.3, 2.2, 4.3])
    f.append(P("**Rule of thumb:** start with `LinearSVC`; if it is not enough and the data is not huge, try `SVC(kernel='rbf')`; tune C and γ with a grid or randomised search (exponentially spaced values)."))
    f.append(H3("Complexity and what to use"))
    f += table([["Class", "Time vs. m instances", "Needs scaling", "Kernel trick", "Out-of-core"],
                ["`LinearSVC`", "≈ O(m × n)", "yes", "no", "no"],
                ["`SGDClassifier(loss='hinge')`", "O(m × n)", "yes", "no", "**yes**"],
                ["`SVC`", "O(m² × n) to O(m³ × n)", "yes", "**yes**", "no"]], widths=[2.4, 1.9, 1.1, 1.1, 1.1])
    f.append(P("**SVM regression (SVR)** reverses the objective: fit as many instances as possible *inside* a street of half-width ε. Errors smaller than ε cost nothing (an **ε-insensitive loss**), so adding points inside the tube does not change the model. "
               "C again trades flatness for violations; larger ε gives a wider tube and fewer support vectors."))

    f.append(H2("5.2  Project step 6 — hands on"))
    f.append(P("Kernel SVMs are expensive, so we work with 12,512 pooled training rows and keep **4,000 rows as an internal hold-out taken from the training data** (the real test set remains locked)."))
    f += code(S6, "setup", skip_lines=0); f += output(S6, "setup")
    f.append(H3("A linear SVM versus logistic regression"))
    f += code(S6, "linear_svm"); f += output(S6, "linear_svm")
    f.append(P("`LinearSVC` fits in about 0.03 s and gets accuracy 0.870, F<sub>1</sub> 0.714, essentially tied with logistic regression (0.871 / 0.719): **a linear SVM and logistic regression find nearly the same boundary** because they minimise similar losses with an ℓ<sub>2</sub> penalty. "
               "Pick by what you need: probabilities (logistic) or margin/kernel options (SVM)."))
    f.append(H3("Why scaling matters, demonstrated"))
    f += code(S6, "scaling_matters"); f += output(S6, "scaling_matters")
    f.append(P("Same algorithm, same data, same hyper-parameters. On raw features (where `total_rooms` is in the thousands and `median_income` below 15) the RBF-SVC *essentially predicts 'not premium' for everything*: accuracy 0.752 is the majority-class baseline and F<sub>1</sub> is 0.11. "
               "After standardisation: accuracy 0.878 and F<sub>1</sub> 0.72. Distance-based models without scaling are a classic silent failure."))
    f.append(H3("What C and γ do to the boundary"))
    f += code(S6, "boundaries")
    f += output(S6, "boundaries")
    f += fig("fig17_svm_boundaries", "**Figure 17.** RBF-SVM decision boundaries on a 2-D slice (income, latitude) for six (γ, C) settings. Top row: γ = 0.1, 1, 50 at C = 1: increasing γ turns a smooth boundary into small islands around individual points. Bottom row: C = 0.01 (over-regularised, almost no structure), C = 100, and γ = 50 with C = 1000 (pure memorisation).")
    f.append(P("Study the numbers beside the picture. **γ = 50, C = 1000** reaches 94.5% *training* accuracy but only 74.5% on the hold-out: it memorised the sample (classic overfitting) and the boundary is a collection of islands. "
               "**C = 0.01** underfits (75% on both). The balanced choices around **γ ≈ 1, C ≈ 1–100** achieve about 84% on this 2-D slice. The tell-tale of overfitting in SVMs: *more support vectors surround each point* with large γ, and train accuracy exceeds hold-out accuracy."))
    f += callout("tip", "Intuition for γ", "Think of each training instance dropping a Gaussian bump of width 1/√(2γ). Small γ: huge bumps overlap into one smooth landscape. Large γ: tiny isolated bumps. If your model overfits, **reduce γ** (and C); if it underfits, increase them.")
    f.append(H3("Tuning on all 16 features"))
    f += code(S6, "grid_search"); f += output(S6, "grid_search")
    f.append(P(f"A 5 × 4 grid with exponentially spaced values and 3-fold CV (60 fits) found **C = {R['svc_best']['C']:.0f}, γ = {R['svc_best']['gamma']}**, scoring F<sub>1</sub> ≈ 0.76 in CV and **{R['svc_holdout_f1']:.2f}** on the 4,000-row hold-out (accuracy 0.885). "
               "Look at the pivot table: F<sub>1</sub> forms a ridge: when C is small, γ must be larger, and vice versa. C and γ interact, so tune them *together*. 29% of the training rows became support vectors."))
    f.append(H3("The scalability problem"))
    f += code(S6, "time_scaling"); f += output(S6, "time_scaling")
    f.append(P("Doubling the sample size multiplies kernel-SVC training time by roughly 3.5–5× (the exact ratio varies from run to run): between quadratic and cubic. At 12,000 rows it takes ~1 s; at 1 million rows it would take days. `LinearSVC` grows almost linearly and stays in milliseconds. "
               "This is the practical limit of kernel SVMs: great on small/medium datasets (up to tens of thousands of instances), impractical beyond."))
    f.append(H3("SVM regression"))
    f += code(S6, "svr"); f += output(S6, "svr")
    f.append(P(f"On the internal hold-out, the **RBF-SVR** trained on only 4,000 rows reaches a RMSE of about **${R['svr_best_rmse']:,.0f}**, better than a `LinearSVR` trained on 12,512 rows ($68,794). "
               "Non-linearity matters more than the sample size. (Compare with the forest and gradient-boosting results of Chapter 7: they are better still.) A wider ε tube uses fewer support vectors and trains faster, at a small cost in accuracy."))
    f.append(H2("5.3  Hands-on experiments"))
    f += NL(["Grid-search `kernel='poly'` with degree 2–4 and compare with RBF. Which wins and how much slower is it?", "Train `SVC(probability=True)` and compare the timing with `SVC()`. What does the extra time buy?",
             "Use `SGDClassifier(loss='hinge', alpha=1/(m*C))` and compare with `LinearSVC`: do the weights match?", "Plot the number of support vectors against γ. Explain the trend.",
             "Run the premium problem with `class_weight='balanced'`. Which of precision and recall moves, and why?"])
    f.append(H2("5.4  Challenge questions"))
    f += QA([
        ("What is a support vector, and what happens if you remove a non-support-vector instance?", "A support vector is a training instance on or inside the margin (or misclassified): it defines the boundary. Removing any other instance does not change the model."),
        ("Explain the role of C in a soft-margin SVM.", "C weights the penalty for margin violations relative to margin width. Small C → wide margin, more violations, simpler model (regularisation). Large C → narrow margin, fewer violations, risk of overfitting. C is the inverse of regularisation strength."),
        ("What is the kernel trick?", "Computing the inner product of instances in a high- (even infinite-) dimensional feature space via a kernel function K(a, b) without explicitly mapping to that space, so non-linear boundaries cost about the same as computing pairwise kernels."),
        ("Your RBF SVM overfits. What do you change?", "Lower γ (smoother boundary) and/or lower C. Check feature scaling, maybe reduce features, and get more data."),
        ("Why can't you use SVC on 5 million rows?", "Training time scales between O(m²) and O(m³) and kernel computations/memory grow quadratically. Use `LinearSVC`/`SGDClassifier`, approximate kernels (Nystroem, random Fourier features), or a different model family (boosting, neural nets)."),
        ("Do SVMs give probabilities?", "Not natively: only a signed distance. `SVC(probability=True)` fits a sigmoid (Platt scaling) using 5-fold CV (slower), and the probabilities can be inconsistent with `predict`."),
        ("What is the loss function of a linear SVM?", "Hinge loss max(0, 1 − t·s) (t = ±1, s = score) plus an ℓ₂ penalty on w, i.e. ½‖w‖² + C Σ hinge."),
        ("How does SVM regression differ from linear regression?", "It ignores errors inside an ε-wide tube and penalises only those outside (linearly), so it is robust to small noise and depends only on support vectors. Linear regression penalises squared error everywhere."),
        ("Why does a linear SVM tend to behave like logistic regression?", "Both are linear models with an ℓ₂ penalty and convex margin-like losses (hinge versus log). On well-behaved data they find almost identical boundaries, as seen in our experiment.")])
    f.append(PageBreak())
    return f
