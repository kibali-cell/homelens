from pdflib import *

S5 = "step05_training_models"


def build():
    f = [H1("Chapter 4 · Training Models"),
         P("**Project step 5** — open the black box. You will solve linear regression three ways (normal equation, batch gradient descent, stochastic/mini-batch), read learning curves, regularise, stop early, and implement logistic and softmax regression. "
           "This chapter is the **conceptual bridge to neural networks**: everything about gradient descent here is exactly what Keras does later. Code: `steps/step05_training_models.py`.")]
    f.append(H2("4.1  Concepts you must own"))
    f.append(H3("Linear regression"))
    f.append(P("The prediction is a weighted sum of the features plus a bias: **ŷ = θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + … + θ<sub>n</sub>x<sub>n</sub> = θ·x** (add a constant feature x<sub>0</sub> = 1 so the bias is part of θ). "
               "Training means finding the θ that minimises the cost, here the **MSE** (mean squared error): MSE(θ) = (1/m) Σ (θ·x<super>(i)</super> − y<super>(i)</super>)<super>2</super>. In matrix form MSE(θ) = (1/m)‖Xθ − y‖<super>2</super>. "
               "The cost is a **convex** bowl: no local minima, a single global minimum."))
    f.append(H3("Solving it directly: the normal equation"))
    f.append(P("Setting the gradient to zero gives the closed form **θ̂ = (XᵀX)<super>−1</super> Xᵀ y**. Scikit-learn's `LinearRegression` actually uses the **SVD-based pseudo-inverse** θ̂ = X<super>+</super> y, which also works when XᵀX is singular (redundant features) "
               "and is numerically safer. Complexity: about O(n<super>2.4</super>)–O(n<super>3</super>) in the *number of features* n (slow for tens of thousands of features), but **linear in the number of instances m**. Predictions are cheap."))
    f.append(H3("Gradient descent: the engine of modern ML"))
    f.append(P("When n is huge, or when there is no closed form (logistic regression, neural nets), we *walk downhill*: start from random θ, compute the gradient of the cost, step in the opposite direction, repeat. "
               "For MSE the gradient is **∇<sub>θ</sub>MSE = (2/m) Xᵀ(Xθ − y)** and the update is **θ ← θ − η ∇<sub>θ</sub>MSE**, where **η is the learning rate**."))
    f += table([
        ["Variant", "Gradient computed on", "Characteristics", "Use"],
        ["**Batch GD**", "The whole training set at every step", "Smooth, stable, slow per step; cannot use data too big for memory", "Small/medium convex problems"],
        ["**Stochastic GD (SGD)**", "One random instance at a time", "Very noisy, jumps around the minimum; can escape local minima/plateaus; works out-of-core. Needs a **learning schedule** (decaying η) to settle", "Huge data, online learning"],
        ["**Mini-batch GD**", "A small random batch (e.g. 32–512)", "Compromise; vectorised on GPUs; the standard for deep learning", "Almost all neural network training"]],
        widths=[1.4, 1.7, 3.4, 1.6], font=7.6)
    f += callout("key", "Three facts about gradient descent you must be able to state", [
        "1. **The learning rate matters hugely**: too small converges painfully slowly; too large overshoots and *diverges*. 2. **Features must be on similar scales**; otherwise the cost surface is an elongated valley and GD zig-zags and needs many more iterations (the normal equation does not care). "
        "3. **An epoch** is one pass through the whole training set; stop when the gradient norm is below a tolerance (convergence) or, better, when validation error stops improving (early stopping)."])
    f.append(H3("Polynomial regression, and the bias/variance trade-off"))
    f.append(P("A linear model can fit non-linear data if you add powers of the features (`PolynomialFeatures`): the model stays *linear in its parameters*. With n features and degree d you get (n + d)!/(d! n!) features, which explodes combinatorially. "
               "More flexible models fit better but risk overfitting. A model's generalisation error decomposes into three parts:"))
    f += table([["Component", "Meaning", "Cause", "Reduce by"],
                ["**Bias**", "Wrong assumptions, e.g. 'it is linear'", "Underfitting", "More flexible model, better features"],
                ["**Variance**", "Excessive sensitivity to small changes in the training data", "Overfitting (too many degrees of freedom)", "More data, regularisation, simpler model, ensembling"],
                ["**Irreducible error**", "Noise in the data itself", "Measurement error, missing information", "Clean the data; cannot be modelled away"]], widths=[1.3, 2.4, 2.1, 2.2], font=7.6)
    f.append(P("Increasing complexity *increases variance and reduces bias*. The art is the sweet spot. **Learning curves** (training and validation error as a function of training-set size) diagnose which side you are on: "
               "if both curves plateau close together and *high*, the model underfits (more data will not help; need a better model); if there is a *gap* between a low training error and a higher validation error, it overfits (more data helps)."))
    f.append(H3("Regularised linear models"))
    f += table([["Model", "Cost function (α controls strength)", "Effect", "Notes"],
                ["**Ridge** (ℓ<sub>2</sub>)", "MSE + α · ½ Σ θ<sub>j</sub><super>2</super>", "Shrinks all weights towards 0; none exactly 0", "Closed form exists; **scale features first**; bias θ<sub>0</sub> is not regularised"],
                ["**Lasso** (ℓ<sub>1</sub>)", "MSE + α · Σ |θ<sub>j</sub>|", "Drives unimportant weights **exactly to 0**: automatic feature selection (sparse model)", "Can be erratic with correlated features"],
                ["**Elastic net**", "MSE + r·α·Σ|θ<sub>j</sub>| + (1−r)·α/2·Σθ<sub>j</sub><super>2</super>", "Mix of both (r = l1_ratio)", "Safe default over lasso when n > m or features correlate"]],
                widths=[1.2, 2.7, 2.3, 1.8], font=7.6)
    f.append(P("Why does ℓ<sub>1</sub> give zeros? The ℓ<sub>1</sub> penalty's gradient has constant magnitude, so it pushes small weights all the way to zero, whereas the ℓ<sub>2</sub> gradient shrinks with the weight and never quite arrives. "
               "**Early stopping** is regularisation by stopping: for gradient-based learners, validation error falls and then rises as the model starts to overfit; stop at the minimum (and keep the best weights)."))
    f.append(H3("Logistic regression (a classifier!)"))
    f.append(P("Compute a linear score t = θ·x, then squash it with the **sigmoid** σ(t) = 1/(1 + e<super>−t</super>) to get a probability p̂ = σ(θ·x). Predict class 1 if p̂ ≥ 0.5, i.e. when θ·x ≥ 0 — the decision boundary is a *hyperplane*. "
               "The cost is the **log loss**: J(θ) = −(1/m) Σ [ y log p̂ + (1 − y) log(1 − p̂) ], which punishes confident wrong answers severely. It is convex, has no closed form, and the gradient is strikingly similar to linear regression: "
               "**∇<sub>θ</sub>J = (1/m) Xᵀ(σ(Xθ) − y)**. Coefficients have a meaning: θ<sub>j</sub> is the change in the **log-odds** per unit of x<sub>j</sub>, so e<super>θ<sub>j</sub></super> multiplies the odds. "
               "Scikit-learn regularises by default (ℓ<sub>2</sub>), controlled by **C = 1/α** (larger C = weaker regularisation)."))
    f.append(H3("Softmax regression (multinomial logistic)"))
    f.append(P("For K classes compute a score per class s<sub>k</sub>(x) = θ<sub>k</sub>·x, then **p̂<sub>k</sub> = exp(s<sub>k</sub>) / Σ<sub>j</sub> exp(s<sub>j</sub>)**: probabilities that sum to 1. Train by minimising the **cross-entropy** −(1/m) Σ<sub>i</sub> Σ<sub>k</sub> y<sub>k</sub><super>(i)</super> log p̂<sub>k</sub><super>(i)</super>. "
               "With K = 2 it reduces to logistic regression. This softmax + cross-entropy pair is *exactly* the output layer of most neural network classifiers in Part II. Note that it predicts one class at a time (multiclass, not multilabel)."))

    f.append(H2("4.2  Project step 5 — hands on"))
    f.append(P("We use four standardised numeric features (income, house age, bedroom ratio, latitude). Outliers are **clipped at ±4 standard deviations**: a handful of extreme rows can otherwise make mini-batch gradients explode, a real-world gotcha. The target is expressed in units of $100,000 to keep gradients friendly."))
    f += code(S5, "setup", skip_lines=0); f += output(S5, "setup")
    f.append(H3("Solution 1: the normal equation"))
    f += code(S5, "normal_equation"); f += output(S5, "normal_equation")
    f.append(P("Three different routes (explicit inverse, SVD pseudo-inverse and scikit-learn) give **identical** parameters. The coefficients are interpretable *because the features are standardised*: +1 standard deviation of income raises the predicted price by about $108k, holding the other features fixed. Notice how the bias θ<sub>0</sub> is the mean price, as expected after centring."))
    f.append(H3("Solution 2: batch gradient descent and the learning rate"))
    f += code(S5, "batch_gd")
    f += output(S5, "batch_gd")
    f += fig("fig11_gd_learning_rate", "**Figure 11.** Batch gradient descent with four learning rates (log y-axis). η = 0.002: still far from the minimum after 200 iterations. η = 0.02: decent. η = 0.2: fastest. η = 1.0: **diverges** (the curve shoots upward).", 0.62)
    f.append(P("This is the single most useful picture in this chapter. **Too small** a learning rate wastes compute; **too large** overshoots the valley and the loss *grows*. The optimum sits just below the divergence threshold. "
               "In deep learning you will tune the very same knob with learning-rate schedulers and finders. After 1,000 iterations at η = 0.2 the result matches the normal equation to four decimal places: gradient descent *is* solving the same problem."))
    f.append(H3("Solutions 3 and 4: stochastic and mini-batch GD"))
    f += code(S5, "sgd_minibatch", stop_at="fig, ax"); f += output(S5, "sgd_minibatch")
    f += fig("fig12_sgd_minibatch", "**Figure 12.** Both SGD (one instance per step, decaying learning rate) and mini-batch GD (64 per step) approach the closed-form optimum (dashed).", 0.6)
    f.append(P("SGD's noise means it only *hovers* around the minimum; the decaying learning rate (`schedule`) makes the hover tighter over time. Mini-batch GD has less noise per step and, in vectorised form, is the most efficient. "
               "Both reach a training MSE within about 0.3% of the optimum. Their parameter vectors are slightly different from the exact solution because the cost surface is flat along some directions (correlated features): the *loss* is optimal even where individual weights are not."))
    f.append(H3("Learning curves: diagnose before you fix"))
    f += code(S5, "learning_curves"); f += output(S5, "learning_curves")
    f += fig("fig13_learning_curves", "**Figure 13.** Learning curves for three models of increasing capacity. Left: a line on income alone: both curves plateau at a high error = **underfitting**. Middle: more features lower the plateau. Right: a degree-4 polynomial has a large gap at small n that closes with more data = **variance shrinking with data**.")
    f.append(P("Use these as a decision tool. Left panel: both errors are high and equal, so *more data is pointless; the model is too simple.* Right panel: with few samples validation error is large (98,762 at n = 264 versus 66,191 at full size), meaning the flexible model is data-hungry; "
               "more data or regularisation helps. When you meet a new model, plot its learning curve before deciding what to do next."))
    f.append(H3("Regularisation in action"))
    f += code(S5, "regularisation", stop_at="fig, ax"); f += output(S5, "regularisation")
    f += fig("fig14_regularisation", "**Figure 14.** Left: validation RMSE versus regularisation strength for Ridge, Lasso and Elastic net on an overfit-prone problem. Right: Lasso zeroes out more and more coefficients as α grows.")
    f.append(P(f"The setup is deliberately an *overfit trap*: 150 training rows against 69 polynomial features. Unregularised, the training error is $47k but validation error is **${R['reg_unreg_val']:,.0f}**: a disaster. "
               f"**Ridge** brings it to ${R['reg_ridge_val']:,.0f}; **Lasso** to **${R['reg_lasso_val']:,.0f}** while keeping only **{R['lasso_nonzero']}** of 69 coefficients: it discovered which polynomial terms matter. "
               "In both cases the *validation curve is U-shaped in α*: too little regularisation overfits, too much underfits. This is the same U as in Figure 1, mirrored (more regularisation = less complexity)."))
    f.append(H3("Early stopping"))
    f += code(S5, "early_stopping", stop_at="fig, ax"); f += output(S5, "early_stopping")
    f += fig("fig15_early_stopping", "**Figure 15.** With SGD, training error keeps decreasing while validation error bottoms out and then rises. Stop at the red line.", 0.6)
    f.append(P(f"The best validation error occurs at epoch **{R['es_best_epoch']}**; training on to epoch 400 makes it **{R['es_worse_pct']:.0f}% worse**. In Keras you will use the `EarlyStopping(restore_best_weights=True)` callback that automates precisely this."))
    f.append(H3("Logistic regression from scratch"))
    f += code(S5, "logistic", stop_at="g = np.linspace"); f += output(S5, "logistic")
    f += fig("fig16_logistic_boundary", "**Figure 16.** Logistic regression with two features (income, latitude). The bold line is p = 0.5: the decision boundary is a straight line; the others show p = 0.15 and 0.85.", 0.62)
    f.append(P("Our 6-line gradient-descent loop produces the same coefficients (−1.43, 1.73, −0.06) and the same log-loss (0.398) as scikit-learn: you have now *derived and implemented* logistic regression. "
               "The odds interpretation: one standard deviation more income multiplies the odds of being premium by e<super>1.73</super> ≈ 5.6. The weak latitude coefficient says that, given income, latitude adds little *linearly* (the true geography is non-linear, which is why trees beat this model in Chapter 3)."))
    f.append(H3("Softmax regression"))
    f += code(S5, "softmax"); f += output(S5, "softmax")
    f.append(P("We compute the softmax manually from the class scores and obtain exactly `predict_proba`'s output: district 0 gets 86% budget, 13% lower-mid, 1% upper-mid, ≈0% premium (true tier: 0). "
               f"A *linear* 4-class model gets {R['softmax_acc']:.2f} accuracy, well below the forest's 0.74 from Chapter 3: a reminder that linear boundaries have limits."))
    f += callout("warn", "Common mistakes in this chapter", [
        "• Forgetting to scale features before gradient descent or regularisation. • Regularising the bias term. • Using a learning rate by habit instead of plotting the loss curve. • Calling logistic regression 'regression' and applying MSE to it. "
        "• Interpreting coefficients of un-standardised or correlated features as importances. • Early stopping on the *training* loss (it never rises!)."])
    f.append(H2("4.3  Hands-on experiments"))
    f += NL(["Without scaling the features, repeat Batch GD with η = 0.2. What happens, and what η now works? Plot the loss.",
             "Implement momentum: v ← βv − η∇, θ ← θ + v (β = 0.9). How many iterations to reach |θ − θ<sub>NE</sub>| < 0.05 versus plain GD? (This is the first step toward Adam in Part II.)",
             "Add an ℓ<sub>2</sub> penalty term to your scratch logistic regression gradient (+ αθ for non-bias terms) and compare with `LogisticRegression(C=1/α)`.",
             "Use `GridSearchCV` to tune `ElasticNet`'s `alpha` and `l1_ratio` on the 150-row problem. How many features survive?",
             "Replace the clipping by `np.log` on a skewed feature. Is the SGD more stable? Why?"])
    f.append(H2("4.4  Challenge questions"))
    f += QA([
        ("Write the gradient-descent update for linear regression with MSE, and say what each symbol is.", "θ ← θ − η (2/m) Xᵀ(Xθ − y). θ: parameter vector (including bias); η: learning rate; m: training instances; X: m×(n+1) design matrix with a leading column of ones; y: targets; (Xθ − y): residual vector."),
        ("When would you use the normal equation versus gradient descent?", "Normal equation: n (features) small to moderate, need an exact solution, no learning-rate tuning. Gradient descent: very many features, data too large for memory (SGD/mini-batch), models without closed form, online learning."),
        ("Why does feature scaling speed up gradient descent?", "Unscaled features make the cost surface a long narrow valley; the gradient points across it rather than along it, so GD zig-zags and needs a tiny learning rate. With similar scales the contours are rounder and one learning rate suits all directions."),
        ("What happens if the learning rate is too large? Too small?", "Too large: overshoots, the loss oscillates or diverges to infinity. Too small: converges very slowly (and may stall on plateaus). Diagnose with a loss-versus-iteration plot."),
        ("Is the MSE cost of linear regression convex? Of logistic regression's log loss? What does it matter?", "Both are convex, so gradient descent (with a suitable learning rate) reaches the global minimum. Neural networks' losses are *not* convex, so initial conditions and optimiser choices matter more."),
        ("Explain the bias–variance decomposition and how a learning curve reveals it.", "Error = bias² + variance + irreducible noise. High bias: train and validation errors are both high and converge. High variance: low train error, higher validation error, large gap that shrinks with more data."),
        ("Why does Lasso produce sparse models and Ridge not?", "The ℓ₁ penalty's gradient is constant (±α) so it can push a weight exactly to 0; the ℓ₂ penalty's gradient is proportional to the weight, so it shrinks weights but never zeroes them. Geometrically, the ℓ₁ ball has corners on the axes where optima tend to land."),
        ("Why must you scale features before Ridge or Lasso?", "The penalty treats all weights equally; a feature measured in large units gets a small coefficient and is barely penalised, while one in tiny units gets a big coefficient and is heavily penalised, so regularisation would depend on units."),
        ("State the formula of the sigmoid and the softmax and how they relate.", "σ(t) = 1/(1 + e⁻ᵗ). softmax(s)ₖ = e^{sₖ}/Σⱼ e^{sⱼ}. For two classes, softmax over (s, 0) equals σ(s): logistic regression is the two-class special case."),
        ("Interpret a logistic-regression coefficient of 0.7.", "Holding other features fixed, a one-unit increase in that feature increases the log-odds by 0.7, i.e. multiplies the odds by e^0.7 ≈ 2.0. (Meaningful for comparing features only if they are on comparable scales.)"),
        ("What is early stopping and what is its risk?", "Halting training when validation error stops improving, keeping the best weights. Risk: validation error is noisy (stop too early) and it consumes validation data (so keep a separate test set). Use patience and restore best weights."),
        ("In SGD, why decrease the learning rate over time?", "Noisy single-sample gradients keep bouncing around the minimum with a large η. Starting large (fast progress, escape plateaus) and shrinking η (settle down) combines speed and precision. Too fast a decay freezes training early.")])
    f.append(PageBreak())
    return f
