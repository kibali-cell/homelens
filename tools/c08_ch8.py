from pdflib import *

S9 = "step09_dimensionality"


def build():
    f = [H1("Chapter 8 · Dimensionality Reduction"),
         P("**Project step 9** — compress and visualise the HomeLens features: you will compute PCA by hand with the SVD, test whether PCA helps models, try kernel PCA, and visualise price tiers with t-SNE. Code: `steps/step09_dimensionality.py`.")]
    f.append(H2("8.1  Concepts you must own"))
    f.append(H3("The curse of dimensionality"))
    f.append(P("High-dimensional spaces behave unlike our 2-D/3-D intuition. A random point in a unit hypercube is very likely to be extremely close to a border (in 1,000 dimensions: 86.5% chance of being within 0.001 of a face; see our experiment). "
               "Two random points are far apart, and *nearest and farthest neighbours become almost the same distance*. The practical consequences: the training set becomes sparse, predictions rely on extrapolation, "
               "distance-based methods degrade, and overfitting risk soars. The number of instances needed for a given density grows **exponentially** with the number of dimensions. Reducing dimensions can speed training, reduce overfitting and enable visualisation, at the cost of losing some information and making the pipeline more complex."))
    f.append(H3("Two approaches"))
    f += BL(["**Projection**: most data lies close to a lower-dimensional *subspace* of the high-dimensional space (many features are correlated or nearly constant); project onto it (PCA, random projection).",
             "**Manifold learning**: the data lies on a curved low-dimensional surface (a *manifold*) twisted in high-dimensional space (a rolled-up Swiss roll); *unroll* it (LLE, Isomap, t-SNE, UMAP). The **manifold assumption** says the task is simpler in the manifold coordinates (not always true)."])
    f.append(H3("PCA: the geometry and the algebra"))
    f.append(P("**PCA** (principal component analysis) finds the axis along which the data has **maximum variance** (preserving the most information, equivalently minimising the mean squared distance between the data and its projection), then the orthogonal axis with the next largest variance, and so on. "
               "The unit vectors defining the axes are the **principal components**. Algebraically: centre the data X (subtract the means), compute the **SVD** X = U Σ Vᵀ; the columns of V are the principal components; "
               "the projection onto the first d components is **X<sub>d</sub> = X V<sub>d</sub>**, and the explained variance of component j is σ<sub>j</sub>²/Σσ²."))
    f += BL(["**Always centre** (scikit-learn's PCA does it) and usually **standardise** first when features have different units.",
             "**Explained variance ratio** tells how much information each component keeps. Choose d to retain e.g. 95% (`PCA(n_components=0.95)`) or tune d by the downstream task's score (a hyper-parameter in a Pipeline).",
             "**Reconstruction**: `inverse_transform` maps back to the original space, with a *reconstruction error* from discarded components. PCA for compression keeps most variance in far fewer dimensions.",
             "**Variants**: *Randomised PCA* (`svd_solver='randomized'`) is much faster for large m, n when d is small. *Incremental PCA* processes mini-batches (data that does not fit in memory; online). *Kernel PCA* applies the kernel trick for non-linear projections. *Random projection* is a JL-lemma-based cheap method."])
    f.append(H3("Kernel PCA, LLE, t-SNE"))
    f += table([["Method", "Idea", "Strengths", "Limits"],
                ["**Kernel PCA**", "PCA in the implicit feature space of a kernel (RBF, sigmoid…)", "Non-linear; unsupervised but can be tuned with a supervised pipeline (grid search on final accuracy)", "O(m²) memory; kernel and γ are hyper-parameters"],
                ["**LLE**", "Express each point as a linear combination of its neighbours, then find a low-d embedding preserving those weights", "Unrolls twisted manifolds", "Scales poorly with m; sensitive to noise"],
                ["**t-SNE**", "Keep similar points close and dissimilar points apart (probabilistic neighbourhood matching)", "Beautiful cluster visualisations in 2-D", "**Visualisation only**; distances and cluster sizes are not meaningful; no `transform` for new data; perplexity matters"]], widths=[1.1, 2.6, 2.3, 1.8], font=7.6)
    f.append(H2("8.2  Project step 9 — hands on"))
    f += code(S9, "setup", skip_lines=0); f += output(S9, "setup")
    f.append(H3("The curse, measured"))
    f += code(S9, "curse"); f += output(S9, "curse")
    f.append(P("In 2-D the nearest neighbour is ~1% as far as the farthest. In 10,000-D, it is **97% as far**: 'near' no longer exists. This is why k-NN, RBF-SVMs and k-means degrade in very high dimensions, and why feature selection/reduction matters."))
    f.append(H3("PCA by hand"))
    f += code(S9, "pca_by_hand", stop_at="fig, ax"); f += output(S9, "pca_by_hand")
    f += fig("fig26_pca", "**Figure 26.** Left: scree plot with cumulative explained variance (dashed line: 95%). Right: districts projected on the first two principal components, coloured by price.")
    f.append(P(f"We computed PCA **two ways**: the SVD of the centred data by hand, and `sklearn.decomposition.PCA`. Explained variance ratios agree to machine precision (difference 2×10<super>−16</super>): PCA *is* the SVD. "
               f"Our 16 features need **{R['pca_d95']} components for 95%** and **{R['pca_d99']} for 99%** of the variance. "
               "The loadings table is interpretable: **PC1** is dominated by `total_rooms`, `total_bedrooms`, `population`, `households` (all ≈ 0.47–0.49): a *district size* axis. **PC2** contrasts latitude/ratios with longitude: a *geography* axis (north-west versus south-east). "
               "PC1, the biggest source of variance, says almost nothing about price (Figure 26 right: the colours are not ordered along PC1). **High variance ≠ predictive power**. PCA is unsupervised."))
    f.append(H3("Reconstruction and PCA variants"))
    f += code(S9, "reconstruction"); f += output(S9, "reconstruction")
    f.append(P("Reconstruction error falls as components are added; with 11 of 16 components 99.3% of the variance is kept and the reconstruction error is 0.07 standard deviations. Incremental and randomised PCA reproduce the exact solution, differing only by 10<super>−4</super> or less in explained variance: use them when data is big."))
    f.append(H3("Does PCA help a supervised model? (honest test)"))
    f += code(S9, "pca_before_model", stop_at="fig, ax"); f += output(S9, "pca_before_model")
    f += fig("fig27_pca_models", "**Figure 27.** CV RMSE as a function of PCA components kept before a linear model or a random forest. The dashed line is the forest on the original features.", 0.6)
    f.append(P(f"Result: *PCA hurt both models*. The forest on the original 16 features has CV RMSE ${R['pca_rf_raw']:,.0f}; with all 16 components (a pure rotation, no information lost) it is already worse (${R['pca_rf_k16']:,.0f}), and with 8 components (95% variance) it is ${R['pca_rf_k8']:,.0f}. "
               "Two reasons: trees rely on axis-aligned, meaningful features (rotation destroys that), and high-variance directions are not the predictive ones. For the linear model, 16 components reproduce the original score (68.7k vs 68.4k), and fewer components lose signal. "
               "**Lesson: dimensionality reduction is a hypothesis, and you validate it like any other.** With only 16 well-chosen features we have no need for it. It becomes valuable with hundreds of correlated features, for visualisation, or for denoising."))
    f.append(H3("Kernel PCA with a supervised grid search"))
    f += code(S9, "kernel_pca"); f += output(S9, "kernel_pca")
    f.append(P("We tuned the kernel and γ by using the **final classifier's CV accuracy** as the criterion (the standard way to evaluate an unsupervised step in a supervised pipeline). The best RBF kernel PCA (γ = 0.01, 6 components) obtained 83.9%, no better than plain PCA with 6 components (84.4%) on this data: non-linearity in the *feature space* adds nothing because the relevant non-linearity is in the *label* structure, which these classifiers can handle after projecting."))
    f.append(H3("t-SNE for visualisation"))
    f += code(S9, "tsne"); f += output(S9, "tsne")
    f += fig("fig28_tsne", "**Figure 28.** The same 3,000 districts with PCA (left) and t-SNE (right), coloured by price tier. t-SNE separates the tiers into visibly structured regions; PCA does not.")
    f.append(P("t-SNE shows that districts with similar prices form neighbourhoods in the full feature space: a kNN on the 2-D t-SNE coordinates gets 57% tier accuracy versus 33% on the 2-D PCA coordinates. **Caution**: t-SNE was fitted on all 3,000 points at once (it cannot transform new data), so that accuracy is a statement about *visual separability*, not a valid generalisation estimate. Use t-SNE to *look*, never to feed production models."))
    f += callout("warn", "Common mistakes in dimensionality reduction", ["• Fitting PCA on all data before the train/validation split (leakage): put it in the Pipeline. • Applying PCA without scaling. • Believing 'more variance = more important'. • Interpreting distances/cluster sizes in a t-SNE plot. • Using PCA before tree models by habit."])
    f.append(H2("8.3  Hands-on experiments"))
    f += NL(["Use `PCA(n_components=0.95)` inside a `Pipeline` with logistic regression and cross-validate. How many components are kept in each fold?", "Compress a few districts with 4 components and reconstruct them. Which original features are reconstructed worst? Why?",
             "Re-run t-SNE with perplexity 5, 40, 200. What changes?", "Try `LocallyLinearEmbedding(n_neighbors=20)` on the 3,000-row sample. Is the result as clear as t-SNE?", "Project the standardised data with `GaussianRandomProjection(eps=0.3)` and measure how well pairwise distances are preserved."])
    f.append(H2("8.4  Challenge questions"))
    f += QA([
        ("What does PCA maximise, and what is the connection to the SVD?", "It finds orthogonal directions of maximal variance (equivalently minimal reconstruction error). With centred X = UΣVᵀ, the columns of V are the principal components and σ²/(m−1) are the variances along them."),
        ("Why standardise before PCA?", "PCA maximises variance, so a feature measured in large units would dominate the first component purely because of its scale. Standardising puts features on equal footing (unless the original units are meaningful)."),
        ("How do you choose the number of components?", "Keep enough to explain a target share of variance (e.g. 95%), look for an elbow in the scree plot, or treat d as a hyper-parameter tuned by the downstream model's cross-validated performance."),
        ("Does PCA improve classifier accuracy?", "Not necessarily. It can reduce noise, multicollinearity and cost, but it can also discard predictive low-variance directions. Test it with cross-validation (in our project it lowered accuracy)."),
        ("PCA vs t-SNE?", "PCA is linear, deterministic, invertible, gives a transform for new data and is suitable as a preprocessing step. t-SNE is non-linear and stochastic, preserves local neighbourhoods for visualisation, cannot transform new data, and its global geometry is not meaningful."),
        ("What is the curse of dimensionality?", "As dimensions grow, data become sparse; distances concentrate; the volume near the surface dominates; required sample size grows exponentially; models overfit and distance-based methods degrade."),
        ("When use Incremental or Randomised PCA?", "Incremental: data too big for memory or arriving in streams (online). Randomised: large data with small target dimension d, much faster at nearly the same accuracy."),
        ("What is reconstruction error and how does it relate to explained variance?", "The mean squared distance between the data and its projection mapped back. It equals the sum of the variances of the discarded components: more explained variance → smaller reconstruction error."),
        ("Why use a pipeline with PCA in cross-validation?", "So that PCA is re-fitted on each training fold. Otherwise validation data influences the learned components (leakage).")])
    f.append(PageBreak())
    return f
