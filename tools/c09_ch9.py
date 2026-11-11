from pdflib import *

S10 = "step10_unsupervised"


def build():
    f = [H1("Chapter 9 · Unsupervised Learning Techniques"),
         P("**Project step 10** — find HomeLens's market segments, use clusters as features, test a semi-supervised shortcut, detect dense urban cores with DBSCAN, and flag anomalous records with a Gaussian mixture. Code: `steps/step10_unsupervised.py`.")]
    f.append(H2("9.1  Concepts you must own"))
    f.append(P("Most data is unlabelled. Unsupervised learning extracts structure without labels. The three classic tasks: **clustering** (group similar instances), **anomaly detection** (learn what is normal; flag what is not) and **density estimation** (model the probability density function of the data, which also enables anomaly detection). "
               "Typical uses: customer segmentation, data analysis, dimensionality reduction, **feature engineering**, anomaly/fraud detection, semi-supervised learning, search by similarity, and image segmentation."))
    f.append(H3("K-means"))
    f.append(P("Pick k, then iterate: **assign** each instance to the nearest centroid; **update** each centroid to the mean of its instances. It stops when assignments stop changing (guaranteed to converge, but maybe to a *local* optimum). "
               "It minimises the **inertia**: the sum of squared distances between instances and their closest centroid. It is fast (roughly linear in m, k, n) but assumes **roughly spherical, similar-sized clusters** and needs **scaled features**. It produces *hard* assignments (`predict`) and distances to all centroids (`transform`)."))
    f += table([["Topic", "Facts"],
                ["**Initialisation**", "Random init can land in poor optima (inertia spread of 89% across seeds in our experiment). **k-means++** (default) picks well-spread initial centroids; `n_init` restarts keep the best inertia"],
                ["**Mini-batch k-means**", "Uses mini-batches: faster (and out-of-core) at a slightly worse inertia"],
                ["**Choosing k**", "*Inertia* always decreases with k (look for an **elbow**, often ambiguous). The **silhouette coefficient** (b − a)/max(a, b) per instance (a = mean intra-cluster distance, b = mean distance to the nearest other cluster) ranges from −1 to +1; take its mean; higher is better. The silhouette diagram shows cluster sizes and quality"],
                ["**Limits**", "Must choose k; fails on elongated, differently sized or non-convex clusters; sensitive to outliers and scale"]], widths=[1.5, 6.3], font=7.8)
    f.append(H3("Using clustering"))
    f += BL(["**Preprocessing / feature engineering**: replace the data with each instance's distance (or similarity) to each centroid. We do this in `ClusterSimilarity`.",
             "**Semi-supervised learning**: when labels are costly, cluster, label only the *representative instance* of each cluster, then propagate its label to the cluster (or to the instances near the centroid). It works when clusters align with classes.",
             "**Active learning** (the model asks a human to label the instances it is least certain about), and **image segmentation** (cluster pixel colours)."])
    f.append(H3("DBSCAN, hierarchical clustering"))
    f.append(P("**DBSCAN** defines clusters as continuous regions of **high density**. A *core instance* has at least `min_samples` neighbours within distance `eps`; clusters are chains of core instances and their neighbours; everything else is **noise** (label −1). "
               "It finds clusters of arbitrary shape, needs no k, and is robust to outliers, but struggles with clusters of *different densities* and has no `predict` (use a classifier on its output for new points). "
               "**Agglomerative (hierarchical) clustering** repeatedly merges the closest clusters (with a linkage rule such as Ward) building a dendrogram; flexible with connectivity constraints, but memory-heavy."))
    f.append(H3("Gaussian mixture models (GMM)"))
    f.append(P("A GMM assumes the data are generated from a mixture of k Gaussians with unknown means, covariances and mixing weights, and fits them by **Expectation-Maximisation (EM)**: *E-step* computes each instance's soft responsibility for each cluster; *M-step* re-estimates the parameters using those responsibilities (a soft k-means). "
               "It gives **soft clustering** (`predict_proba`), ellipsoidal clusters (`covariance_type`), a generative model (`sample`) and a **density** (`score_samples`). "
               "Choose k with the **BIC** (Bayesian information criterion = p·ln m − 2 ln L̂) or **AIC** (2p − 2 ln L̂), where p is the number of parameters and L̂ the maximised likelihood: they reward fit and punish complexity (BIC more strongly)."))
    f.append(P("**Anomaly detection with a density model**: instances in regions of density below a threshold are anomalies. The threshold sets the false-positive/false-negative trade-off (e.g. flag the 2% lowest density). **Novelty detection** assumes a *clean* training set; anomaly detection assumes it may be contaminated. Other tools: Isolation Forest, One-Class SVM, PCA reconstruction error, autoencoders (Part II)."))
    f.append(H2("9.2  Project step 10 — hands on"))
    f += code(S10, "setup", skip_lines=0)
    f.append(H3("How many market regions? Inertia and silhouette"))
    f += code(S10, "choose_k", stop_at="fig, ax"); f += output(S10, "choose_k")
    f += fig("fig29_choose_k", "**Figure 29.** Elbow and silhouette plots for k-means on (longitude, latitude).", 0.8)
    f.append(P("The inertia curve bends smoothly with no sharp elbow, and the silhouette is highest at k = 2 (north/south California: 0.75), falling after that. **There is no single correct k.** "
               "k = 2 is mathematically 'best' but useless for a business that needs regional markets; k ≈ 6–8 balances detail and stability (silhouette ≈ 0.53–0.57). This is a typical situation: the metric guides, domain knowledge decides. We choose **k = 8** for the segmentation report."))
    f.append(H3("Initialisation matters"))
    f += code(S10, "kmeans_details"); f += output(S10, "kmeans_details")
    f += fig("fig30_kmeans_map", "**Figure 30.** The eight k-means regions of California (black crosses = centroids).", 0.5)
    f.append(P("With one random initialisation the final inertia varied by **89%** between seeds (some runs were stuck in poor optima); **k-means++** reduced the spread to 6.7%, and 10 restarts gave an identical best result for every seed: the reason `n_init=10` is the right default. "
               "The eight regions are geographic markets with very different average prices: from ~$86k to ~$263k, which is a business-ready deliverable by itself."))
    f.append(H3("Clusters as features: does it help?"))
    f += code(S10, "clusters_as_features"); f += output(S10, "clusters_as_features")
    f.append(P(f"With the *same* random forest and CV folds, adding 25 geographic cluster-similarity features lowered the RMSE from **${R['rmse_no_geo']:,.0f}** to **${R['rmse_geo']:,.0f}**, a **{R['cluster_feat_gain_pct']:.1f}% improvement** from one unsupervised step. "
               "It is the single most effective feature we engineered: location is not linear in latitude/longitude, but similarity to market centres is easy for any model to use."))
    f.append(H3("Semi-supervised learning: an honest negative result"))
    f += code(S10, "semi_supervised"); f += output(S10, "semi_supervised")
    f.append(P(f"We pretended only 20 labels were affordable and compared: 20 random labels (79%), 20 cluster representatives (79%), label propagation (79%) and 20% closest-to-centroid (76%), against 88% with 8,000 labels. **The trick barely helped.** "
               f"The diagnostic line explains why: cluster purity is only {R['semi_purity']:.2f}, barely above the 0.75 majority baseline: *price classes overlap heavily in feature space*, so the cluster's representative's label is a poor proxy for the whole cluster. "
               "Cluster-then-label shines on data where clusters equal classes (the book's handwritten digits). **Reporting a method that did not work, with the reason, is a sign of professionalism**: you now know the assumption to check first (cluster purity)."))
    f.append(H3("DBSCAN: dense urban cores"))
    f += code(S10, "dbscan"); f += output(S10, "dbscan")
    f += fig("fig31_dbscan", "**Figure 31.** DBSCAN with eps = 0.12 degrees and min_samples = 40. Grey: noise (rural/sparse districts); coloured: dense regions.", 0.5)
    f.append(P("DBSCAN, with no k, finds 25 dense regions and labels 11% of the districts as noise: the sparse rural and desert areas. The two biggest regions (6,884 and 3,564 districts) are the Los Angeles and Bay Area metros, which k-means would have cut into pieces. Ward hierarchical clustering agrees with k-means moderately (adjusted Rand index 0.73)."))
    f.append(H3("Gaussian mixtures, BIC and anomaly detection"))
    f += code(S10, "gmm_anomaly", stop_at="fig, ax"); f += output(S10, "gmm_anomaly")
    f += fig("fig32_gmm", "**Figure 32.** Left: BIC and AIC versus the number of Gaussian components. Right: the 2% lowest-density districts (red) are scattered statewide.")
    f.append(P("Five standardised (log) features describe each district: income, households, people per household, house age and price. BIC keeps decreasing up to k = 8, the edge of our grid, so **the true optimum may be higher; a careful analyst would extend the range** and watch for diminishing returns. "
               f"We flagged the **{R['n_anomalies']}** lowest-density districts. Look at the six strongest anomalies: they are mostly **data errors or special cases**: a district with 6 households and a population of 7,460 (1,243 people per household: a prison or dormitory), another with 4 households and 3 people, and 105 households with 8,733 people. "
               "These are the rows data engineers should review *before* training: the same districts that inflated the ratio features' tails. Anomaly detection is as much a data-quality tool as a modelling one."))
    f += callout("warn", "Common mistakes in unsupervised learning", ["• Not scaling features before k-means/DBSCAN. • Believing the silhouette or the elbow gives 'the' k. • Running k-means once (use `n_init`). • Treating clusters as ground truth rather than a modelling choice; validate by stability and business meaning. "
                                                                          "• Using DBSCAN with one `eps` on data with very different densities. • Fitting the clustering on test data (leakage) when it feeds a supervised model."])
    f.append(H2("9.3  Hands-on experiments"))
    f += NL(["Extend the GMM BIC search to k = 20. Where does BIC finally stop improving? Try `covariance_type='diag'` and 'tied'.", "Run `KMeans` on *standardised* 5-feature data (income, age, ratios, price). Profile each cluster (mean of every feature) and name it as a marketing persona.",
             "Vary DBSCAN's `eps` from 0.05 to 0.3. Plot the number of clusters and the noise share. How would you pick eps (hint: k-distance plot)?", "Use `IsolationForest` instead of the GMM and compare the flagged districts.",
             "Raise the label budget in the semi-supervised experiment (20 → 50 → 200). When does the cluster trick start to beat random labelling, if ever?"])
    f.append(H2("9.4  Challenge questions"))
    f += QA([
        ("Describe the k-means algorithm and its objective.", "Initialise k centroids; repeat: assign each point to the nearest centroid, then move each centroid to the mean of its points; until no change. It minimises the within-cluster sum of squared distances (inertia) and converges to a local minimum."),
        ("Why run k-means several times?", "The result depends on the initialisation; different runs can reach different local optima. Run `n_init` times (with k-means++) and keep the lowest inertia."),
        ("How can you choose the number of clusters?", "Elbow of inertia, silhouette score/diagram, BIC (for GMMs), stability across resamples, and, above all, whether the clusters are useful for the application."),
        ("When does k-means fail and what alternative would you use?", "Non-spherical, differently sized/dense or non-convex clusters, many outliers. Use a Gaussian mixture (ellipsoidal), DBSCAN (arbitrary shape, noise), or hierarchical clustering."),
        ("How does DBSCAN work and what are its two hyper-parameters?", "It labels core points (≥ min_samples neighbours within eps), connects core points that are within eps of each other into clusters, adds border points, and labels the rest noise. eps = neighbourhood radius; min_samples = density requirement."),
        ("Compare k-means with a Gaussian mixture.", "GMM is a soft, probabilistic generalisation: each cluster has its own mean *and covariance*, instances get membership probabilities, and it provides a density for anomaly detection. k-means is the special case of equal spherical covariances with hard assignments. GMM is slower and can converge to poor solutions/degenerate covariances."),
        ("BIC versus AIC?", "Both trade log-likelihood against parameter count; BIC = p ln m − 2 ln L̂ penalises complexity more (for large m), so it tends to choose simpler models than AIC = 2p − 2 ln L̂."),
        ("How can clustering improve a supervised model?", "Add cluster IDs, distances or similarities as features (our +9.7% gain), cluster-then-label for semi-supervised learning, or use cluster-specific models."),
        ("How would you detect anomalies with a GMM?", "Fit the mixture on normal data, compute log-density with `score_samples`, and flag instances below a density quantile that matches the tolerated false-positive rate."),
        ("The semi-supervised cluster-then-label trick did not help in our project. Why?", "It assumes clusters are (nearly) pure in the label. Cluster purity was 0.78, close to the 0.75 baseline, so propagated labels were often wrong. Check purity (or label agreement on a handful of points) before using the method."),
        ("Why must you scale features before k-means?", "Distances drive assignment; unscaled features with large ranges dominate it and clusters then reflect only those features.")])
    f.append(PageBreak())
    return f
