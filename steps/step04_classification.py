"""STEP 4 - Chapter 3: Classification. Screening tool: is a district 'premium' (top price quartile)?"""
# %% setup
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from homelens.tasks import load_task
from homelens.utils import set_style, savefig, record, PALETTE
from homelens.config import RANDOM_STATE
set_style()
T = load_task()
X, y = T.X, T.y_premium
print("Premium threshold (75th percentile of TRAIN prices): $%.0f" % T.thr)
print("Class balance: premium = %.1f%%  not premium = %.1f%%" % (100 * y.mean(), 100 * (1 - y.mean())))
print("Feature matrix:", X.shape)

# %% accuracy_trap
dummy = DummyClassifier(strategy="most_frequent")
acc = cross_val_score(dummy, X, y, cv=3, scoring="accuracy")
print("A model that ALWAYS says 'not premium' scores accuracy = %.3f" % acc.mean())
print("... yet it finds 0 premium districts. Accuracy is misleading on skewed classes.")
record("dummy_accuracy", acc.mean())

# %% sgd_confusion
sgd = SGDClassifier(random_state=RANDOM_STATE, loss="hinge", max_iter=2000, tol=1e-4)
pred = cross_val_predict(sgd, X, y, cv=5)
cm = confusion_matrix(y, pred)
print("Confusion matrix (rows = actual, cols = predicted)")
print(pd.DataFrame(cm, index=["actual not-premium", "actual premium"], columns=["pred not", "pred premium"]).to_string())
p, r, f = precision_score(y, pred), recall_score(y, pred), f1_score(y, pred)
print("\nprecision = TP/(TP+FP) = %d/%d = %.3f" % (cm[1, 1], cm[1, 1] + cm[0, 1], p))
print("recall    = TP/(TP+FN) = %d/%d = %.3f" % (cm[1, 1], cm[1, 1] + cm[1, 0], r))
print("F1        = harmonic mean         = %.3f" % f)
print("accuracy  = %.3f" % accuracy_score(y, pred))
record("sgd_precision", p); record("sgd_recall", r); record("sgd_f1", f)

# %% pr_curve
scores = cross_val_predict(sgd, X, y, cv=5, method="decision_function")
prec, rec, thr = precision_recall_curve(y, scores)
idx90 = np.argmax(prec >= 0.90)
t90 = thr[idx90]
pred90 = scores >= t90
print("Threshold needed for >=90%% precision: %.3f" % t90)
print("   -> precision = %.3f, recall = %.3f" % (precision_score(y, pred90), recall_score(y, pred90)))
print("Default threshold 0:  precision = %.3f, recall = %.3f" % (precision_score(y, scores >= 0), recall_score(y, scores >= 0)))
record("sgd_p90_recall", recall_score(y, pred90))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(thr, prec[:-1], label="precision", color=PALETTE["blue"], lw=2)
ax[0].plot(thr, rec[:-1], label="recall", color=PALETTE["orange"], lw=2)
ax[0].axvline(t90, color=PALETTE["grey"], ls="--"); ax[0].set_xlabel("decision threshold"); ax[0].legend(); ax[0].set_title("Precision/recall trade-off")
ax[1].plot(rec, prec, color=PALETTE["green"], lw=2); ax[1].plot(rec[idx90], prec[idx90], "o", color=PALETTE["red"])
ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_title("Precision-recall curve")
savefig(fig, "fig08_pr_curve")

# %% roc_compare
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=2)
proba_rf = cross_val_predict(rf, X, y, cv=5, method="predict_proba")[:, 1]
auc_sgd, auc_rf = roc_auc_score(y, scores), roc_auc_score(y, proba_rf)
print("ROC AUC  SGD (hinge/linear) = %.4f" % auc_sgd)
print("ROC AUC  Random forest      = %.4f" % auc_rf)
from sklearn.metrics import average_precision_score
print("Average precision (PR-AUC)  SGD = %.4f | RF = %.4f" % (average_precision_score(y, scores), average_precision_score(y, proba_rf)))
fig, ax = plt.subplots(figsize=(5.8, 4.8))
for name, s, c in [("SGD linear", scores, PALETTE["orange"]), ("Random forest", proba_rf, PALETTE["blue"])]:
    fpr, tpr, _ = roc_curve(y, s); ax.plot(fpr, tpr, lw=2, color=c, label=f"{name} (AUC {roc_auc_score(y, s):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate (recall)")
ax.set_title("ROC curves"); ax.legend(loc="lower right")
savefig(fig, "fig09_roc")
record("auc_sgd", auc_sgd); record("auc_rf", auc_rf)
prf = proba_rf >= 0.5
print("RF @0.5: precision = %.3f recall = %.3f" % (precision_score(y, prf), recall_score(y, prf)))
record("rf_precision", precision_score(y, prf)); record("rf_recall", recall_score(y, prf))

# %% multiclass
yt = T.y_tier
labels = ["budget", "lower-mid", "upper-mid", "premium"]
print("Tier sizes:", np.bincount(yt))
models = {
    "SGD one-vs-rest (built-in)": SGDClassifier(random_state=RANDOM_STATE, max_iter=2000),
    "SGD one-vs-one (6 binary models)": OneVsOneClassifier(SGDClassifier(random_state=RANDOM_STATE, max_iter=2000)),
    "Random forest (native multiclass)": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=2),
}
store = {}
for n, m in models.items():
    p_ = cross_val_predict(m, X, yt, cv=5); store[n] = p_
    print(f"{n:36s} accuracy = {accuracy_score(yt, p_):.3f}  macro-F1 = {f1_score(yt, p_, average='macro'):.3f}")
record("tier_acc_rf", accuracy_score(yt, store["Random forest (native multiclass)"]))
ovo = OneVsOneClassifier(SGDClassifier(random_state=RANDOM_STATE)).fit(X[:2000], yt[:2000])
print("Number of binary classifiers inside OvO for 4 classes:", len(ovo.estimators_), "= 4*3/2")
print("Number inside OvR for 4 classes:", len(OneVsRestClassifier(SGDClassifier()).fit(X[:2000], yt[:2000]).estimators_))

# %% error_analysis
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
pr = store["Random forest (native multiclass)"]
ConfusionMatrixDisplay.from_predictions(yt, pr, display_labels=labels, ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title("Counts"); ax[0].grid(False)
ConfusionMatrixDisplay.from_predictions(yt, pr, display_labels=labels, ax=ax[1], cmap="Blues",
                                        normalize="true", values_format=".2f", colorbar=False)
ax[1].set_title("Normalised by true row (=recall)"); ax[1].grid(False)
fig.suptitle("Random forest, 4 price tiers (out-of-fold)", fontweight="bold")
savefig(fig, "fig10_confusion_tiers")
cmn = confusion_matrix(yt, pr)
adjacent = sum(cmn[i, j] for i in range(4) for j in range(4) if abs(i - j) == 1)
far = sum(cmn[i, j] for i in range(4) for j in range(4) if abs(i - j) >= 2)
print("Mistakes landing in an ADJACENT tier: %d | two or more tiers away: %d" % (adjacent, far))
print("=> %.1f%% of all errors are off by only one tier" % (100 * adjacent / (adjacent + far)))
record("adjacent_err_pct", 100 * adjacent / (adjacent + far))

# %% multilabel
# Multilabel: each district gets TWO yes/no labels at once.
coastal = T.X_raw["ocean_proximity"].isin(["NEAR BAY", "NEAR OCEAN", "ISLAND"]).to_numpy().astype(int)
Y_multi = np.c_[y, coastal]
knn = KNeighborsClassifier(n_jobs=2)
pm = cross_val_predict(knn, X, Y_multi, cv=3)
print("Labels: [is_premium, is_coastal]")
print("F1 per label:", f1_score(Y_multi, pm, average=None).round(3), "| macro F1 = %.3f" % f1_score(Y_multi, pm, average="macro"))
print("Example district -> predicted", pm[0], "actual", Y_multi[0])
