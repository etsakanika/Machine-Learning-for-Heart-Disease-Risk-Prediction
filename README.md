# Heart Disease Risk Prediction and Analysis: Supervised vs. Unsupervised Approaches

A two-part machine learning study on the UCI Hungarian Heart Disease dataset, comparing what a **supervised** classifier and an **unsupervised** clustering algorithm extract from the same data.

---

## Random Forest - DBSCAN

| Question | Method | Result |
|---|---|---|
| Can we **predict** heart disease from clinical measurements? | Random Forest (3 `max_features` configurations + GridSearchCV tuning) | **Yes** — test F1 ≈ 0.80–0.85, AUC > 0.85 |
| Does the data **naturally cluster** by disease status without supervision? | DBSCAN (+ sensitivity grid) | **No** — best ARI ≈ 0.02, robust across (eps, min_samples) |

The two findings together tell a more interesting story than either alone: heart disease in this cohort is a real, learnable signal, but it is **encoded distributively** across features rather than expressed as a separable region of feature space.

---

## The Dataset

- **Source:** UCI Machine Learning Repository — Heart Disease dataset, Hungarian subset (collected at the Hungarian Institute of Cardiology, Budapest)
- **Samples:** 295 patients
- **Features:** 13 clinical measurements (age, sex, chest pain type, resting blood pressure, cholesterol, max heart rate, etc.)
- **Target:** `num` — heart disease severity (0 = none, 1–4 = present); binarised to 0/1 for both tasks
- **Notable challenge:** Substantial missingness in `slope`, `ca`, `thal` (>40%), informing the preprocessing strategy

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── reprocessed.hungarian.data         # Download from UCI (see below)
└── notebooks/
    ├── 01_random_forest_classification.ipynb
    └── 02_dbscan_clustering.ipynb
```

---

## How to Run

```bash
# 1. Clone
git clone https://github.com/<your-username>/heart-disease-ml.git
cd heart-disease-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
#    From https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
#    Place `reprocessed.hungarian.data` inside the `data/` folder.

# 4. Run the notebooks
jupyter lab notebooks/
```

Tested with Python 3.10+.

---

## Task 1 — Random Forest Classification

**Notebook:** [`01_random_forest_classification.ipynb`](notebooks/01_random_forest_classification.ipynb)

### What it does

Investigates how the `max_features` hyperparameter shapes the bias-variance trade-off in Random Forest classifiers, by comparing three configurations under identical conditions: `log2`, `sqrt` (default), and `None`. Then validates the manual findings against a systematic hyperparameter search.

### Methodology highlights

- **Bivariate EDA** — violin plots (numerical features vs. target) and grouped bar charts with disease prevalence per category (categorical features vs. target). This sanity-checks which features should later show high importance in the trained model.
- **Leakage-safe preprocessing** via `sklearn.pipeline.Pipeline` — `SimpleImputer` is fit on training folds only during cross-validation, and on `X_train` only for the final fit. This rules out the most common silent leakage in academic notebooks.
- **5-fold stratified cross-validation** as the primary evaluation, with a held-out 20% test set for final reporting.
- **Class imbalance handled via `class_weight='balanced'`** rather than resampling — justified in the notebook (small sample, mild imbalance, sparse 10D feature space).
- **Overfit gap analysis** (train vs. test accuracy) as a comparison axis between configurations.
- **Hyperparameter optimisation via `GridSearchCV`** over `max_features`, `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` (324 combinations × 5 folds = 1,620 fits), with side-by-side comparison against the sklearn-default RF2 baseline.

### Key results

- `max_features='sqrt'` (sklearn default) gives the best balance: competitive CV F1, smallest train-test gap.
- **GridSearchCV confirms the manual baseline is hard to beat substantially** — improvements over RF2 are typically small on this dataset size. A useful empirical demonstration that *Random Forest defaults are strong* and tuning has diminishing returns on small data.
- Top predictors are consistent across all three models: chest pain type (`cp`), max heart rate (`thalach`), and exercise-induced angina (`exang`) — agreeing with established cardiology literature.
- Feature importance ranking is stable across configurations, indicating a robust interpretation.

---

## Task 2 — DBSCAN Clustering

**Notebook:** [`02_dbscan_clustering.ipynb`](notebooks/02_dbscan_clustering.ipynb)

### What it does

Asks whether the same dataset exhibits **natural cluster structure** that aligns with the disease label — without using the label during clustering. The label is held aside for *post-hoc* evaluation via ARI and NMI.

### Methodology highlights

- **Feature scaling with `StandardScaler`** — essential for distance-based DBSCAN (in contrast to RF in Task 1, which is scale-invariant).
- **k-distance graph** to select `eps` candidates.
- **(eps × min_samples) sensitivity grid** — confirms results are robust to parameter choice rather than artefacts of one specific combination.
- **Honest external evaluation:** samples with missing target values are excluded from ARI/NMI (not imputed — imputing a target value would fabricate ground truth).

### Key results

- ARI ≈ 0.02 across the entire (eps × min_samples) grid → the disease label does not correspond to density-based clusters in this feature space.
- DBSCAN ends up doing **outlier detection rather than disease detection** — the dense cluster captures the "typical patient", noise points are patients with extreme feature values.
- This is a substantive finding, not a methodological failure: it complements Task 1 by showing that the label is *multivariate and subtle*, not geometrically expressed.

---

## Project Insights

Beyond the headline results, the deliberate choices in this project — that I would highlight in interviews or with academic supervisors — are:

1. **Treating leakage as a first-class concern.** Median imputation before the train-test split is a common silent bug in academic notebooks; using `Pipeline` makes leakage prevention part of the model object itself, and the philosophy is documented explicitly above the preprocessing code so reviewers don't need to reverse-engineer it.
2. **Manual experiment first, then GridSearchCV to validate.** Rather than running a hyperparameter search blindly as step one, I first characterised the effect of a single hyperparameter (`max_features`) in isolation, *then* used `GridSearchCV` to test whether systematic tuning could improve on the manual finding. This produces a more interpretable narrative than "tuned model gives F1 = X" and demonstrates honest cost-benefit thinking about when tuning is worth the compute.
3. **Using the unsupervised result to interpret the supervised result.** DBSCAN's "failure" to recover the label is itself evidence about the geometry of the problem — and that justifies why the Random Forest needed multiple features in combination.
4. **Stating limitations honestly.** Small `n`, dropped clinically-meaningful features, single random seed for the split — these are noted, not papered over.
5. **Connecting findings to the domain.** The feature importance ranking is sanity-checked against cardiology literature, not just reported as numbers.

---

## Tech Stack

- **Python 3.10+** | `numpy`, `pandas`, `matplotlib`, `seaborn`
- **scikit-learn** — `Pipeline`, `SimpleImputer`, `StratifiedKFold`, `cross_val_score`, `GridSearchCV`, `RandomForestClassifier`, `DBSCAN`, `StandardScaler`, `PCA`
- Single-machine, CPU-only — the dataset is small enough that GPU is unnecessary.
- Reproducibility ensured via `random_state=42` applied consistently across data splits, CV folds, and model initialisation.

---

## License

Code: MIT.
Dataset: [UCI Machine Learning Repository terms](https://archive.ics.uci.edu/ml/citation_policy.html) — please cite the original source if you redistribute.
