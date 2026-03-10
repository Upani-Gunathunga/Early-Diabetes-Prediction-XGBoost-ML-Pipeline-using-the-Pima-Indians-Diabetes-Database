# Early-Diabetes-Prediction-XGBoost-ML-Pipeline-using-the-Pima-Indians-Diabetes-Database
This project predicts early-onset diabetes using the **Pima Indians Diabetes  Database** (768 patients, 8 clinical features). 

# 🩺 Early Diabetes Prediction — XGBoost ML Pipeline

> A hands-on machine learning project built while completing Andrew Ng's 
> Machine Learning Specialization (Courses 1 & 2), applied to a real 
> clinical dataset with medical reasoning driving every design decision.

---

## 📌 Project Overview

This project predicts early-onset diabetes using the **Pima Indians Diabetes 
Database** (768 patients, 8 clinical features). The pipeline goes far beyond 
a standard "fit and evaluate" workflow — every modelling decision maps to a 
specific concept from Andrew Ng's courses, and clinical reasoning guides 
feature engineering choices throughout.

Three parallel pipelines are implemented and compared:
- **Original** — full feature set including Age
- **Continuation A** — Age-excluded (hard removal)
- **Continuation B** — Age soft-penalised (variance shrinkage)

---

## 🧠 Course Concepts Applied

| Concept | Where Applied |
|---|---|
| Feature engineering (log transforms, interactions, ratios) | Step 4 |
| Forward feature selection via cross-validated AUC | Step 6 |
| Stratified train/test split | Step 7 |
| Data leakage prevention (scaler fit on train only) | Step 8 |
| Entropy & information gain (manual implementation) | Step 9 |
| Decision trees with entropy splits | Step 9 |
| Ensemble learning & boosting | Step 10 |
| L1/L2 regularisation inside XGBoost | Step 10 |
| Precision/recall tradeoff & threshold selection | Step 11 |
| ROC analysis & AUC | Step 11 |
| Class imbalance correction (scale_pos_weight) | Step 10 |
| Bias-variance tradeoff (boosting curve) | Step 10 |

---

## 🗂️ Pipeline Structure
```
Step 1   Data loading (direct URL, no local files needed)
Step 2   Raw feature visualisation — scatter plots before any preprocessing
Step 3   Preprocessing — KNN imputation + IQR outlier removal
Step 4   Feature engineering — log transforms, interactions, ratios
Step 5   Engineered feature visualisation
Step 6   Forward feature selection by cross-validated AUC
Step 7   Train/test split — stratified 80-20
Step 8   Feature scaling — StandardScaler fit on train only
Step 9   Single decision tree — entropy splits, manual IG calculation
Step 10  XGBoost with RandomizedSearchCV hyperparameter tuning
Step 11  Threshold optimisation for medical screening context
Step 12  Evaluation — confusion matrix, ROC, classification report
Step 13  Summary and course knowledge map
Step 14  False positive error analysis (Welch's t-test per feature)

Continuation A   Age-excluded pipeline
Continuation B   Age soft-penalised pipeline (variance shrink factor)
```

---

## ⚙️ Feature Engineering Decisions

The core idea from Course 1: **better features beat better algorithms**.

### 🔢 Log Transforms
```python
df_eng["Glucose_log"] = np.log1p(df_eng["Glucose"])
```
Glucose, BMI, Insulin, and DPF are all right-skewed. `log1p` compresses 
extreme values, making distributions more symmetric. The `+1` prevents 
`log(0)` errors. Linear and additive models fit symmetric distributions 
significantly better.

### 🔗 Interaction Terms
```python
df_eng["BMI_x_Age"]     = df_eng["BMI_log"] * df_eng["Age"]
df_eng["Glucose_x_BMI"] = df_eng["Glucose_log"] * df_eng["BMI_log"]
```
A 50-year-old with BMI 35 carries more metabolic risk than a 25-year-old 
with the same BMI. Multiplying creates a single feature that captures the 
**joint** effect — neither feature alone encodes this interaction.

### ➗ Ratio Feature
```python
df_eng["Pregnancies_div_Age"] = df_eng["Pregnancies"] / (df_eng["Age"] + 1e-6)
```
Three pregnancies at age 22 is clinically different from three pregnancies 
at 40. This ratio normalises parity by reproductive lifespan. The `1e-6` 
is defensive coding against division by zero.

---

## 🔍 Forward Feature Selection
```python
MIN_IMPROVEMENT = 0.003   # feature must earn its place
```
A greedy forward search using logistic regression as a proxy model 
(fast, interpretable linear separability measure). Features are added 
one at a time; a candidate is kept only if it improves 5-fold 
cross-validated AUC by ≥ 0.3pp. This directly prevents overfitting 
by variance inflation — the Course 1 regularisation philosophy 
applied at the feature level.

---

## 🌲 Decision Tree — Course 2 Lab 2 Connection

Entropy and information gain are implemented **manually** to mirror the 
Course 2 lab exercises exactly:
```python
def manual_entropy(y_arr):
    p1 = y_arr.mean()
    return -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)

def manual_information_gain(X_arr, y_arr, feature_idx):
    parent_H  = manual_entropy(y_arr)
    left_mask = X_arr[:, feature_idx] > 0
    ...
    return parent_H - (len(y_left)/n * manual_entropy(y_left)
                     + len(y_right)/n * manual_entropy(y_right))
```
The sklearn tree applies this same logic recursively. The manual 
implementation demonstrates understanding of what happens inside 
the black box.

---

## 🚀 XGBoost — Sequential Boosting
```python
spw = neg_count / pos_count   # 65/35 class imbalance correction
XGBClassifier(scale_pos_weight=spw, reg_alpha=..., reg_lambda=...)
```
Each tree is trained on the **residuals** of all previous trees — 
the core boosting idea from Course 2. The learning rate shrinks each 
tree's contribution (analogous to step size in gradient descent). 
`reg_alpha` (L1) and `reg_lambda` (L2) are the Course 1 regularisation 
terms applied inside the ensemble.

Hyperparameters are tuned via `RandomizedSearchCV` (40 combinations × 
5-fold CV) on the training set only — the test set is never seen during 
tuning.

---

## 🎯 Threshold Optimisation

The default 0.5 threshold is clinically wrong for screening:

| Error Type | Clinical Consequence | Relative Cost |
|---|---|---|
| False Negative (missed diabetic) | Delayed treatment, complications | **High** |
| False Positive (healthy flagged) | One extra blood test | Low |
```python
score = f1 - 0.002 * fp   # maximise F1 with small FP penalty
```
The threshold is swept across 181 values from 0.05 to 0.95. This 
sweep is the same operation as plotting the ROC curve — TPR vs FPR 
as the threshold varies.

---

## 🔬 False Positive Error Analysis
```python
stats.ttest_ind(FP[feat], TN[feat], equal_var=False)
```
Welch's t-test compares each feature's mean between false positive 
patients (healthy, but flagged as diabetic) and true negatives. 
Features with large mean differences and p < 0.05 are systematically 
elevated in misclassified patients — they are driving the false alarms.

This analysis revealed **Age** as the primary FP driver, motivating 
the two continuation experiments.

---

## 🔄 Three-Pipeline Comparison

### Continuation A — Age Hard Removal
All Age-derived features (`Age`, `BMI_x_Age`, `Pregnancies_div_Age`) 
are excluded. Tests whether removing the FP driver reduces false 
alarms without unacceptably increasing missed diagnoses.

### Continuation B — Age Soft Penalisation
```python
SHRINK_FACTOR = 10

def shrink_age_cols(X_scaled_arr, feature_list, factor, age_cols):
    X_out = X_scaled_arr.copy().astype(float)
    for col in age_cols:
        idx = feature_list.index(col)
        X_out[:, idx] /= factor   # std: 1.0 → 0.1
    return X_out
```
After `StandardScaler`, every feature has std=1. Dividing Age columns 
by 10 gives them std=0.1. XGBoost chooses splits by gain — compressed 
variance produces smaller gains, so the model uses Age less without 
discarding its real predictive signal entirely.

This is the **regularisation analogy**: L2 shrinks weights toward zero 
without zeroing them. Continuation B applies this same philosophy to 
feature influence rather than model parameters. The extended 
`reg_alpha` range compounds the suppression effect.

**How to read the results:** the preferred pipeline minimises false 
positives without a meaningful increase in false negatives, and retains 
AUC within 0.01 of the original. To experiment, adjust `SHRINK_FACTOR` 
(5 = milder, 20 = stronger suppression) and re-run Steps B-6 through 
B-14c.

---

## 📊 Results Summary

| Metric | Original | Age-Free | Soft-Pen /10 |
|---|---|---|---|
| AUC | — | — | — |
| Recall | — | — | — |
| Precision | — | — | — |
| F1 Score | — | — | — |
| False Positives | — | — | — |
| False Negatives | — | — | — |

> ℹ️ Run the pipeline to populate this table — results vary slightly 
> with environment. The relative ordering between pipelines is stable.

---

## 🖼️ Generated Plots

| File | Description |
|---|---|
| `raw_feature_scatter_plots.png` | Pre-processing feature distributions |
| `corr_heatmap.png` | Feature correlation matrix after cleaning |
| `engineered_feature_scatter_plots.png` | Post-engineering distributions |
| `decision_tree_structure.png` | Full tree visualisation (entropy splits) |
| `xgb_learning_curve.png` | Train vs test AUC per boosting round |
| `xgb_feature_importance.png` | Gain-based feature importances |
| `xgb_prob_distribution.png` | Predicted probability distributions |
| `roc_comparison.png` | XGBoost vs Decision Tree ROC |
| `threshold_analysis.png` | Precision/recall/F1 sweep |
| `xgb_confusion_matrix.png` | Final confusion matrix with annotations |
| `fp_feature_diff.png` | False positive feature analysis |
| `af_roc_comparison.png` | Age-free vs original ROC |
| `sp_roc_3way_comparison.png` | Three-pipeline ROC overlay |

---

## 🛠️ Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost scipy
```

## ▶️ Run
```bash
python diabetes_prediction_pipeline.py
```
No dataset download needed — the script loads directly from a public URL.

---

## 📚 References

- Pima Indians Diabetes Database — Smith et al., 1988
- Andrew Ng, Machine Learning Specialization — Coursera (Courses 1 & 2)
- XGBoost: Chen & Guestrin, 2016

---

## 👤 Author

**Upani Gunathunga**  
Undergraduate | Embedded Systems & Machine Learning  
[GitHub](https://github.com/Upani-Gunathunga)
