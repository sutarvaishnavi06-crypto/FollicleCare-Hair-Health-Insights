# FollicleCare — Hair & Scalp Health Insights

## 1. Overview
FollicleCare is an interactive application for organizing common hair and scalp observations and demonstrating probability calibration. It includes a guided check-in, an optional image preview workspace and a model evaluation laboratory.

## 2. Objective
To demonstrate the difference between raw model confidence scores and calibrated probabilities using Logistic Regression, Random Forest, sigmoid calibration and isotonic calibration.

## 3. Methodology
A reproducible simulated binary dataset is generated with scikit-learn and divided into stratified training and test partitions. Logistic Regression and Random Forest are fitted on training data. Sigmoid and isotonic calibration are fitted using cross-validation on training data. All model variants are evaluated on the held-out test data.

## 4. Evaluation
- ROC-AUC: ranking ability.
- Brier score: squared error of probability estimates.
- Expected Calibration Error (ECE): weighted gap between confidence and observed event frequency across bins.
- Reliability diagram: visual comparison of predicted probability and observed frequency.
- Accuracy: threshold-based classification performance.

## 5. Data limitation and responsible use
The model data are simulated and do not represent real patients, scalp images or disease prevalence. The check-in is an educational interaction, not a diagnostic assessment. Image upload is limited to preview and note-taking. The displayed score must not be interpreted as an individual's medical risk.

## 6. Future scope
A future research version could use an appropriately licensed and representative scalp/hair dataset with expert-verified labels, participant-aware train/test splits, external validation, subgroup analysis and dermatology oversight.

## 7. Conclusion
FollicleCare provides an interactive setting for exploring hair/scalp observations and learning how calibration changes probability estimates. It emphasizes transparency and the difference between a software demonstration and a clinically validated healthcare model.
