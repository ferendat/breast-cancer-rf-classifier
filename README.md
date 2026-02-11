# Breast Cancer Classifier — Random Forest with Auto Threshold Optimization

A Random Forest binary classifier for breast cancer detection, trained on the Wisconsin Breast Cancer Dataset. Achieves zero false negatives through automatic decision threshold optimization — prioritizing recall for malignant cases over raw accuracy.

## Features
- Random Forest classifier with 200 estimators
- Automatic threshold search: finds the highest threshold that eliminates false negatives
- ROC curve analysis with AUC scoring
- Confusion matrix and classification report
- Feature importance visualization

## Dataset
[Breast Cancer ML Ready Dataset – Kaggle (garvsachdeva0205)](https://www.kaggle.com/datasets/garvsachdeva0205/ml-ready-breast-cancer-diagnosis-dataset/data)

Derived from the Wisconsin Breast Cancer Dataset. File used: `cleaned_breast_cancer_data.csv`

- 569 samples, 30 features
- Target: `diagnosis` — 0 (Benign), 1 (Malignant)
- Balanced classes — no resampling required

## Requirements
```
scikit-learn
pandas
numpy
matplotlib
seaborn
```
All available by default in Google Colab and Kaggle notebooks.

## Usage

**Colab:** Upload `cleaned_breast_cancer_data.csv` to your session, then run the single notebook cell.

**Kaggle:** Add the dataset to your notebook and find the file path by running:
```python
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
Then update the `read_csv` path accordingly. No other configuration needed — the threshold is selected automatically.

## How It Works
1. **Train/Test Split** — 80/20 stratified split to preserve class ratios
2. **Train** — Random Forest with 200 trees
3. **Probability Scoring** — uses `predict_proba` instead of hard `predict`
4. **Auto Threshold Search** — scans all ROC thresholds, selects the highest one where false negatives = 0 and precision is maximized
5. **Final Evaluation** — confusion matrix, classification report, feature importance plot

## Results

| Metric | Value |
|---|---|
| AUC | 0.9942 |
| Auto-selected Threshold | 0.145 |
| Recall (Malignant) | 1.00 |
| Precision (Malignant) | 0.84 |
| Overall Accuracy | 93% |
| False Negatives | 0 |

## Why Threshold Optimization Matters
Default classifiers use a 0.5 decision threshold. In cancer screening, a false negative (missed malignant case) is far more dangerous than a false positive (unnecessary follow-up). By lowering the threshold automatically, the model flags every malignant case at the cost of a small number of false alarms — an acceptable trade-off in a medical context.

## Top Predictive Features
1. perimeter_worst
2. area_worst
3. concave points_worst
4. concave points_mean
5. radius_worst

These are all size and shape descriptors of the worst (largest) cell nuclei — consistent with known pathological markers for malignancy.

## Limitations
- Test set is 114 samples — solid for a portfolio project, but external validation would be needed for clinical use
- Model is not saved to disk — retraining is required each session
- Intended for educational purposes only — not for real medical diagnosis

## License
MIT
