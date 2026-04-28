# Spam Detection using Random Forest with Active Learning

## Overview
This project implements a spam detection system using a Random Forest classifier combined with an Active Learning strategy. The model improves performance iteratively by selecting the most informative samples instead of relying on fully labeled data.

---

## Key Improvements
- Metrics (Accuracy, Precision, Recall) displayed as (%) percentages with 2 decimal precision**
- Cleaner and more professional console output formatting
- High-quality plots with increased DPI for better visualization
- Clear file-saving messages for improved usability and traceability

---

## Dataset
- File: `spam.csv`
- Columns used:
  - `label_num` → renamed to `label`
  - `text` → email content
- Preprocessing:
  - Removed missing values
  - Encoded labels into numerical format

---

## Methodology

### 1. Data Processing
- Cleaned dataset and prepared features

### 2. Feature Engineering
- TF-IDF Vectorization
  - English stop words removed
  - Max features: 3000

### 3. Data Split
- 80% → Training + Unlabeled Pool
- 20% → Test Set

### 4. Active Learning Strategy
- Initial labeled set: 5% of training data
- Iterative learning (`n_queries = 20`):
  - Train Random Forest model
  - Select most uncertain sample
  - Add to labeled dataset
  - Retrain and evaluate

---

## Results
- Accuracy: 94.01%
- Precision: 86.50%
- Recall: 94.00%

---

## Outputs
- `rf_al_performance.csv` → Iteration-wise performance
- `rf_al_learning_curve.csv` → Learning progression
- `rf_al_learning_curve.png` → Performance visualization

---

## Tech Stack
- Python
- pandas
- numpy
- scikit-learn
- matplotlib

---

## How to Run
1. Place `spam.csv` in the project directory
2. Run the notebook or script
3. Check generated outputs and plots

---

## Key Insight
Active Learning significantly reduces labeling effort by focusing only on uncertain samples while maintaining high model performance.

---

## Future Work
- Compare with other models (Naive Bayes, Logistic Regression, BERT)
- Hyperparameter tuning
- Advanced active learning strategies