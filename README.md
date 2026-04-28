<<<<<<< HEAD
# Spam Detection with Active Learning

## Project Overview

This project demonstrates a spam detection workflow using a Random Forest classifier combined with a basic active learning loop. The notebook processes a labeled email dataset, transforms text using TF-IDF, and incrementally improves the classifier by querying uncertain samples from an unlabeled pool.

## Dataset

- Input file: `spam.csv`
- Expected columns: `label_num`, `text`
- The notebook selects and renames `label_num` to `label`, drops missing values, and encodes labels as integers.

## Key Steps

1. Load and clean the dataset.
2. Vectorize email text using `TfidfVectorizer` with English stop words and up to 3000 features.
3. Split data into:
    - Training + unlabeled pool (80%)
    - Test set (20%)
4. Create an initial labeled set from the training pool (5% of `X_train_pool`).
5. Ensure the initial labeled set contains both classes.
6. Run active learning for a fixed number of queries (`n_queries = 20`):
    - Train a `RandomForestClassifier`
    - Select the most uncertain pool sample
    - Add it to the labeled set
    - Evaluate on the test set after each query
7. Save performance results and learning curve data.

## Results

- Final accuracy: `0.9401`
- Precision: `0.8650`
- Recall: `0.9400`

## Outputs

- `rf_al_performance.csv`
- `rf_al_learning_curve.csv`
- `rf_al_learning_curve.png`

## Requirements
=======
# Spam Detection using Random Forest with Active Learning

## Overview

This project implements a spam detection system using a Random Forest classifier enhanced with an Active Learning strategy. The goal is to improve model performance efficiently by selectively labeling the most informative data points rather than relying on fully labeled datasets.

The workflow includes text preprocessing, TF-IDF vectorization, iterative model training, and performance tracking through an active learning loop.

---

## Dataset

- File: `spam.csv`
- Columns used:
  - `label_num` → renamed to `label`
  - `text` → email content
- Data preprocessing:
  - Removed missing values
  - Encoded labels into numerical format

---

## Methodology

### 1. Data Preparation
- Cleaned dataset and selected relevant columns
- Converted labels into integer format

### 2. Feature Engineering
- Applied `TfidfVectorizer`
  - English stop words removed
  - Maximum features: 3000

### 3. Data Splitting
- 80% → Training pool (labeled + unlabeled)
- 20% → Test set

### 4. Initial Training Set
- Randomly selected 5% from training pool
- Ensured representation of both classes

### 5. Active Learning Loop
Repeated for `n_queries = 20`:
- Train `RandomForestClassifier`
- Identify most uncertain sample (based on prediction probability)
- Add selected sample to labeled dataset
- Retrain model
- Evaluate on test set

---

## Results

- Accuracy: **94.01%**
- Precision: **86.50%**
- Recall: **94.00%**

The model shows strong performance with limited labeled data, demonstrating the effectiveness of active learning.

---

## Outputs

- `rf_al_performance.csv` → Performance metrics per iteration
- `rf_al_learning_curve.csv` → Learning progression data
- `rf_al_learning_curve.png` → Visual representation of performance improvement

---

## Tech Stack
>>>>>>> 58d6bf57bb2e54631c4631bbfc170b20f21ed672

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

<<<<<<< HEAD
## Usage

1. Place `spam.csv` in the same directory as the notebook.
2. Run the notebook cells sequentially.
3. Inspect the saved CSV files and generated learning curve plot.

## Notes

- The notebook uses a random seed for reproducibility (`np.random.seed(42)`).
- The active learning strategy is uncertainty sampling based on class probability difference.
=======
---

## How to Run

1. Place `spam.csv` in the project directory
2. Open the notebook
3. Run all cells sequentially
4. Review generated outputs and performance plots

---

## Key Insight

Instead of labeling large datasets upfront, this approach focuses on labeling only the most uncertain samples. This reduces labeling effort while maintaining high model performance.

---

## Reproducibility

- Random seed fixed using: `np.random.seed(42)`
- Results are consistent across runs

---

## Future Improvements

- Compare with other models (SVM, Logistic Regression, BERT)
- Experiment with different active learning strategies
- Optimize hyperparameters using GridSearchCV
>>>>>>> 58d6bf57bb2e54631c4631bbfc170b20f21ed672
