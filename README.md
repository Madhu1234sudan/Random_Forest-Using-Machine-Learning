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

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

## Usage

1. Place `spam.csv` in the same directory as the notebook.
2. Run the notebook cells sequentially.
3. Inspect the saved CSV files and generated learning curve plot.

## Notes

- The notebook uses a random seed for reproducibility (`np.random.seed(42)`).
- The active learning strategy is uncertainty sampling based on class probability difference.