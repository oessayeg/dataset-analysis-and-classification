# Data Science x Logistic Regression

A machine learning project that classifies Hogwarts students into their houses using logistic regression.

The project covers the following topics:

- **Descriptive statistics** — hand-rolled computation of count, mean, std, min, max, quartiles, variance, IQR and outliers without any library helpers
- **Data visualization** — histograms, scatter plots, and pair plots for data exploration and identifying similar or redundant features
- **Logistic regression** — one-vs-all classifier trained on student course scores to predict one of four Hogwarts houses
- **Gradient descent** — batch gradient descent, stochastic gradient descent (SGD), and mini-batch gradient descent

## Usage

```bash
# Descriptive statistics
python describe.py dataset_train.csv

# Data visualization
python histogram.py
python scatter_plot.py
python pair_plot.py

# Train the model
python logreg_train.py dataset_train.csv [--optimizer batch|sgd|minibatch] [--epochs N]

# Predict
python logreg_predict.py dataset_test.csv [weights.json]
```
