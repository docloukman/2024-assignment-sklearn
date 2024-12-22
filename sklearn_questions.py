"""
Assignment - making a sklearn estimator and CV splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples correctly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `check_*` functions imported in the file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.

Detailed instructions for question 2:
The data to split should contain the index or one column in
datetime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict on the following. For example if you have data distributed from
November 2020 to March 2021, you have have 4 splits. The first split
will allow to learn on November data and predict on December data, the
second split to learn December and predict on January etc.

We also ask you to respect the PEP8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.validation import (
    check_X_y, check_array, check_is_fitted, _check_sample_weight,
    _num_samples
)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import BaseCrossValidator


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_min_samples=1)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")
        
        distances = pairwise_distances(X, self.X_)
        nearest_neighbors = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        
        predictions = []
        for neighbors in nearest_neighbors:
            neighbor_labels = self.y_[neighbors]
            most_common = np.bincount(neighbor_labels).argmax()
            predictions.append(most_common)
        
        return np.array(predictions)

class MonthlySplit(BaseCrossValidator):
    def __init__(self, time_col='index'):
        self.time_col = time_col

    def split(self, X, y=None, groups=None):
        times = self._get_time_data(X)
        months = times.dt.to_period('M')
        unique_months = sorted(months.unique())

        for i in range(len(unique_months) - 1):
            train_mask = months == unique_months[i]
            test_mask = months == unique_months[i + 1]
            yield np.where(train_mask)[0], np.where(test_mask)[0]

    def get_n_splits(self, X=None, y=None, groups=None):
        if X is None:
            raise ValueError("X cannot be None")
        times = self._get_time_data(X)
        return times.dt.to_period('M').nunique() - 1

    def _get_time_data(self, X):
        if self.time_col == 'index':
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("Index must be DatetimeIndex when time_col='index'")
            return pd.Series(X.index)
        
        if self.time_col not in X.columns:
            raise ValueError(f"Column {self.time_col} not found")
            
        time_values = X[self.time_col]
        if not pd.api.types.is_datetime64_any_dtype(time_values):
            raise ValueError(f"Column {self.time_col} must be datetime type")
            
        return time_values
