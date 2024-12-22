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


"""Implementation of KNN classifier and monthly split cross-validator."""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):
        """Initialize the classifier with the number of neighbors."""
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        distances = pairwise_distances(X, self.X_)
        k_nearest = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        y_pred = np.array([np.bincount(self.y_[neighbors]).argmax() for neighbors in k_nearest])
        return y_pred

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.
        y : ndarray of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        return np.mean(self.predict(X) == y)


class MonthlySplit(BaseCrossValidator):
    """
    Cross-validator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, default='index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_col` to `'index'`.
    """

    def __init__(self, time_col='index'):
        """Initialize the cross-validator with the time column."""
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """
        Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        dates = X.index if self.time_col == 'index' else X[self.time_col]
        dates = pd.to_datetime(dates)
        return len(pd.unique(dates.to_period('M'))) - 1

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """
        if self.time_col == 'index':
            dates = X.index
        else:
            if not pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
                raise ValueError('time_col must be of type datetime')
            dates = X[self.time_col]

        dates = pd.to_datetime(dates)
        periods = dates.to_period('M')
        unique_periods = periods.unique()

        for i in range(len(unique_periods) - 1):
            train_period = unique_periods[i]
            test_period = unique_periods[i + 1]

            train_mask = periods == train_period
            test_mask = periods == test_period

            yield (
                np.where(train_mask)[0],
                np.where(test_mask)[0]
            )
