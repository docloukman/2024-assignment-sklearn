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
    check_X_y, check_array, check_is_fitted, check_consistent_length
)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import BaseCrossValidator


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """K-nearest neighbors classifier implementation.

    Parameters
    ----------
    n_neighbors : int, default=1
        Number of neighbors to use for classification.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input samples.
    y_ : ndarray of shape (n_samples,)
        The target values.
    classes_ : ndarray of shape (n_classes,)
        The unique classes labels.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(self, n_neighbors=1):
        """Initialize the classifier.

        Parameters
        ----------
        n_neighbors : int, default=1
            Number of neighbors to use.
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y, ensure_2d=True, allow_nd=False)
        check_classification_targets(y)

        if self.n_neighbors < 1:
            raise ValueError(
                f"Expected n_neighbors > 0, got {self.n_neighbors}"
            )

        self.n_features_in_ = X.shape[1]
        self.le_ = LabelEncoder()
        self.y_ = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.X_ = X

        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for each data sample.
        """
        check_is_fitted(self, ['X_', 'y_', 'classes_'])
        X = check_array(X, ensure_2d=True, allow_nd=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        distances = pairwise_distances(X, self.X_)
        neigh_ind = np.argpartition(
            distances, min(self.n_neighbors - 1, len(self.y_) - 1), axis=1
        )[:, :self.n_neighbors]

        neigh_labels = self.y_[neigh_ind]
        y_pred = np.array([
            np.bincount(labels).argmax()
            for labels in neigh_labels
        ])

        return self.le_.inverse_transform(y_pred)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) with respect to y.
        """
        X = check_array(X, ensure_2d=True, allow_nd=False)
        check_consistent_length(X, y)
        return np.mean(self.predict(X) == y)


class MonthlySplit(BaseCrossValidator):
    """Monthly cross-validation splitter.

    Parameters
    ----------
    time_col : str, default='index'
        Column name containing datetime values. If 'index', uses the index.
    """

    def __init__(self, time_col='index'):
        """Initialize the splitter.

        Parameters
        ----------
        time_col : str, default='index'
            Column containing datetime values or 'index'.
        """
        self.time_col = time_col

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations.

        Parameters
        ----------
        X : pd.DataFrame, required
            Training data.
        y : array-like, default=None
            Always ignored, exists for compatibility.
        groups : array-like, default=None
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations.
        """
        time_data = self._get_time_data(X)
        n_months = time_data.dt.to_period('M').nunique()
        return max(0, n_months - 1)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : array-like, default=None
            Always ignored, exists for compatibility.
        groups : array-like, default=None
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            Training set indices.
        test : ndarray
            Test set indices.
        """
        time_data = self._get_time_data(X)
        months = time_data.dt.to_period('M')
        unique_months = sorted(months.unique())

        for i in range(len(unique_months) - 1):
            train_idx = np.where(months == unique_months[i])[0]
            test_idx = np.where(months == unique_months[i + 1])[0]
            yield train_idx, test_idx

    def _get_time_data(self, X):
        """Get datetime data from DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        pd.Series
            Series containing datetime values.
        """
        if self.time_col == 'index':
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("Index must be DatetimeIndex when time_col='index'")
            return pd.Series(X.index)

        if self.time_col not in X.columns:
            raise ValueError(f"Column {self.time_col} not found in X")

        time_values = X[self.time_col]
        if not pd.api.types.is_datetime64_any_dtype(time_values):
            raise ValueError(f"Column {self.time_col} must be datetime type")

        return time_values
