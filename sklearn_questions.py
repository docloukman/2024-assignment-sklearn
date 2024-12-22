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
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import BaseCrossValidator


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """Classifier implementing the k-nearest neighbors algorithm.

    This classifier predicts the target of a test point based on the target
    of its nearest neighbor in the training set, using Euclidean distance.

    Parameters
    ----------
    n_neighbors : int, default=1
        Number of neighbors to consider for prediction.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input samples.
    y_ : ndarray of shape (n_samples,)
        The target values.
    classes_ : ndarray of shape (n_classes,)
        The unique classes labels.
    """

    def __init__(self, n_neighbors=1):
        """Initialize the KNearestNeighbors classifier.

        Parameters
        ----------
        n_neighbors : int, default=1
            Number of neighbors to use for prediction.
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the k-nearest neighbors classifier.

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
        # Input validation using sklearn's check functions
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Store number of features for predict step validation
        self.n_features_in_ = X.shape[1]

        # Encode class labels
        self.le_ = LabelEncoder()
        y = self.le_.fit_transform(y)

        self.X_ = X
        self.y_ = y
        self.classes_ = self.le_.classes_

        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but KNearestNeighbors "
                f"was trained with {self.n_features_in_} features."
            )

        # Compute distances between test points and training points
        distances = pairwise_distances(X, self.X_)

        # Find indices of k nearest neighbors
        k_neighbors = np.argsort(distances, axis=1)[:, :self.n_neighbors]

        # Get labels of k nearest neighbors
        k_neighbors_labels = self.y_[k_neighbors]

        # Predict by majority voting
        y_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=1,
            arr=k_neighbors_labels
        )

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
            Mean accuracy of predictions.
        """
        return np.mean(self.predict(X) == y)


class MonthlySplit(BaseCrossValidator):
    """Monthly cross-validation splitter.

    Provides train/test indices to split time series data between successive
    months. For each split, test indices must be higher than before, and thus
    shuffling in cross validator is inappropriate.

    Parameters
    ----------
    time_col : str, default='index'
        Column name containing datetime values. If 'index', the index is used.

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range('2020-01-01', '2020-03-31', freq='D')
    >>> X = pd.DataFrame({'val': range(len(dates))}, index=dates)
    >>> cv = MonthlySplit()
    >>> for train_idx, test_idx in cv.split(X):
    ...     print(f"TRAIN:", X.index[train_idx].min(), X.index[train_idx].max())
    ...     print(f"TEST:", X.index[test_idx].min(), X.index[test_idx].max())
    """

    def __init__(self, time_col='index'):
        """Initialize the monthly splitter.

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
        X : pd.DataFrame
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
        unique_months = time_data.dt.to_period('M').unique()
        return max(len(unique_months) - 1, 0)

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
            train_mask = months == unique_months[i]
            test_mask = months == unique_months[i + 1]
            yield np.where(train_mask)[0], np.where(test_mask)[0]

    def _get_time_data(self, X):
        """Extract datetime data from DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        pd.Series
            Series containing datetime values.

        Raises
        ------
        ValueError
            If datetime column is not found or is invalid.
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
