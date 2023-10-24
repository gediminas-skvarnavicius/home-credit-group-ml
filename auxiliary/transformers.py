from sklearn.base import BaseEstimator, TransformerMixin  # type:ignore
import polars as pl
import numpy as np


class TargetMeanOrderedLabeler(BaseEstimator, TransformerMixin):
    """
    Transformer for labeling categorical values based on their mean target
    values.

    This transformer labels categorical values based on their mean
    target values. The labels can be determined using different methods,
    such as 'label' (integer labels ordered by target mean), 'mean'
    (mean target values),
    or 'last_mean' (mean target values plus one for last year means).

    Parameters:
    -----------
    how : str, optional (default='mean')
        Method for determining labels. Accepted values are
        'label', 'mean', or 'last_mean'.

    Attributes:
    -----------
    map : dict
        A mapping of categorical values to their corresponding labels.
    how : str
        The method used to determine labels.

    Methods:
    --------
    fit(X, y)
        Fit the labeler to the input data.

    transform(X, y=None)
        Transform the input data by labeling categorical values.

    Returns:
    --------
    X : pl.Series
        Transformed Polars Series with labeled categorical values.
    """

    def __init__(self, how: str = "mean") -> None:
        """
        Initialize the TargetMeanOrderedLabeler.

        Parameters:
        -----------
        how : str, optional (default='mean')
            Method for determining labels. Accepted values are
            'label', 'mean', or 'last_mean'.

        Returns:
        --------
        None
        """
        self.map = {}
        self.how = how

    def fit(self, X: pl.Series, y: pl.Series):
        """
        Fit the labeler to the input data.

        Parameters:
        -----------
        X : pl.Series
            Categorical feature values.
        y : pl.Series
            Target values for the corresponding features.

        Returns:
        --------
        self : TargetMeanOrderedLabeler
            The fitted labeler instance.
        """
        self.map = {}
        self.sort_df = pl.DataFrame([X, y]).group_by(X.name).mean().sort(y.name)

        if self.how not in ["label", "mean", "last_mean"]:
            raise ValueError(
                """Invalid value for 'how' argument.
                Accepted values are 'label', 'mean', or 'last_mean'."""
            )

        if self.how == "label":
            for i, val in enumerate(self.sort_df[X.name]):
                self.map[val] = i
        if self.how == "mean":
            for mean, val in zip(self.sort_df[y.name], self.sort_df[X.name]):
                self.map[val] = mean
        if self.how == "last_mean":
            for mean, val in zip(self.sort_df[y.name], self.sort_df[X.name]):
                self.map[val + 1] = mean
                self.map[self.sort_df[X.name].min()] = None
        return self

    def transform(self, X: pl.Series, y=None):
        """
        Transform the input data by labeling categorical values.

        Parameters:
        -----------
        X : pl.Series
            Categorical feature values to be labeled.
        y : None
            Ignored. It is not used in the transformation process.

        Returns:
        --------
        X : pl.Series
            Transformed Polars Series with labeled categorical values.
        """
        X = X.map_dict(self.map)
        return X
