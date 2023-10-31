from sklearn.base import BaseEstimator, TransformerMixin  # type:ignore
import polars as pl
import numpy as np
from typing import Union, Iterable
from collections import OrderedDict


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


class PolarsColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for performing a specified sequence of transformations
    on Polars DataFrames.

    This transformer applies a series of transformations, each associated
    with a specific column, to the input Polars DataFrame.
    The transformations are specified as a list of 'Step' objects
    and can include any Polars or custom transformations.

    Parameters:
    -----------
    steps : Iterable[Step]
        List of 'Step' objects, each defining a transformation to
        apply to a specific column.
    step_params : dict, optional
        Dictionary of parameters for each step.

    Attributes:
    -----------
    steps : OrderedDict
        A dictionary containing the 'Step' objects and their
        associated transformations.
    step_params : dict
        Dictionary of parameters for each step.

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer to the input data.

    transform(X)
        Transform the input Polars DataFrame using the specified
        sequence of transformations.

    Returns:
    --------
    X : pl.DataFrame
        Transformed Polars DataFrame.
    """

    class Step:
        def __init__(self, name, transformer, col) -> None:
            """
            Initialize a transformation step.

            Parameters:
            -----------
            name : str
                Name of the step.
            transformer
                Transformer to apply to the specified column.
            col : str
                Name of the column to apply the transformation to.

            Returns:
            --------
            None
            """
            self.transformer = transformer
            self.col = col
            self.name = name

        def fit(self, X, y=None):
            """
            Fit the transformer in the step to the input data.

            Parameters:
            -----------
            X : pl.Series or pl.DataFrame
                Input data.
            y : None
                Ignored. It is not used in the fitting process.

            Returns:
            --------
            self
            """
            self.transformer.fit(X, y)
            return self

        def transform(self, X):
            """
            Transform the input data using the transformer in the step.

            Parameters:
            -----------
            X : pl.Series or pl.DataFrame
                Input data.

            Returns:
            --------
            Transformed data.
            """
            return self.transformer.transform(X)

    def __init__(self, steps: Iterable[Step], step_params={}):
        """
        Initialize the PolarsColumnTransformer.

        Parameters:
        -----------
        steps : Iterable[Step]
            List of transformation steps to be applied.
        step_params : dict, optional
            Dictionary of parameters for each step.

        Returns:
        --------
        None
        """
        self.steps = OrderedDict()
        for step in steps:
            self.steps[step.name] = step
        self.step_params = step_params

    def fit(self, X: Union[pl.Series, pl.DataFrame], y=None):
        """
        Fit the PolarsColumnTransformer to the input data.

        Parameters:
        -----------
        X : pl.Series or pl.DataFrame
            Input data.
        y : None
            Ignored. It is not used in the fitting process.

        Returns:
        --------
        self
        """
        if self.step_params:
            for id, params in self.step_params.items():
                self.steps[id].transformer.set_params(**params)

        for step in self.steps.values():
            step.fit(X[step.col], y)
        return self

    def transform(self, X: Union[pl.Series, pl.DataFrame], y=None):
        """
        Transform the input data using the specified steps.

        Parameters:
        -----------
        X : pl.Series or pl.DataFrame
            Input data.
        y : None
            Ignored. It is not used in the transformation process.

        Returns:
        --------
        X : pl.DataFrame
            Transformed Polars DataFrame.
        """
        for step in self.steps.values():
            transformed_col = step.transform(X[step.col])
            if len(transformed_col.shape) == 1:
                if isinstance(transformed_col, np.ndarray):
                    transformed_col = pl.Series(name=step.col, values=transformed_col)
                elif isinstance(transformed_col, pl.DataFrame):
                    transformed_col = transformed_col[step.col]
                X = X.with_columns(transformed_col.alias(step.col))
            else:
                if not isinstance(transformed_col, pl.DataFrame):
                    transformed_col = pl.DataFrame(transformed_col)

                X = pl.concat(
                    [X.drop(columns=step.col), transformed_col], how="horizontal"
                )

        if len(X.shape) == 1:
            X = X.values.reshape(-1, 1)
        return X


class PolarsOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encoder for Polars DataFrames.

    This encoder converts categorical columns into one-hot encoded columns.
    The resulting DataFrame has binary columns for each category, indicating
    the presence or absence of the category.

    Parameters:
    -----------
    drop : bool, default=False
        Whether to drop one of the binary columns to avoid multicollinearity.
        If True, one binary column for each category is dropped.

    Attributes:
    -----------
    categories : list
        List of unique categories found in the fitted data.
    cats_not_in_transform : list
        List of categories not found in the transformed data (if any).
    drop : bool
        Whether to drop one binary column for each category.

    Methods:
    --------
    fit(X, y=None)
        Fit the encoder to the input data.

    transform(X, y=None)
        Transform the input data into one-hot encoded format.

    Returns:
    --------
    X : pl.DataFrame
        Transformed Polars DataFrame with one-hot encoded columns.
    """

    def __init__(self, drop: bool = False) -> None:
        """
        Initialize the PolarsOneHotEncoder.

        Parameters:
        -----------
        drop : bool, default=False
            Whether to drop one of the binary columns to avoid
            multicollinearity.

        Returns:
        --------
        None
        """
        self.categories: list
        self.cats_not_in_transform: list
        self.drop = drop

    def fit(self, X: Union[pl.Series, pl.DataFrame], y=None):
        """
        Fit the one-hot encoder to the input data.

        Parameters:
        -----------
        X : pl.Series or pl.DataFrame
            Input data.
        y : None
            Ignored. It is not used in the fitting process.

        Returns:
        --------
        self
        """
        self.categories = X.unique().to_list()
        return self

    def transform(self, X: pl.Series, y=None):
        """
        Transform the input data into one-hot encoded format.

        Parameters:
        -----------
        X : pl.Series
            Input data to be one-hot encoded.
        y : None
            Ignored. It is not used in the transformation process.

        Returns:
        --------
        X : pl.DataFrame
            Transformed Polars DataFrame with one-hot encoded columns.
        """
        name = X.name
        self.cats_not_in_transform = [
            i for i in self.categories if i not in X.unique().to_list()
        ]
        X = X.to_dummies()
        for col in self.cats_not_in_transform:
            X = X.with_columns(
                pl.zeros(len(X), pl.Int8, eager=True).alias((f"{name}_{col}"))
            )
        X = X.select(sorted(X.columns))
        if self.drop:
            X = X[:, ::2]
        return X


class FeatureRemover(BaseEstimator, TransformerMixin):
    """
    Transformer for removing specified features from a Polars DataFrame.

    This transformer removes the specified columns (features)
    from a Polars DataFrame.

    Parameters:
    -----------
    feats_to_drop : Iterable of str, optional (default=[])
        List of column names to be removed from the DataFrame.

    Attributes:
    -----------
    feats_to_drop : Iterable of str
        List of column names to be removed from the DataFrame.

    Methods:
    --------
    fit(X, y)
        Fit the feature remover to the input data.

    transform(X, y=None)
        Transform the input data by removing specified features.

    Returns:
    --------
    X : pl.DataFrame
        Transformed Polars DataFrame with specified features removed.
    """

    def __init__(self, feats_to_drop: Iterable[str] = []) -> None:
        """
        Initialize the FeatureRemover.

        Parameters:
        -----------
        feats_to_drop : Iterable of str, optional (default=[])
            List of column names to be removed from the DataFrame.

        Returns:
        --------
        None
        """
        self.feats_to_drop = feats_to_drop

    def fit(self, X: pl.DataFrame, y: pl.Series):
        """
        Fit the feature remover to the input data.

        Parameters:
        -----------
        X : pl.DataFrame
            Input data (DataFrame).
        y : pl.Series
            Target data (Series).

        Returns:
        --------
        self : FeatureRemover
            The fitted feature remover instance.
        """
        return self

    def transform(self, X: pl.DataFrame, y=None):
        """
        Transform the input data by removing specified features.

        Parameters:
        -----------
        X : pl.DataFrame
            Input data (DataFrame) to have specified features removed.
        y : None
            Ignored. It is not used in the transformation process.

        Returns:
        --------
        X : pl.DataFrame
            Transformed Polars DataFrame with specified features removed.
        """
        X = X.drop(columns=self.feats_to_drop)
        return X
