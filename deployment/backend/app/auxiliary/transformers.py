from sklearn.base import BaseEstimator, TransformerMixin  # type:ignore
import polars as pl
import numpy as np
from typing import Union, Iterable, Optional, Any
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
        def __init__(
            self, name: str, transformer, col: Union[str, Iterable[str]]
        ) -> None:
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
            X = self.transformer.transform(X)
            if not isinstance(X, (pl.DataFrame, pl.Series)):
                X = pl.DataFrame(X, schema=self.col)
            return X

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


class NotInImputerPolars(BaseEstimator, TransformerMixin):
    """
    Transformer for imputing values in a Polars DataFrame by filtering out
    values not in the specified number of most frequent values
    and replacing them with the most frequent value.

    This transformer filters each specified column in the input data
    to retain only the values that are among the most frequent.
    It then replaces the remaining values with the most frequent value.

    Parameters:
    -----------
    filter : Optional[Iterable]
        Values to filter out for each column. If not provided, it will be
        computed during fitting.
    cat_no : Optional[int]
        Number of most frequent categories to consider for filtering.
        Ignored if `filter` is provided.
    fill_value : Optional[Union[int, str, float]]
        Value to fill in for filtered-out values. If not provided, it will
        be computed during fitting.
    most_frequent : bool
        If True, replace missing values with the most frequent value in
        each column.
    return_format : str
        Output format. Can be 'pl' for Polars DataFrame or 'np'
        for NumPy array.

    Attributes:
    -----------
    filter : dict
        A dictionary containing information about the most frequent values
        for each specified column.
        The dictionary structure is as follows:
        {
            'column_name': [most_frequent_value_1, most_frequent_value_2, ...],
            ...
        }

    Methods:
    --------
    fit(X)
        Fit the imputer to the specified columns in the input data.

    transform(X)
        Transform the input data by imputing values based on the most frequent
        values.

    Returns:
    --------
    X : pl.DataFrame or np.ndarray
        Transformed data with filled values in the specified format.
    """

    def __init__(
        self,
        filter: Optional[Iterable] = None,
        cat_no: Optional[int] = None,
        min_values: int = None,
        fill_value: Optional[Union[int, str, float]] = None,
        most_frequent: bool = False,
        return_format: str = "pl",
    ):
        """
        Initialize the NotInImputer.

        Parameters:
            filter (Iterable, optional): Values to filter out for each column.
            If not provided, it will be computed during fitting.
            cat_no (int, optional): Number of most frequent categories to
            consider for filtering. Ignored if `filter` is provided.
            fill_value (int, str, float, optional): Value to fill in for
            filtered-out values. If not provided,
            it will be computed during fitting.
        """
        if filter is None and cat_no is None and min_values is None:
            raise ValueError(
                "Either 'filter', 'min_values' or 'cat_no' must be defined."
            )
        if cat_no is not None and min_values is not None:
            raise ValueError("Can not use both cat_no and min_values together")
        self.fill_value = fill_value
        self.filter = filter
        self.cat_no = cat_no
        self.most_frequent = most_frequent
        self.min_values = min_values
        self.return_format = return_format

    def fit(self, X: Union[pl.Series, pl.DataFrame], y=None):
        """
        Fit the NotInImputer to the input data.

        Parameters:
            X (pl.Series or pl.DataFrame): Input data.

        Returns:
            self
        """
        if len(X.shape) == 1:
            # Convert the Series to a DataFrame-like structure
            if hasattr(X, "name"):
                X = pl.DataFrame({X.name: X})
            else:
                X = pl.DataFrame(X)
        if not self.filter and self.cat_no is not None:
            self.filter = {}
            for col in X.columns:
                self.filter[col] = (
                    X[col].value_counts().sort("counts")[col][-self.cat_no :].to_list()
                )
        if not self.filter and self.min_values is not None:
            self.filter = {}
            for col in X.columns:
                self.filter[col] = (
                    X[col]
                    .value_counts()
                    .filter(pl.col("counts") >= self.min_values)[col]
                    .to_list()
                )
        if self.most_frequent:
            self.fill_values = {}
            for col in X.columns:
                self.fill_value[col] = (
                    X[col].value_counts().sort("counts")[col].to_list()[-1]
                )
        else:
            self.fill_values = {}
            for col in X.columns:
                self.fill_values[col] = self.fill_value
        return self

    def transform(
        self, X: Union[pl.Series, pl.DataFrame]
    ) -> Union[pl.Series, pl.DataFrame]:
        """
        Transform the input data by filling in values.

        Parameters:
            X (pl.Series or pl.DataFrame): Input data.

        Returns:
            Union[pl.Series, pl.DataFrame]: Transformed data with
            filled values.
        """
        if len(X.shape) == 1:
            # Convert the Series to a DataFrame-like structure
            if hasattr(X, "name"):
                X = pl.DataFrame({X.name: X})
            else:
                X = pl.DataFrame(X)
        X_filled = X
        for col in X_filled.columns:
            X_filled = X_filled.with_columns(
                pl.col(col)
                .map_elements(
                    lambda x: self.fill_values[col] if x not in self.filter[col] else x
                )
                .alias(col)
            )
        if len(X_filled.shape) == 1:
            X_filled = X_filled.to_numpy()
        if self.return_format == "np":
            X_filled = X_filled.to_numpy()
        return X_filled


class PolarsNullImputer(BaseEstimator, TransformerMixin):
    """
    Null imputer for Polars DataFrames.

    This imputer replaces null (missing) values in the input data with
    specified fill values.

    Parameters:
    -----------
    fill_value : Any
        List of fill values to replace null values in each column.

    Attributes:
    -----------
    fill_value : List
        List of fill values to be used for imputation.

    Methods:
    --------
    fit(X, y=None)
        Fit the imputer to the input data.

    transform(X, y=None)
        Transform the input data by replacing null values with the specified
        fill values.

    Returns:
    --------
    X : pl.DataFrame
        Transformed Polars DataFrame with null values replaced by fill values.
    """

    def __init__(self, fill_value: Any) -> None:
        """
        Initialize the PolarsNullImputer.

        Parameters:
        -----------
        fill_value : List
            List of fill values to replace null values in each column.

        Returns:
        --------
        None
        """
        self.fill_value = fill_value

    def fit(self, X, y=None):
        """
        Fit the null imputer to the input data.

        Parameters:
        -----------
        X : pl.DataFrame
            Input data.
        y : None
            Ignored. It is not used in the fitting process.

        Returns:
        --------
        self
        """
        return self

    def transform(self, X: pl.DataFrame, y=None):
        """
        Transform the input data by replacing null values with the specified
        fill values.

        Parameters:
        -----------
        X : pl.DataFrame
            Input data to be imputed.
        y : None
            Ignored. It is not used in the transformation process.

        Returns:
        --------
        X : pl.DataFrame
            Transformed Polars DataFrame with null values replaced by
            fill values.
        """
        if not isinstance(X, pl.DataFrame):
            X = pl.DataFrame(X)
        bool_cols = X.select(pl.col(pl.Boolean)).columns
        for col in bool_cols:
            X = X.with_columns(pl.col(col).cast(pl.Int32).alias(col))
        X = X.fill_null(self.fill_value)
        X = X.fill_nan(self.fill_value)
        return X


def hour_cyclic_features(data: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Create cyclic features for the hour from an integer column.

    This function takes a Polars DataFrame and a column containing hour values
    in 24-hour format (1 to 24).
    It adds two new columns, 'hour_sin' and 'hour_cos', which represent the hour
    cyclically using sine and cosine functions. These features can help capture the
    cyclical patterns in daily data.

    Parameters:
    -----------
    data : pl.DataFrame
        The input Polars DataFrame containing hour values in 24-hour format.
    col : str
        The name of the column containing hour values.

    Returns:
    --------
    data : pl.DataFrame
        The input Polars DataFrame with 'hour_sin' and 'hour_cos' features.
    """
    data = data.with_columns(
        pl.col(col)
        .map_elements(lambda x: np.sin((x - 1) / 24 * 2 * np.pi))
        .alias("hour_sin")
    )
    data = data.with_columns(
        pl.col(col)
        .map_elements(lambda x: np.cos((x - 1) / 24 * 2 * np.pi))
        .alias("hour_cos")
    )
    data = data.drop(col)
    return data


def weekday_cyclic_features(data: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Create cyclic features for the weekday from a string column.

    This function takes a Polars DataFrame and a column containing weekday
    values in string format (e.g., "MONDAY", "TUESDAY", etc.).
    It adds two new columns, 'weekday_sin' and 'weekday_cos', which represent
    the weekday cyclically using sine and cosine functions.

    Parameters:
    -----------
    data : pl.DataFrame
        The input Polars DataFrame containing weekday values in string format.
    col : str
        The name of the column containing weekday values.

    Returns:
    --------
    data : pl.DataFrame
        The input Polars DataFrame with added 'weekday_sin'
        and 'weekday_cos' features.
    """
    weekdays = [
        "MONDAY",
        "TUESDAY",
        "WEDNESDAY",
        "THURSDAY",
        "FRIDAY",
        "SATURDAY",
        "SUNDAY",
    ]

    data = data.with_columns(
        pl.col(col)
        .map_elements(
            lambda x: weekdays.index(x.upper())
        )  # Convert weekday string to index
        .map_elements(lambda x: np.sin(x / 7 * 2 * np.pi))
        .alias("weekday_sin")
    )
    data = data.with_columns(
        pl.col(col)
        .map_elements(
            lambda x: weekdays.index(x.upper())
        )  # Convert weekday string to index
        .map_elements(lambda x: np.cos(x / 7 * 2 * np.pi))
        .alias("weekday_cos")
    )
    data = data.drop(col)
    return data
