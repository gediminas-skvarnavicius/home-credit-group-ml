from sklearn.base import BaseEstimator, TransformerMixin  # type:ignore
import polars as pl
import numpy as np
from typing import Union, Iterable, Optional, Any
from collections import OrderedDict
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed


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


class NumDiffFromRestImputer(BaseEstimator, TransformerMixin):
    def __init__(self, coef: float = 1.0) -> None:
        """
        Imputer that fills missing numeric values based on the
        relationship with the target variable.

        Parameters:
        - coef (float, optional): Coefficient to determine the fill value.
        Default is 1.0.
        """
        self.coef = coef

    def fit(self, X: pl.Series, y: pl.Series) -> "NumDiffFromRestImputer":
        """
        Fit the imputer on the input data.

        Parameters:
        - X (pl.Series): The feature column with missing values.
        - y (pl.Series): The target variable.

        Returns:
        - NumDiffFromRestImputer: The fitted imputer.
        """
        if X.is_null().any():
            self.null_mean = y.filter(X.is_null()).mean()
            self.not_null_mean = y.filter(X.is_not_null()).mean()
            self.corr = pl.DataFrame([X, y]).select(
                pl.corr(
                    X.name,
                    y.name,
                )
            )[0, 0]
            self.x_min = X.min()
            self.x_max = X.max()

            if (
                self.corr is None
                or self.null_mean is None
                or self.not_null_mean is None
            ):
                self.fill_val = -9999

            else:
                if self.corr >= 0:
                    if self.null_mean <= self.not_null_mean:
                        self.fill_val = self.x_min - np.abs(self.coef * self.x_min) - 1
                    else:
                        self.fill_val = self.x_max + np.abs(self.coef * self.x_max) + 1
                else:
                    if self.null_mean <= self.not_null_mean:
                        self.fill_val = self.x_max + np.abs(self.coef * self.x_max) - 1
                    else:
                        self.fill_val = np.abs(self.x_min - self.coef * self.x_min) + 1
        return self

    def transform(self, X: pl.Series, y=None) -> pl.Series:
        """
        Transform the input data by filling missing values.

        Parameters:
        - X (pl.Series): The feature column with missing values.
        - y: Ignored.

        Returns:
        - pl.Series: The transformed feature column.
        """
        if hasattr(self, "fill_val"):
            X = X.fill_null(pl.lit(self.fill_val))
        else:
            X = X.fill_null(pl.lit(X.min() - np.abs(X.min()) - 1))
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


class SamplingModelWrapper(BaseEstimator, TransformerMixin):
    """
    A transformer that wraps an optional oversampling technique with a model.

    Parameters:
    ----------
    model : object
        The underlying model to be used for classification.

    sampler : str, optional (default=None)
        The oversampling technique to use. Supported values are
        'smote', 'adasyn', 'random', or None.
        If set to None, no oversampling will be applied.

    Attributes:
    ----------
    model : object
        The underlying model for classification.

    sampler : str or None
        The oversampling technique to be applied. If None,
        no oversampling is used.

    transformer : object
        The oversampling transformer (SMOTE, ADASYN, or RandomOverSampler)
        based on the 'sampler' parameter.

    Methods:
    --------
    fit(X, y)
        Fits the oversampling transformer (if specified) and the underlying
        model to the input data.

    predict(X)
        Predicts class labels for input data using the underlying model.

    Returns:
    --------
    self : object
        Returns the instance of the transformer.
    """

    def __init__(
        self,
        model,
        sampler=None,
        model_params=None,
    ) -> None:
        """
        Initialize the SamplingModelWrapper.

        Parameters:
        ----------
        model : object
            The underlying model to be used for classification.

        sampler : str, optional (default=None)
            The oversampling technique to use. Supported values are
            'smote', 'adasyn', 'random', or None.
            If set to None, no oversampling will be applied.
        """
        self.sampler = sampler
        self.model = model
        self.model_params = model_params
        if self.model_params:
            self.model.set_params(**self.model_params)

    def fit(
        self,
        X: Union[pl.Series, pl.DataFrame, np.ndarray],
        y: pl.Series = None,
    ) -> "SamplingModelWrapper":
        """
        Fit the transformer to the input data.

        Parameters:
        ----------
        X : Union[pl.Series, pl.DataFrame]
            The input features for training.

        y : pl.Series, optional (default=None)
            The target labels for training.

        Returns:
        --------
        self : object
            Returns the instance of the transformer.
        """
        if self.sampler:
            if self.sampler == "smote":
                self.transformer = SMOTE(random_state=1)
            elif self.sampler == "adasyn":
                self.transformer = ADASYN(random_state=1)
            elif self.sampler == "random":
                self.transformer = RandomOverSampler(random_state=1)
            if isinstance(X, (pl.DataFrame, pl.Series)):
                cols = X.columns
                X, y = self.transformer.fit_resample(
                    X.to_numpy(),
                    y.to_numpy(),
                )
                X = pl.DataFrame(X, schema=cols)
            else:
                X, y = self.transformer.fit_resample(X, y)
        self.model.fit(X, y)
        return self

    def predict(self, X: Union[pl.Series, pl.DataFrame]) -> np.ndarray:
        """
        Predict class labels for the input data.

        Parameters:
        ----------
        X : Union[pl.Series, pl.DataFrame]
            The input features for prediction.

        Returns:
        --------
        pd.Series
            Predicted class labels.
        """
        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X: Union[pl.Series, pl.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities for the input data.

        Parameters:
        ----------
        X : Union[pl.Series, pl.DataFrame]
            The input features for prediction.

        Returns:
        --------
        pd.Series
            Predicted class labels.
        """
        predictions = self.model.predict_proba(X)
        return predictions


class SimplerStacker(BaseEstimator, TransformerMixin):
    """
    A simple stacking transformer that combines predictions from base models
    using cross-validated predictions and then fits a final estimator
    on the aggregated predictions.

    Parameters:
    - base_models (Iterable): An iterable containing the base
    models to be stacked.
    - final_estimator: The final estimator that will be trained on
    the aggregated predictions of base models.

    Methods:
    - fit(X, y): Fit the base models and the final estimator on the input data.
    - predict_proba(X): Generate class probabilities for the input data.

    Note: The final estimator should be compatible with the `fit` and `predict_proba` methods.
    """

    def __init__(
        self,
        base_models: Iterable[Any],
        final_estimator: Any,
    ) -> None:
        """
        Initialize the SimplerStacker.

        Parameters:
        - base_models (Iterable): An iterable containing the base models
        to be stacked.
        - final_estimator: The final estimator that will be trained on the
        aggregated predictions of base models.
        """
        self.base_models = base_models
        self.final_estimator = final_estimator

    def fit(self, X, y):
        """
        Fit the SimplerStacker on the input data.

        Parameters:
        - X: Input features.
        - y: Target values.

        Returns:
        - self: Returns an instance of the fitted SimplerStacker.
        """
        base_predictions = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        # Define a function for parallel execution
        def train_and_predict(train_index, test_index):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            model.fit(X_train, y_train)
            return model.predict_proba(X_test)[:, 1]

        for model in self.base_models:
            y_pred_cv = np.zeros_like(y, dtype=float)

            # Perform parallelized cross-validated predictions
            results = Parallel(n_jobs=-1)(
                delayed(train_and_predict)(train_index, test_index)
                for train_index, test_index in kf.split(X, y)
            )
            for i, (train_index, test_index) in enumerate(kf.split(X, y)):
                y_pred_cv[test_index] = results[i]

            base_predictions.append(y_pred_cv)
            model.fit(X, y)
        base_predictions = np.column_stack(base_predictions)
        self.final_estimator.fit(base_predictions, y)
        return self

    def predict_proba(self, X):
        """
        Generate class probabilities for the input data.

        Parameters:
        - X: Input features.

        Returns:
        - predictions: Class probabilities.
        """
        base_predictions = []
        for model in self.base_models:
            base_predictions.append(model.predict_proba(X)[:, 1])
        base_predictions = np.column_stack(base_predictions)
        predictions = self.final_estimator.predict_proba(base_predictions)
        return predictions


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
