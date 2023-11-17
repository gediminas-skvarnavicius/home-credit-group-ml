from ray import tune, train
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from typing import Optional, Dict, Union
import polars as pl
import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from ray.tune.search.optuna import OptunaSearch
from sklearn.model_selection import StratifiedKFold, KFold
from ray.tune.stopper import (
    CombinedStopper,
    MaximumIterationStopper,
    TrialPlateauStopper,
)


def objective(
    pipeline: Pipeline,
    params: dict,
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_val: pl.DataFrame,
    y_val: pl.Series,
    n: int,
    metric: str = "roc_auc",
) -> float:
    """
    Objective function for hyperparameter tuning.

    Parameters:
    - pipeline (Pipeline): The machine learning pipeline to be evaluated.
    - params (dict): Hyperparameter configuration for the pipeline.
    - X_train (pl.DataFrame): Training data features as a Polars DataFrame.
    - y_train (pl.Series): Training data labels as a Polars Series.
    - X_val (pl.DataFrame): Validation data features as a Polars DataFrame.
    - y_val (pl.Series): Validation data labels as a Polars Series.
    - n (int): The current iteration number.
    - metric (str, optional): The metric for scoring. Supported metrics:
    "roc_auc", "f1", "rmse".

    Returns:
    - float: The score based on the specified metric.
    """
    pipeline.set_params(**params)
    pipeline.fit(X_train, y_train)

    if metric == "roc_auc":
        preds = pipeline.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
    elif metric == "f1":
        preds = pipeline.predict(X_val)
        score = f1_score(y_val, preds)
    elif metric == "rmse":
        preds = pipeline.predict(X_val)
        score = -mean_squared_error(y_val, preds, squared=False)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    print(f"Step {n} Score: {score}")
    return score


class TrainableCV(tune.Trainable):
    """
    A custom Ray Tune trainable class for hyperparameter tuning.

    This class is used to configure and execute hyperparameter
    tuning experiments using Ray Tune. It sets up the necessary
    parameters and data for each trial, and performs steps to
    evaluate the hyperparameter configurations.

    Attributes:
    - config (dict): A dictionary of hyperparameters for the pipeline.
    - pipeline: The machine learning pipeline to be configured and evaluated.
    - X_train: Training data features.
    - y_train: Training data labels.
    - sample_size (Union[int, str]): The sample size for data splitting.
    - metric (str): The metric used.
    - stratify (bool): Whether to stratify data splitting.

    Methods:
    - setup(config, pipeline, X_train, y_train, X_val, y_val, sample_size,
    metric, stratify):
        Set up the trainable object with hyperparameters and data.

    - step():
        Perform a training step and return the score.

    """

    def setup(
        self,
        config: dict,
        pipeline: Pipeline,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        sample_size: Optional[int] = None,
        metric: str = "roc_auc",
        stratify: bool = True,
        n_splits: int = 5,
    ):
        """
        Set up the trainable object with hyperparameters and data.

        Args:
        config (dict): A dictionary of hyperparameters.
        pipeline: The machine learning pipeline.
        X_train: Training data features.
        y_train: Training data labels.
        sample_size (Union[int, str], optional): The sample size for data
        splitting.
        n_splits: The number of splits for cross-validation.

        metric (str, optional): The metric used for scoring.

        stratify (bool, optional): Whether to stratify data splitting.
        Default is True.
        """
        self.x = 0
        self.params = config
        self.pipeline = pipeline
        self.X_train = X_train
        self.y_train = y_train
        self.sample_size = sample_size
        self.metric = metric
        self.scores = np.array([])

        if stratify:
            self.splitter = StratifiedKFold(n_splits)
        else:
            self.splitter = KFold(n_splits)

        self.fold_indices = []
        if not sample_size:
            for train_index, test_index in self.splitter.split(
                X_train,
                y_train,
            ):
                self.fold_indices.append((train_index, test_index))
        else:
            for train_index, test_index in self.splitter.split(
                X_train.sample(sample_size, seed=1),
                y_train.sample(
                    sample_size,
                    seed=1,
                ),
            ):
                self.fold_indices.append((train_index, test_index))

    def step(self):
        """
        Perform a training step.

        Returns:
        dict: A dictionary containing the score for the current step.
        """
        try:
            score = objective(
                self.pipeline,
                self.params,
                self.X_train[self.fold_indices[self.x][0]],
                self.y_train[self.fold_indices[self.x][0]],
                self.X_train[self.fold_indices[self.x][1]],
                self.y_train[self.fold_indices[self.x][1]],
                self.x,
                self.metric,
            )
            self.scores = np.append(self.scores, score)
            self.x += 1
        except:
            print(f"cross val {self.x} False")
        return {"score": self.scores.mean()}


class Models:
    """
    Container for managing and evaluating machine learning models using Polars
    and Ray Tune for hyperparameter optimization.

    This class allows you to add, tune, and evaluate machine learning models.

    Attributes:
    -----------
    models : dict
        A dictionary to store machine learning models.

    Methods:
    --------
    add_model(model_name, pipeline, param_grid, override_n=None,
    metric_threshold=0.55)
        Add a machine learning model to the container.

    remove_model(model_name)
        Remove a machine learning model from the container.

    tune_all(X_train, y_train, X_val, y_val, **kwargs)
        Tune and cross-validate all models in the container.

    """

    def __init__(self) -> None:
        self.models: dict = {}

    class Model:
        """
        Represents an individual machine learning model with methods for
        tuning and evaluation.

        Parameters:
        -----------
        name : str
            The name of the model.
        pipeline : Pipeline
            The scikit-learn pipeline for the model.
        param_grid : dict
            A dictionary of hyperparameter search spaces for the model.
        metric_threshold : float, optional
            Threshold of tuning metric for early stopping. Default is 0.55.
        override_n : int, optional
            Number of trials for hyperparameter optimization.
            Overrides the default value.

        Methods:
        --------
        tune_model(X_train, y_train, X_val, y_val, n=10, n_training=10,
        sample_size=100000, metric="roc_auc", stratify=True)
            Tune the model's hyperparameters using Ray Tune and Optuna.

        """

        def __init__(
            self,
            name: str,
            pipeline: Pipeline,
            param_grid: Dict,
            metric_threshold: float = 0.55,
            override_n: Optional[int] = None,
        ) -> None:
            self.pipeline: Pipeline = pipeline
            self.param_grid = param_grid
            self.best_params: Dict
            self.override_n = override_n
            self.name = name
            self.metric_threshold = metric_threshold

        def tune_model(
            self,
            X_train: Union[pl.DataFrame, np.ndarray, pl.Series],
            y_train: Union[pl.DataFrame, np.ndarray, pl.Series],
            X_val: Union[pl.DataFrame, np.ndarray, pl.Series] = None,
            y_val: Union[pl.DataFrame, np.ndarray, pl.Series] = None,
            n: int = 100,
            n_iter: int = 5,
            sample_size: int = None,
            metric: str = "roc_auc",
            stratify=True,
        ):
            """
            Tune the model's hyperparameters using Ray Tune and Optuna.

            Parameters:
            X_train : Union[pl.DataFrame, np.ndarray, pl.Series]
                The feature matrix for training.
            y_train : Union[pl.DataFrame, np.ndarray, pl.Series]
                The target variable for training.
            X_val : Union[pl.DataFrame, np.ndarray, pl.Series]
                The feature matrix for validation.
            y_val : Union[pl.DataFrame, np.ndarray, pl.Series]
                The target variable for validation.
            n : int, optional
                Number of trials for hyperparameter optimization.
                Default is 10.
            n_iter : int, optional
                Number of training iterations. Default is 10.
            sample_size : int, optional
                The sample size for data splitting. Default is 100,000.
            metric : str, optional
                The metric for evaluation. Default is "roc_auc".
            stratify : bool, optional
                Whether to stratify data splitting. Default is True.
            """
            stopper = CombinedStopper(
                MaximumIterationStopper(n_iter),
                TrialPlateauStopper(
                    std=1,
                    grace_period=1,
                    num_results=1,
                    metric="score",
                    metric_threshold=self.metric_threshold,
                    mode="min",
                ),
            )
            tuner = tune.Tuner(
                trainable=tune.with_resources(
                    tune.with_parameters(
                        TrainableCV,
                        pipeline=self.pipeline,
                        X_train=X_train,
                        y_train=y_train,
                        sample_size=sample_size,
                        metric=metric,
                        stratify=stratify,
                    ),
                    resources={"CPU": 2},
                ),
                run_config=train.RunConfig(
                    stop=stopper,
                    storage_path="/tmp/tune_results/",
                    name=self.name,
                    checkpoint_config=train.CheckpointConfig(
                        checkpoint_at_end=False,
                    ),
                ),
                tune_config=tune.TuneConfig(
                    search_alg=OptunaSearch(),
                    mode="max",
                    metric="score",
                    num_samples=n,
                ),
                param_space=self.param_grid,
            )

            results = tuner.fit()
            self.best_params = results.get_best_result().config
            self.pipeline.set_params(**self.best_params)

        def cross_val_roc_auc(
            self, X: pl.DataFrame, y: pl.Series, n: int = 5, n_jobs: int = -1
        ) -> np.ndarray:
            """
            Perform parallelized cross-validation using ROC AUC as the
            evaluation metric.

            Parameters:
            - X (pl.DataFrame): The input features as a Polars DataFrame.
            - y (pl.Series): The target variable as a Polars Series.
            - n (int, optional): Number of folds for cross-validation.
            - n_jobs (int, optional): Number of parallel jobs. Default is -1,
            which means using all available CPUs.

            Returns:
            - np.ndarray: Array of ROC AUC scores for each fold.
            """

            def process_fold(train_index, test_index, X, y, pipeline):
                pipeline.fit(X[train_index], y[train_index])
                return roc_auc_score(
                    y[test_index], pipeline.predict_proba(X[test_index])[:, 1]
                )

            cv = StratifiedKFold(n)
            parallel = Parallel(n_jobs=n_jobs)

            scores = parallel(
                delayed(process_fold)(
                    train_index,
                    test_index,
                    X,
                    y,
                    self.pipeline,
                )
                for train_index, test_index in cv.split(X, y)
            )

            return scores

        def cross_val_mse(
            self, X: pl.DataFrame, y: pl.Series, n: int = 5, **kwargs
        ) -> list:
            """
            Perform cross-validation using mean squared error (MSE) as the
            evaluation metric.

            Parameters:
            - X (pl.DataFrame): The input features as a Polars DataFrame.
            - y (pl.Series): The target variable as a Polars Series.
            - n (int, optional): Number of folds for cross-validation.
            Default is 5.
            - **kwargs: Additional keyword arguments to be passed to
            `mean_squared_error`.

            Returns:
            - list: Array of MSE scores for each fold.
            """
            cv = KFold(n)
            scores = []
            for train_index, test_index in cv.split(X, y):
                self.pipeline.fit(X[train_index], y[train_index])
                score = mean_squared_error(
                    y[test_index],
                    self.pipeline.predict(X[test_index]),
                    **kwargs,
                )
                scores.append(score)
            return scores

    def add_model(
        self,
        model_name: str,
        pipeline: Pipeline,
        param_grid: Dict,
        override_n: Optional[int] = None,
        metric_threshold: float = 0.55,
    ):
        """
        Add a machine learning model to the container.

        Parameters:
        model_name : str
            The name of the model.
        pipeline : Pipeline
            The scikit-learn pipeline for the model.
        param_grid : Dict
            A dictionary of hyperparameter search spaces for the model.
        override_n : int, optional
            Number of trials for hyperparameter optimization.
            Overrides the default value.
        metric_threshold : float, optional
            Threshold of tuning metric for early stopping. Default is 0.55.
        """
        self.models[model_name] = self.Model(
            model_name,
            pipeline,
            param_grid,
            override_n=override_n,
            metric_threshold=metric_threshold,
        )

    def remove_model(self, model_name: str):
        """
        Remove a machine learning model from the container.

        Parameters:
            model_name : str
                The name of the model to be removed.
        """
        if model_name in self.models:
            del self.models[model_name]

    def tune_all(
        self,
        X_train: Union[pl.DataFrame, np.ndarray, pl.Series],
        y_train: Union[pl.DataFrame, np.ndarray, pl.Series],
        X_val: Union[pl.DataFrame, np.ndarray, pl.Series] = None,
        y_val: Union[pl.DataFrame, np.ndarray, pl.Series] = None,
        **kwargs,
    ):
        """
        Tune and cross-validate all models in the container.

        Parameters:
        X_train : Union[pl.DataFrame, np.ndarray, pl.Series]
            The feature matrix for training.
        y_train : Union[pl.DataFrame, np.ndarray, pl.Series]
            The target variable for training.
        X_val : Union[pl.DataFrame, np.ndarray, pl.Series]
            The feature matrix for validation.
        y_val : Union[pl.DataFrame, np.ndarray, pl.Series]
            The target variable for validation.
        **kwargs
            Additional keyword arguments to be passed to the tune_model method.
        """
        for name, model in self.models.items():
            if model.override_n:
                kwargs["n"] = model.override_n
            model.tune_model(X_train, y_train, X_val, y_val, **kwargs)
            print(f"{name} tuned.")
