import polars as pl
from sklearn.metrics import roc_auc_score, roc_curve
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from IPython.display import Markdown
from tabulate import tabulate
import numpy as np
from typing import Union, Tuple, List
import matplotlib.pyplot as plt


def test_with_catboost_crossval(
    X: pl.DataFrame,
    y: pl.Series,
    cat_features=None,
    sample_size=None,
    kfolds: int = None,
):
    # Wether to sample
    if sample_size:
        X = X.sample(sample_size, shuffle=True, seed=1)
        y = y.sample(sample_size, shuffle=True, seed=1)

    # Filling nulls
    if not cat_features:
        cat_features = X.select(pl.col(pl.Utf8)).columns

    cat_indices = []
    for col in cat_features:
        X = X.with_columns(pl.col(col).fill_null("None").alias(col))
    for col in cat_features:
        cat_indices.append(X.find_idx_by_name(col))

    num_cols = [col for col in X.columns if col not in cat_features]

    for col in num_cols:
        X = X.with_columns(pl.col(col).fill_null(-1).alias(col))
    # Categorical col indices

    # Model
    cat = CatBoostClassifier(
        cat_features=cat_indices,
        random_seed=1,
        l2_leaf_reg=10,
        auto_class_weights="Balanced",
    )
    scores = []
    if kfolds:
        k_fold = StratifiedKFold(
            n_splits=kfolds,
            shuffle=True,
            random_state=42,
        )
        for train_index, test_index in k_fold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fitting and evaluation
            cat.fit(X_train.to_numpy(), y_train.to_numpy(), verbose=0)
            # preds = cat.predict(X_test.to_numpy())
            preds_proba = cat.predict_proba(X_test.to_numpy())[:, 1]
            scores.append(roc_auc_score(y_test, preds_proba))

    cat.fit(X.to_numpy(), y.to_numpy(), verbose=0)
    if not kfolds:
        preds_proba = cat.predict_proba(X.to_numpy())[:, 1]
        scores.append(roc_auc_score(y, preds_proba))

    initial_importances = pl.DataFrame(
        {
            "feature": X.columns,
            "importance": cat.feature_importances_,
        }
    ).sort("importance", descending=True)

    # Preparing Output
    outputs = {}
    outputs["scores"] = scores
    outputs["features"] = initial_importances
    return outputs


def make_aggregations(
    original_df,
    df_to_agg,
    columns,
    id,
    aggregations=["mean", "sum", "min", "max", "std", "mode"],
    join_suffix=None,
):
    for col in columns:
        if "mean" in aggregations:
            original_df = original_df.join(
                df_to_agg.group_by(pl.col(id)).agg(pl.col(col).mean().suffix("_mean")),
                on=id,
                how="left",
                suffix=join_suffix if join_suffix is not None else "",
            )
        if "sum" in aggregations:
            original_df = original_df.join(
                df_to_agg.group_by(pl.col(id)).agg(pl.col(col).sum().suffix("_sum")),
                on=id,
                how="left",
                suffix=join_suffix if join_suffix is not None else "",
            )

        if "min" in aggregations:
            original_df = original_df.join(
                df_to_agg.group_by(pl.col(id)).agg(pl.col(col).min().suffix("_min")),
                on=id,
                how="left",
                suffix=join_suffix if join_suffix is not None else "",
            )

        if "max" in aggregations:
            original_df = original_df.join(
                df_to_agg.group_by(pl.col(id)).agg(pl.col(col).max().suffix("_max")),
                on=id,
                how="left",
                suffix=join_suffix if join_suffix is not None else "",
            )

        if "std" in aggregations:
            original_df = original_df.join(
                df_to_agg.group_by(pl.col(id)).agg(pl.col(col).std().suffix("_std")),
                on=id,
                how="left",
                suffix=join_suffix if join_suffix is not None else "",
            )

        if "mode" in aggregations:
            original_df = original_df.join(
                df_to_agg.group_by(pl.col(id)).agg(
                    pl.col(col).mode().first().suffix("_mode")
                ),
                on=id,
                how="left",
                suffix=join_suffix if join_suffix is not None else "",
            )
    return original_df


def missing_values_by_feature(data):
    missing = pl.DataFrame(
        {
            "missing_fraction": data.null_count().transpose().to_series() / len(data),
            "feature": data.columns,
        }
    ).sort("missing_fraction", descending=True)
    return missing


def table_display(table: pl.DataFrame):
    return Markdown(
        tabulate(
            table.to_pandas(),
            showindex=False,
            headers="keys",
            tablefmt="pipe",
        )
    )


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


def hour_cyclic_features(data: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Create cyclic features for the hour from an integer column.

    This function takes a Polars DataFrame and a column containing hour values
    in 24-hour format (1 to 24).
    It adds two new columns, 'hour_sin' and 'hour_cos', which represent
    the hour cyclically using sine and cosine functions.
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


def plot_roc_curve(
    y_test: Union[List[int], np.ndarray],
    predictions: Union[List[float], np.ndarray],
    ax: plt.Axes = None,
    **subplots_kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the Receiver Operating Characteristic (ROC) curve with the
    Area Under the Curve (AUC) score.

    Parameters:
        y_test (List[int] or np.ndarray):
        True binary labels for the test set.

        predictions (List[float] or np.ndarray):
        Predicted probabilities for the positive class.

        base_fig_width (int, optional):
        Width of the generated plot in inches. Default is 8.

        base_fig_height (int, optional):
        Height of the generated plot in inches. Default is 6.

    Returns:
        Tuple[plt.Figure, plt.Axes]:
        The generated matplotlib Figure and Axes objects.

    Example:
        y_test = [0, 1, 1, 0, 1]
        predictions = [0.1, 0.8, 0.6, 0.2, 0.9]
        fig, ax = plot_roc_curve(y_test, predictions)
        plt.show()
    """
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    auc_score = roc_auc_score(y_test, predictions)

    if ax:
        ax_combo_roc = ax
    else:
        fig_combo_roc, ax_combo_roc = plt.subplots(**subplots_kwargs)
    ax_combo_roc.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % auc_score)
    ax_combo_roc.plot([0, 1], [0, 1], "k--")  # Random guessing line
    ax_combo_roc.set_xlabel("False Positive Rate")
    ax_combo_roc.set_ylabel("True Positive Rate")
    ax_combo_roc.legend(loc="lower right")

    if not ax:
        return fig_combo_roc, ax_combo_roc
