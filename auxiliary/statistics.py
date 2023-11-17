from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import combinations
import numpy as np
from multiprocessing import Pool
import polars as pl
import scipy.stats as stats
from typing import Optional, Tuple
from multiprocessing import get_context
import networkx


def get_correlation(data: pl.DataFrame, combo: tuple) -> float:
    """
    Calculates the Spearman correlation between two columns
    in a Polars DataFrame.

    Parameters:
    - data (pl.DataFrame): The input DataFrame.
    - combo (tuple): A tuple containing two column names
    for which to calculate the correlation.

    Returns:
    - float: The Spearman correlation coefficient between
    the specified columns.
    """
    corr = data.select(
        pl.corr(combo[0], combo[1], method="spearman"),
    ).to_numpy()[
        0
    ][0]

    return corr


def get_correlation_pairs(
    data: pl.DataFrame,
    max_threshold: float = 0.99,
    min_threshold: float = -0.99,
    num_processes=6,
) -> pl.DataFrame:
    """
    Find pairs of features in a DataFrame with correlations above a
    certain threshold using Spearman's method.

    Parameters:
    - data (DataFrame): The input DataFrame containing the features.
    - max_threshold (float): The maximum correlation threshold
    (default is 0.99).
    - min_threshold (float): The minimum correlation threshold
    (default is -0.99).

    Returns:
    - DataFrame: A DataFrame containing the correlated feature pairs
    and their correlations.

    This function computes correlations between all possible pairs of features
    using Spearman's method
    and returns those pairs with correlations exceeding the specified
    thresholds.
    """

    list_combos = list(combinations(data.columns, 2))

    with get_context("spawn").Pool(num_processes) as pool:
        correlations = pool.starmap(
            get_correlation,
            [(data, combo) for combo in list_combos],
        )

    correlation_df = pl.DataFrame(
        {"features": list_combos, "correlation": correlations}
    )

    correlation_df = correlation_df.filter(
        ~(pl.col("correlation").is_between(min_threshold, max_threshold))
    )
    results = {}
    results["pairs"] = correlation_df
    results["clusters"] = []
    graph = networkx.Graph()
    graph.add_edges_from(correlation_df["features"].to_list())
    for connected in networkx.connected_components(graph):
        results["clusters"].append(connected)
    return results


def calc_vif(X: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates the Variance Inflation Factor (VIF)
    for each variable in a DataFrame.

    Parameters:
    - X (pl.DataFrame): The input DataFrame.

    Returns:
    - pl.DataFrame: A DataFrame containing the calculated VIF
    values for each variable, sorted in descending order.
    """
    # Calculating VIF
    vif = pl.DataFrame()
    vif = vif.with_columns(pl.Series(X.columns).alias("variables"))
    vif = vif.with_columns(
        pl.Series(
            [variance_inflation_factor(X.to_numpy(), i) for i in range(X.shape[1])]
        ).alias("VIF")
    )

    return vif.sort("VIF", descending=True)


def kruskal_polars(
    data: pl.DataFrame,
    class_col: str,
    var_col: str,
    sample_size: Optional[int] = None,
    seed: int = 1,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Perform the Kruskal-Wallis test on a Polars DataFrame.

    Args:
        data (pl.DataFrame): The Polars DataFrame containing the data.
        class_col (str): The name of the column containing class or group
        information.
        var_col (str): The name of the column containing the
        variable of interest.
        sample_size (int, optional): The sample size for each class.
        If None, use all available data for each class.
        seed (int): Seed for random sampling (if sample_size is provided).

    Returns:
        Tuple[float, float]: The Kruskal-Wallis H-statistic
        and the associated p-value.
    """
    # Get unique class values and sort them
    unique_classes = data[class_col].unique().sort()

    samples = []
    # Sample and collect data for each class
    for class_val in unique_classes:
        class_data = (
            data.filter(data[class_col] == class_val)
            .select(
                var_col,
            )
            .drop_nulls()
        )
        if sample_size is not None:
            class_data = class_data.sample(sample_size, seed=seed)
        samples.append(class_data[var_col])

    # Perform the Kruskal-Wallis test
    result = {}
    result["statistic"], result["p_value"] = stats.kruskal(*samples)
    if result["p_value"] < alpha:
        result["message"] = (
            f"Reject the null hypothesis: There are significant "
            f"differences in '{var_col}' between groups in '{class_col}'."
        )
    else:
        result["message"] = (
            f"Fail to reject the null hypothesis: There are no "
            f"significant differences in '{var_col}' between groups "
            f"in '{class_col}'."
        )
    return result


def chi_squared_polars(
    data: pl.DataFrame, class_col: str, var_col: str, alpha: float = 0.05
):
    """
    Perform a chi-squared test of independence on a Polars DataFrame.

    Args:
        data (pl.DataFrame): The Polars DataFrame containing the data.
        class_col (str): The name of the column containing class or group
        information.
        var_col (str): The name of the column to be used for the test
        (e.g., "disbursement_method").
        alpha (float): The significance level for hypothesis testing.

    Returns:
        dict: A dictionary containing the p-value, test statistic,
        and result message.
    """
    data = data.select([class_col, var_col]).drop_nulls()
    contingency_table = data[var_col].value_counts()
    for cat in data[class_col].unique():
        contingency_table = contingency_table.join(
            data.filter(data[class_col] == cat)[var_col].value_counts(),
            on=var_col,
            how="left",
            suffix=f"_{cat}",
        )

    contingency_table = contingency_table.drop(columns=["count", var_col])
    contingency_table = contingency_table.drop_nulls()
    test = stats.chi2_contingency(contingency_table)

    result = {}
    result["p_value"] = test.pvalue
    result["statistic"] = test.statistic

    if result["p_value"] < alpha:
        result["message"] = (
            f"Reject the null hypothesis: There are significant "
            f"differences in '{var_col}' between groups in '{class_col}'."
        )
    else:
        result["message"] = (
            f"Fail to reject the null hypothesis: There are no "
            f"significant differences in '{var_col}' between groups "
            f"in '{class_col}'."
        )

    return result
