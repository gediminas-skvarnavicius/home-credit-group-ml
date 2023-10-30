from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import combinations
import numpy as np
from multiprocessing import Pool
import polars as pl
from multiprocessing import get_context
import networkx


def get_correlation(data, combo):
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
    results["pairs"] = "correlation_df"
    results["clusters"] = []
    graph = networkx.Graph()
    graph.add_edges_from(correlation_df["features"].to_list())
    for connected in networkx.connected_components(graph):
        results["clusters"].append(connected)
    return results


def calc_vif(X):
    # Calculating VIF
    vif = pl.DataFrame()
    vif = vif.with_columns(pl.Series(X.columns).alias("variables"))
    vif = vif.with_columns(
        pl.Series(
            [variance_inflation_factor(X.to_numpy(), i) for i in range(X.shape[1])]
        ).alias("VIF")
    )

    return vif.sort("VIF", descending=True)
