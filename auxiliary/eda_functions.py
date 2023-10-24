import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold


def test_with_catboost(
    data: pl.DataFrame, data_val: pl.DataFrame, cat_features, sample_size=None
):
    # Wether to sample
    if sample_size:
        training_sample = data.sample(sample_size, shuffle=True, seed=1)
        val_sample = data_val.sample(int(sample_size / 3), shuffle=True, seed=1)
    else:
        training_sample = data
        val_sample = data_val
    # Filling nulls
    for col in cat_features:
        training_sample = training_sample.with_columns(
            pl.col(col).fill_null("None").alias(col)
        )
        val_sample = val_sample.with_columns(pl.col(col).fill_null("None").alias(col))
    for col in [col for col in training_sample.columns if col not in cat_features]:
        training_sample = training_sample.with_columns(
            pl.col(col).fill_null(-1).alias(col)
        )
        val_sample = val_sample.with_columns(pl.col(col).fill_null(-1).alias(col))
    # Specifying X and y
    X_train = training_sample.drop(columns=["TARGET", "SK_ID_CURR"])
    y_train = training_sample["TARGET"]

    X_val = val_sample.drop(columns=["TARGET", "SK_ID_CURR"])
    y_val = val_sample["TARGET"]

    # Categorical cols
    cat_indices = []

    for col in cat_features:
        cat_indices.append(X_train.find_idx_by_name(col))
    # Model
    cat = CatBoostClassifier(
        cat_features=cat_indices,
        random_seed=1,
        l2_leaf_reg=10,
        auto_class_weights="Balanced",
    )

    # Fitting and evaluation
    cat.fit(X_train.to_numpy(), y_train.to_numpy(), verbose=0)
    preds = cat.predict(X_val.to_numpy())
    preds_proba = cat.predict_proba(X_val.to_numpy())

    initial_importances = pl.DataFrame(
        {
            "feature": X_train.columns,
            "importance": cat.feature_importances_,
        }
    ).sort("importance", descending=True)

    # Preparing Output
    outputs = {}
    outputs["report"] = classification_report(y_val, preds)
    outputs["roc_auc"] = roc_auc_score(y_val, preds_proba[:, 1])
    outputs["features"] = initial_importances
    return outputs


def test_with_catboost_crossval(
    X: pl.DataFrame, y: pl.Series, cat_features=None, sample_size=None
):
    # Wether to sample
    if sample_size:
        X = X.sample(sample_size, shuffle=True, seed=1)
        y = y.sample(sample_size, shuffle=True, seed=1)

    # Filling nulls
    cat_indices = None
    if cat_features:
        cat_indices = []
        for col in cat_features:
            X = X.with_columns(pl.col(col).fill_null("None").alias(col))
        for col in cat_features:
            cat_indices.append(X.find_idx_by_name(col))
    if cat_features:
        num_cols = [col for col in X.columns if col not in cat_features]
    else:
        num_cols = X.columns
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
    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in k_fold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fitting and evaluation
        cat.fit(X_train.to_numpy(), y_train.to_numpy(), verbose=0)
        # preds = cat.predict(X_test.to_numpy())
        preds_proba = cat.predict_proba(X_test.to_numpy())[:, 1]
        scores.append(roc_auc_score(y_test, preds_proba))

    initial_importances = pl.DataFrame(
        {
            "feature": X_train.columns,
            "importance": cat.feature_importances_,
        }
    ).sort("importance", descending=True)

    # Preparing Output
    outputs = {}
    outputs["scores"] = scores
    outputs["features"] = initial_importances
    return outputs
