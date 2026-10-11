"""Shared LightGBM settings. Fold 0 (20% of the S1 entities) is the validation set throughout."""

import lightgbm as lgb
import polars as pl

NON_FEATURES = {"s1_uid", "r_uid", "y", "fold", "simple"}

PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, num_threads=8, verbose=-1, metric=["binary_logloss", "auc"])


def pseudo_random(frac):
    """Deterministic row filter keeping ~frac of pairs (hash of the pair ids)."""
    return ((pl.col("s1_uid") * 1000003 + pl.col("r_uid")) % 1000) < int(frac * 1000)


def train_model(X, y, Xv, yv, params, rounds, log_every=100):
    """Fit LightGBM with early stopping on the validation log-loss."""
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    dva = lgb.Dataset(Xv, yv, reference=dtr)
    return lgb.train(params, dtr, rounds, valid_sets=[dva], valid_names=["val"],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(log_every)])
