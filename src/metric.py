"""Challenge metric: F0.5 per Source-1 entity, averaged over all entities.

An entity with no true match scores 1 if its prediction is empty and 0 otherwise.
"""

import polars as pl

BETA2 = 0.25  # beta = 0.5 -> beta^2 = 0.25


def score_pairs(pred_pairs, true_pairs, s1_universe, by=None):
    """Macro F0.5 of pred_pairs against true_pairs over every S1 in s1_universe (optionally per group `by`)."""
    pred_pairs = pred_pairs.select("s1_uid", "r_uid").unique()
    true_pairs = true_pairs.select("s1_uid", "r_uid").unique()
    n_pred = pred_pairs.group_by("s1_uid").len("n_pred")
    n_true = true_pairs.group_by("s1_uid").len("n_true")
    n_hit = pred_pairs.join(true_pairs, on=["s1_uid", "r_uid"]).group_by("s1_uid").len("n_hit")
    e = (
        s1_universe.join(n_pred, on="s1_uid", how="left")
        .join(n_true, on="s1_uid", how="left")
        .join(n_hit, on="s1_uid", how="left")
        .with_columns(pl.col("n_pred", "n_true", "n_hit").fill_null(0))
    )
    p = pl.col("n_hit") / pl.col("n_pred")
    r = pl.col("n_hit") / pl.col("n_true")
    f = (
        pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
        .when(pl.col("n_hit") == 0).then(0.0)
        .otherwise((1 + BETA2) * p * r / (BETA2 * p + r))
    )
    e = e.with_columns(f.alias("f05"))
    overall = e["f05"].mean()
    if by is None:
        return overall, e
    groups = e.group_by(by).agg(pl.len().alias("n"), pl.col("f05").mean()).sort(by)
    return overall, e, groups
