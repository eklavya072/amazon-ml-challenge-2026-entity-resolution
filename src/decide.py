"""Turn pair probabilities into match lists and write the output TSV files.

Each record is kept only for its best S1 (a record belongs to at most one S1). Then, per S1, the top-k candidates that
maximise the expected F0.5 are kept: E[F | k] = 1.25 * sum_{i<=k} p_i / (0.25 * sum_all p + k) and
E[F | 0] = prod(1 - p_i). alpha scales the probabilities first (0.8, chosen on validation).
"""

import polars as pl


def exclusive(pred, col="p"):
    """Keep, for every r_uid, only the pair(s) with the highest probability."""
    return pred.filter(pl.col(col) == pl.col(col).max().over("r_uid"))


def select_expected_f(pred, col="p", alpha=1.0, min_p=0.02):
    """Per S1 entity, keep the top-k candidates maximising the expected F0.5."""
    q = (pred.filter(pl.col(col) >= min_p)
         .with_columns((pl.col(col) * alpha).clip(0.0, 0.999999).alias("_p"))
         .sort(["s1_uid", "_p"], descending=[False, True])
         .with_columns(
             pl.col("_p").cum_sum().over("s1_uid").alias("_cum"),
             pl.col("_p").sum().over("s1_uid").alias("_tot"),
             pl.int_range(1, pl.len() + 1).over("s1_uid").alias("_k"),
             (1 - pl.col("_p")).log().sum().over("s1_uid").exp().alias("_ef0"))
         .with_columns((1.25 * pl.col("_cum") / (0.25 * pl.col("_tot") + pl.col("_k"))).alias("_ef")))
    best = (q.group_by("s1_uid")
            .agg(pl.col("_ef").max().alias("_efmax"),
                 pl.col("_k").get(pl.col("_ef").arg_max()).alias("_kbest"),
                 pl.col("_ef0").first()))
    q = q.join(best, on="s1_uid")
    return q.filter((pl.col("_efmax") > pl.col("_ef0")) & (pl.col("_k") <= pl.col("_kbest"))).select(
        "s1_uid", "r_uid")


def id_lists(pairs, r_ids, col_name):
    """(s1_uid, comma-joined sorted S2/S3 entity_ids) for the S1 entities present in ``pairs``."""
    return (pairs.join(r_ids, on="r_uid")
            .group_by("s1_uid").agg(pl.col("r_entity_id").unique().sort().str.join(",").alias(col_name)))


def full_rows(lists, s1_ids, col_name):
    """One row per test S1 entity (empty list when it has no entry), sorted by id."""
    return (s1_ids.join(lists, on="s1_uid", how="left")
            .select(pl.col("entity_id").alias("source1_entity_id"), pl.col(col_name).fill_null(""))
            .sort("source1_entity_id"))


def write_tsv(df, path):
    """Write a TAB-separated, UTF-8, unquoted file with a header row."""
    df.write_csv(path, separator="\t", quote_style="never", include_header=True)
