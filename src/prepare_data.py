"""Convert the raw TSV files to Parquet and give every record an integer id.

uid = source number * 10**10 + numeric part of entity_id (S2-681193310 -> 20681193310). The original entity_id is
kept for writing the outputs.

    python src/prepare_data.py --train-dir "Dataset/Train files tsv" --test-dir "Dataset/Test files tsv"
"""

import argparse
import os

import polars as pl

UID_BASE = 10**10  # numeric parts of entity_ids have at most 10 digits (checked below)


def read_tsv(path):
    """Read a challenge TSV with every column as text (quotes are literal characters)."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def add_uid(df):
    """Add src (1/2/3) and the int64 uid; fails if an id is not S<1|2|3>-<digits>."""
    ok = df["entity_id"].str.contains(r"^S[123]-\d{1,10}$")
    if not ok.all():
        bad = df.filter(~ok)["entity_id"].head(5).to_list()
        raise ValueError(f"unexpected entity_id format, e.g. {bad}")
    return df.with_columns(
        pl.col("entity_id").str.slice(1, 1).cast(pl.Int8).alias("src"),
        (
            pl.col("entity_id").str.slice(1, 1).cast(pl.Int64) * UID_BASE
            + pl.col("entity_id").str.slice(3).cast(pl.Int64)
        ).alias("uid"),
    )


def convert_sources(split, in_dir, out_dir):
    """Convert <split>_source{1,2,3}.tsv to Parquet and print a one-line summary each."""
    for s in (1, 2, 3):
        src_path = os.path.join(in_dir, f"{split}_source{s}.tsv")
        df = add_uid(read_tsv(src_path))
        assert df["uid"].n_unique() == df.height, f"duplicate IDs in {src_path}"
        out = os.path.join(out_dir, f"{split}_s{s}.parquet")
        df.write_parquet(out, compression="zstd")
        print(f"{split} S{s}: {df.height:,} rows -> {out}")


def convert_ground_truth(in_dir, out_dir):
    """gt_pairs.parquet: one row per (S1, matched record); gt_s1.parquet: number of matches per S1."""
    gt = read_tsv(os.path.join(in_dir, "train_ground_truth.tsv"))
    gt = gt.rename({"source1_entity_id": "entity_id"})
    gt = add_uid(gt).rename({"uid": "s1_uid"})
    lists = gt.select(
        "s1_uid",
        pl.col("matched_entity_ids").fill_null("").str.split(",")
        .list.eval(pl.element().filter(pl.element() != "")).alias("m"),
    )
    lists.select("s1_uid", pl.col("m").list.len().alias("n_matches")).write_parquet(
        os.path.join(out_dir, "gt_s1.parquet"), compression="zstd")
    pairs = lists.explode("m").drop_nulls("m").rename({"m": "entity_id"})
    pairs = add_uid(pairs).select("s1_uid", pl.col("uid").alias("r_uid"))
    assert pairs["r_uid"].n_unique() == pairs.height, "an S2/S3 id is matched to >1 S1"
    pairs.write_parquet(os.path.join(out_dir, "gt_pairs.parquet"), compression="zstd")
    print(f"ground truth: {gt.height:,} S1 rows, {pairs.height:,} true pairs")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out-dir", default="work/parquet")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    convert_sources("train", a.train_dir, a.out_dir)
    convert_sources("test", a.test_dir, a.out_dir)
    convert_ground_truth(a.train_dir, a.out_dir)


if __name__ == "__main__":
    main()
