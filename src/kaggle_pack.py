"""Package for the first cross-encoder (kaggle/xenc_kaggle.py).

texts.parquet        "cleaned name | cleaned address" per record
train_pairs.parquet  stage-2 pairs of folds 1-2 with label y
score_pairs.parquet  pairs to score: folds 0, 3, 4 and the test
"""
import glob
import os

import polars as pl

W = "work"
OUT = "work/kaggle_pack"


def main():
    os.makedirs(OUT, exist_ok=True)
    folds = pl.read_parquet(f"{W}/norm/folds.parquet")
    gt = pl.read_parquet(f"{W}/parquet/gt_pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
    tr = (pl.scan_parquet(sorted(glob.glob(f"{W}/s2te_train/part_*.parquet"))).select("s1_uid", "r_uid")
          .join(folds.lazy(), on="s1_uid").collect(engine="streaming"))
    te = pl.scan_parquet(sorted(glob.glob(f"{W}/s2te_test/part_*.parquet"))).select("s1_uid", "r_uid").collect(
        engine="streaming")
    train_pairs = (tr.filter(pl.col("fold").is_in([1, 2])).drop("fold")
                   .join(gt, on=["s1_uid", "r_uid"], how="left").with_columns(pl.col("y").fill_null(0)))
    score_pairs = pl.concat([tr.filter(pl.col("fold").is_in([0, 3, 4])).drop("fold").with_columns(pl.lit("train").alias("split")),
                             te.with_columns(pl.lit("test").alias("split"))])
    train_pairs.write_parquet(f"{OUT}/train_pairs.parquet", compression="zstd")
    score_pairs.write_parquet(f"{OUT}/score_pairs.parquet", compression="zstd")
    texts = []
    for split, pairs in (("train", pl.concat([train_pairs.select("s1_uid", "r_uid"),
                                              score_pairs.filter(pl.col("split") == "train").select("s1_uid", "r_uid")])),
                         ("test", score_pairs.filter(pl.col("split") == "test").select("s1_uid", "r_uid"))):
        ids = pl.concat([pairs.select(pl.col("s1_uid").alias("uid")), pairs.select(pl.col("r_uid").alias("uid"))]).unique()
        for s in (1, 2, 3):
            texts.append(pl.scan_parquet(f"{W}/norm/{split}_s{s}.parquet").select("uid", "n_full", "a_clean")
                         .join(ids.lazy(), on="uid", how="semi")
                         .select(pl.lit(split).alias("split"), "uid",
                                 (pl.col("n_full") + " | " + pl.col("a_clean").fill_null("")).alias("t"))
                         .collect(engine="streaming"))
    pl.concat(texts).write_parquet(f"{OUT}/texts.parquet", compression="zstd")
    for f in sorted(os.listdir(OUT)):
        print(f, f"{os.path.getsize(os.path.join(OUT, f)) / 1e6:.0f} MB",
              pl.scan_parquet(os.path.join(OUT, f)).select(pl.len()).collect().item(), "rows")
    print("train pairs positive share:", round(train_pairs["y"].mean(), 3))


if __name__ == "__main__":
    main()
