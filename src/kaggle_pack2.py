"""Raw-text package for the second cross-encoder (kaggle/xenc2_kaggle.py); same pairs as kaggle_pack.py."""
import polars as pl

W, OUT = "work", "work/kaggle_pack2"
pairs = pl.concat([pl.scan_parquet(f"{W}/kaggle_pack/train_pairs.parquet").select("s1_uid", "r_uid", pl.lit("train").alias("split")),
                   pl.scan_parquet(f"{W}/kaggle_pack/score_pairs.parquet").select("s1_uid", "r_uid", "split")]).collect()
for split in ("train", "test"):
    p = pairs.filter(pl.col("split") == split)
    ids = pl.concat([p.select(pl.col("s1_uid").alias("uid")), p.select(pl.col("r_uid").alias("uid"))]).unique()
    for s in (1, 2, 3):
        (pl.scan_parquet(f"{W}/norm/{split}_s{s}.parquet").select("uid", "business_name", "business_address")
                     .join(ids.lazy(), on="uid", how="semi")
                     .select(pl.lit(split).alias("split"), "uid",
                             (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias("t"))
                     .sink_parquet(f"{OUT}/texts_raw_{split}_s{s}.parquet", compression="zstd", compression_level=9))
        print(split, s, flush=True)
print("done")
