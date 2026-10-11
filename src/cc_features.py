"""Per-source copy counts (cc_* columns).

For each pair: how many confident copies (p1 >= 0.9) the S1 already has in the record's source, and the same for the
record's best other S1. Useful for address-less records whose name several S1 share.
Reads work/xside_{split}, writes work/xside_cc_{split}.
"""
import glob

import os

import polars as pl

W = "work"

# SPLITS=test rebuilds only the test side (after freq_fix.py)
SPLITS = os.environ.get("SPLITS", "train,test").split(",")
HI = 0.9


def main():
    for sp in SPLITS:
        p = (pl.scan_parquet(sorted(glob.glob(f"{W}/s2te_{sp}/part_*.parquet"))).select("s1_uid", "r_uid", "p1")
             .with_columns((pl.col("r_uid") // 10**10).cast(pl.Int8).alias("src")).collect(engine="streaming"))
        nul = pl.concat([pl.scan_parquet(f"{W}/norm/{sp}_s{s}.parquet").select(pl.col("uid").alias("r_uid"), pl.col("a_null").cast(pl.Int16).alias("rnull"))
                         .join(p.lazy().select("r_uid").unique(), on="r_uid", how="semi").collect() for s in (2, 3)])
        p = p.join(nul, on="r_uid", how="left").with_columns(pl.col("rnull").fill_null(0))
        conf = (pl.col("p1") >= HI).cast(pl.Int16)
        s1c = p.group_by("s1_uid", "src").agg(conf.sum().alias("n"), pl.col("p1").sum().alias("s"))
        # confident copies without an address per S1 and source
        s1n = (p.filter(pl.col("rnull") == 1).group_by("s1_uid", "src").agg(conf.sum().alias("nn"))
               .pivot(on="src", index="s1_uid", values="nn").rename({"2": "nn2", "3": "nn3"}, strict=False))
        for c in ("nn2", "nn3"):
            if c not in s1n.columns:
                s1n = s1n.with_columns(pl.lit(0).alias(c))
        wide = (s1c.pivot(on="src", index="s1_uid", values=["n", "s"])
                .rename({"n_2": "n2", "n_3": "n3", "s_2": "s2", "s_3": "s3"}, strict=False))
        for c in ("n2", "n3", "s2", "s3"):
            if c not in wide.columns:
                wide = wide.with_columns(pl.lit(0).alias(c))
        wide = wide.fill_null(0)
        q = p.join(wide, on="s1_uid", how="left").join(s1n, on="s1_uid", how="left").with_columns(
            pl.col("nn2", "nn3").fill_null(0)).with_columns(
            (pl.when(pl.col("src") == 2).then(pl.col("nn2")).otherwise(pl.col("nn3")) - conf * pl.col("rnull")).alias("cc_null_same"),
            pl.when(pl.col("src") == 2).then(pl.col("n2")).otherwise(pl.col("n3")).alias("_ns"),
            pl.when(pl.col("src") == 2).then(pl.col("n3")).otherwise(pl.col("n2")).alias("cc_other"),
            pl.when(pl.col("src") == 2).then(pl.col("s2")).otherwise(pl.col("s3")).alias("_ss"),
        ).with_columns(
            (pl.col("_ns") - conf).alias("cc_same"),                    # other confident copies in the record's source
            (pl.col("_ss") - pl.col("p1")).alias("cs_same"),            # expected other copies in that source
        )
        # the same counts for the record's best other S1
        q = q.with_columns(pl.col("p1").rank("ordinal", descending=True).over("r_uid").alias("_rk"))
        best = q.filter(pl.col("_rk") <= 2).select("r_uid", "s1_uid", "_rk", "cc_same", "cs_same")
        top1 = best.filter(pl.col("_rk") == 1).select("r_uid", pl.col("s1_uid").alias("_b1"), pl.col("cc_same").alias("_c1"), pl.col("cs_same").alias("_s1"))
        top2 = best.filter(pl.col("_rk") == 2).select("r_uid", pl.col("cc_same").alias("_c2"), pl.col("cs_same").alias("_s2"))
        q = (q.join(top1, on="r_uid", how="left").join(top2, on="r_uid", how="left")
             .with_columns(pl.when(pl.col("s1_uid") == pl.col("_b1")).then(pl.col("_c2")).otherwise(pl.col("_c1")).alias("cc_comp_same"),
                           pl.when(pl.col("s1_uid") == pl.col("_b1")).then(pl.col("_s2")).otherwise(pl.col("_s1")).alias("cs_comp_same"))
             .with_columns((pl.col("cc_same") - pl.col("cc_comp_same")).alias("cc_diff"),
                           (pl.col("cs_same") - pl.col("cs_comp_same")).alias("cs_diff"))
             .select("s1_uid", "r_uid", *[pl.col(c).cast(pl.Float32) for c in
                                         ("cc_same", "cc_other", "cs_same", "cc_comp_same", "cs_comp_same", "cc_diff", "cs_diff", "cc_null_same")]))
        x = pl.read_parquet(f"{W}/xside_{sp}.parquet").join(q, on=["s1_uid", "r_uid"], how="left")
        x.write_parquet(f"{W}/xside_cc_{sp}.parquet")
        print(f"{sp}: {q.height:,} pairs with copy counts; side table {x.height:,} rows, cc coverage "
              f"{x['cc_same'].is_not_null().mean():.3f}", flush=True)
        del p, q, x


if __name__ == "__main__":
    main()
