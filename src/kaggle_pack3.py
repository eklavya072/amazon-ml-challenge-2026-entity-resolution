"""Package for the third cross-encoder (kaggle/xenc3_kaggle.py); the texts come from kaggle_pack2.

xtrain.parquet  training pairs of folds 1-2: all negatives and a sample of positives
xscore.parquet  pairs to score: xenc or xenc2 between 0.01 and 0.99 (folds 0, 3, 4 and the test)
"""
import os

import polars as pl

W, OUT = "work", "work/kaggle_pack3"
N_TRAIN, BAND = 1_600_000, 0.01


def main():
    os.makedirs(OUT, exist_ok=True)
    tp = pl.read_parquet(f"{W}/kaggle_pack/train_pairs.parquet")
    neg = tp.filter(pl.col("y") == 0)
    pos = tp.filter(pl.col("y") == 1).sample(max(0, N_TRAIN - neg.height), seed=11)
    tr = pl.concat([neg, pos]).sample(fraction=1.0, shuffle=True, seed=12)
    tr.write_parquet(f"{OUT}/xtrain.parquet", compression="zstd")
    band = (((pl.col("xenc2") > BAND) & (pl.col("xenc2") < 1 - BAND))
            | ((pl.col("xenc") > BAND) & (pl.col("xenc") < 1 - BAND)))
    sc = pl.concat([pl.read_parquet(f"{W}/xside_{sp}.parquet", columns=["s1_uid", "r_uid", "xenc", "xenc2"])
                    .filter(band).select("s1_uid", "r_uid", pl.lit(sp).alias("split")) for sp in ("test", "train")])
    sc.write_parquet(f"{OUT}/xscore.parquet", compression="zstd")
    with open(f"{OUT}/dataset-metadata.json", "w") as f:
        f.write('{"title": "amlc-pack-e5", "id": "eklavya09/amlc-pack-e5", "licenses": [{"name": "other"}]}')
    print(f"train pairs {tr.height:,} (negative share {1 - tr['y'].mean():.3f}); pairs to score "
          f"{sc.group_by('split').len().sort('split').rows()}")


if __name__ == "__main__":
    main()
