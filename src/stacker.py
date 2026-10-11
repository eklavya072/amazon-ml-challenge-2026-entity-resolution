"""Final stacker: stage-2 features, three cross-encoder scores, copy counts and raw-spelling features.

On 30% of the training rows that add a name word the word track record is hidden, so the model also learns to judge
added words from the cross-encoders. This matters for France, whose words are not in the training data.
Trained on folds 3-4, early stopping and validation on fold 0.

    python src/stacker.py prep
    python src/stacker.py train --tag main [--drop "xenc,xenc_*"]
    python src/stacker.py predict --tag main
"""
import argparse
import glob
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from decide import exclusive, select_expected_f
from metric import score_pairs
from train import NON_FEATURES, PARAMS, pseudo_random, train_model

W = "work"
SRCS = [("xenc", f"{W}/kaggle_out/xenc_scores.parquet"), ("xenc2", f"{W}/kaggle_out2/xenc2_scores.parquet"),
        ("xenc3", f"{W}/kaggle_out3/xenc3_scores.parquet")]
TE_NA = ["te_na_min", "te_na_mean"]   # track record of the words the record ADDS to the S1 name
SIDE = "xside_rl"                     # side table from prep, cc_features.py and rw_features.py
ROWS, MASK_TE = 4_500_000, 0.3        # training rows (folds 3-4), share of word-adding rows with hidden track record


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def group_cols(x, c):
    """Score + its S1-group view: rank among the S1's candidates, best of the S1, gap, count > 0.5."""
    return (x.with_columns(pl.col(c).rank("ordinal", descending=True).over("s1_uid").cast(pl.Float32).alias(f"{c}_rank_s1"),
                           pl.col(c).max().over("s1_uid").alias(f"{c}_max_s1"),
                           (pl.col(c) > 0.5).sum().over("s1_uid").cast(pl.Float32).alias(f"{c}_n50_s1"))
            .with_columns((pl.col(f"{c}_max_s1") - pl.col(c)).alias(f"{c}_gap_s1")))


def cmd_prep(a):
    """Cross-encoder scores (whichever of xenc / xenc2 / xenc3 exist) with their S1-group views -> xside_{split}."""
    srcs = [(c, p) for c, p in SRCS if glob.glob(p)]
    for sp in ("train", "test"):
        x = None
        for c, path in srcs:
            d = group_cols(pl.read_parquet(path).filter(pl.col("split") == sp)
                           .select("s1_uid", "r_uid", pl.col(c).cast(pl.Float32)).unique(["s1_uid", "r_uid"]), c)
            x = d if x is None else x.join(d, on=["s1_uid", "r_uid"], how="full", coalesce=True)
        x.write_parquet(f"{W}/xside_{sp}.parquet")
        log(f"{sp}: {x.height:,} scored pairs, sources {[c for c, _ in srcs]}")


def mask_rows(df, frac):
    """Hide the added-word track record on a fixed pseudo-random share of rows that add a word."""
    if frac <= 0:
        return df
    pick = (((pl.col("s1_uid") * 7919 + pl.col("r_uid") * 104729) % 997) < int(frac * 997)) & (pl.col("te_na_n") > 0)
    return df.with_columns([pl.when(pick).then(None).otherwise(pl.col(c)).alias(c) for c in TE_NA])


def features(drop):
    """Stage-2 parts of the training split and the feature list (stage-2 columns + side-table columns)."""
    parts = sorted(glob.glob(f"{W}/s2te_train/part_*.parquet"))
    dropped = lambda c: c in drop or any(d.endswith("*") and c.startswith(d[:-1]) for d in drop)
    base = [c for c in pl.read_parquet_schema(parts[0]) if c not in NON_FEATURES and not dropped(c)]
    return parts, base + [c for c in pl.read_parquet_schema(f"{W}/{SIDE}_train.parquet")
                          if c not in ("s1_uid", "r_uid") and not dropped(c)]


def predict_frame(m, d, feats):
    p = np.empty(d.height, np.float32)
    for i in range(0, d.height, 300_000):
        p[i:i + 300_000] = m.predict(d[i:i + 300_000].select(pl.col(feats).cast(pl.Float32)).to_numpy(), num_threads=8)
    return p


def part_frames(parts, folds, keep, frac, feats, gt, xe, mask_frac=0.0):
    """One stage-2 part at a time: ids, label y and features (+ side-table columns) for the given folds."""
    lab = gt.lazy().with_columns(pl.lit(1, pl.Int8).alias("y"))
    fk = folds.lazy().filter(pl.col("fold").is_in(keep))
    for f in parts:
        d = (pl.scan_parquet(f).filter(pseudo_random(min(1.0, frac))).join(fk, on="s1_uid", how="semi")
             .join(xe.lazy(), on=["s1_uid", "r_uid"], how="left")
             .join(lab, on=["s1_uid", "r_uid"], how="left").with_columns(pl.col("y").fill_null(0))
             .select(["s1_uid", "r_uid"] + feats + ["y"]).collect())
        yield mask_rows(d, mask_frac)


def matrix(parts, folds, keep, frac, feats, gt, xe, mask_frac):
    """float32 feature matrix built part by part into one pre-sized array (no 64-bit copy of the whole table)."""
    fk = folds.lazy().filter(pl.col("fold").is_in(keep))
    n = sum(pl.scan_parquet(f).select("s1_uid", "r_uid").filter(pseudo_random(min(1.0, frac)))
            .join(fk, on="s1_uid", how="semi").select(pl.len()).collect().item() for f in parts)
    X, y, i = np.empty((n, len(feats)), np.float32), np.empty(n, np.int8), 0
    for d in part_frames(parts, folds, keep, frac, feats, gt, xe, mask_frac):
        X[i:i + d.height] = d.select(pl.col(feats).cast(pl.Float32)).to_numpy()
        y[i:i + d.height] = d["y"].to_numpy()
        i += d.height
    return X[:i], y[:i]


def cmd_train(a):
    """Stacker on folds 3-4 (early stopping on fold 0), validation predictions for fold 0."""
    drop = set(a.drop.split(",")) if a.drop else set()
    parts, feats = features(drop)
    gt = pl.read_parquet(f"{W}/parquet/gt_pairs.parquet")
    folds = pl.read_parquet(f"{W}/norm/folds.parquet")
    xe = pl.read_parquet(f"{W}/{SIDE}_train.parquet")
    n_all = pl.scan_parquet(parts).select(pl.len()).collect().item()
    X, y = matrix(parts, folds, [3, 4], ROWS / (0.4 * n_all), feats, gt, xe, MASK_TE)
    Xe, ye = matrix(parts, folds, [0], 1_000_000 / (0.2 * n_all), feats, gt, xe, MASK_TE)
    log(f"stacker train {X.shape} (folds 3-4), early-stop {Xe.shape}, word track record hidden on {MASK_TE:.0%} "
        f"of word-adding rows")
    m = train_model(X, y, Xe, ye, PARAMS, 8000, log_every=200)
    del X, y, Xe, ye
    m.save_model(f"{W}/models/s2_{a.tag}.txt")
    v = pl.concat([d.select("s1_uid", "r_uid", "p1").with_columns(pl.Series("p", predict_frame(m, d, feats)))
                   for d in part_frames(parts, folds, [0], 1.0, feats, gt, xe)])
    v.write_parquet(f"{W}/models/valpred_s2_{a.tag}.parquet")
    s1 = (pl.read_parquet(f"{W}/parquet/train_s1.parquet", columns=["uid"]).rename({"uid": "s1_uid"})
          .join(folds.filter(pl.col("fold") == 0), on="s1_uid", how="semi"))
    score = score_pairs(select_expected_f(exclusive(v.select("s1_uid", "r_uid", "p")), alpha=0.8),
                        gt.join(s1, on="s1_uid", how="semi"), s1)[0]
    json.dump({"features": feats, "best_iter": m.best_iteration, "val_f05": score},
              open(f"{W}/models/s2_{a.tag}.json", "w"), indent=1)
    log(f"{a.tag}: validation F0.5 {score:.5f}")


def cmd_predict(a):
    """Test predictions -> work/models/testpred_s2_<tag>.parquet."""
    meta = json.load(open(f"{W}/models/s2_{a.tag}.json"))
    m = lgb.Booster(model_file=f"{W}/models/s2_{a.tag}.txt")
    xe = pl.read_parquet(f"{W}/{SIDE}_test.parquet")
    preds = []
    for f in sorted(glob.glob(f"{W}/s2te_test/part_*.parquet")):
        d = pl.read_parquet(f).join(xe, on=["s1_uid", "r_uid"], how="left")
        preds.append(d.select("s1_uid", "r_uid").with_columns(pl.Series("p", predict_frame(m, d, meta["features"]))))
    pl.concat(preds).write_parquet(f"{W}/models/testpred_s2_{a.tag}.parquet")
    log(f"test predictions written: testpred_s2_{a.tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "predict"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--drop", default="", help="comma-separated feature names (prefix* allowed) left out")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
