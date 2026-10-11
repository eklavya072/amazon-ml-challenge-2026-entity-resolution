"""Stage-1 model and stage-2 features.

Stage 1 (LightGBM on the features.py columns) is trained twice, on folds 1-2 and on folds 3-4, so every training pair
is scored by the model that did not see it; validation and test pairs get one of the two models, picked per S1.
Stage 2 adds what a single pair cannot see: its rank among the S1's and the record's candidates, how similar the record
is to the S1's confident matches, whether its differences recur among them, and word track records (how often an
added or dropped word meant a true match in training).

    python src/stage2.py oof | p1 | ctx | group | te_build | te_apply | candidates
"""

import argparse
import glob
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from train import NON_FEATURES, PARAMS, pseudo_random, train_model

W = "work"

# SPLITS=test rebuilds only the test side (after freq_fix.py)
SPLITS = os.environ.get("SPLITS", "train,test").split(",")
MIN_P1 = 0.005          # stage 2 only scores pairs stage 1 does not rule out
ANCHOR_P1 = 0.3         # "confident" matches of an S1 used as group evidence
MAX_ANCHORS = 6


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parts_of(split):
    return sorted(glob.glob(f"{W}/features_{split}/part_*.parquet"))


def stage1_features():
    return [c for c in pl.read_parquet_schema(parts_of("train")[0]) if c not in NON_FEATURES]


def load_sample(parts, folds, keep_folds, frac, feats, gt):
    """Row sample of stage-1 features (one file at a time) with labels, as float32 matrix."""
    lab = gt.lazy().with_columns(pl.lit(1, pl.Int8).alias("y"))
    frames = []
    for f in parts:
        lf = (pl.scan_parquet(f).filter(pseudo_random(min(1.0, frac)))
              .join(folds.lazy().filter(pl.col("fold").is_in(keep_folds)), on="s1_uid", how="semi"))
        frames.append(lf.join(lab, on=["s1_uid", "r_uid"], how="left").with_columns(pl.col("y").fill_null(0))
                      .select(feats + ["y"]).collect())
    n = sum(fr.height for fr in frames)
    X, y, i = np.empty((n, len(feats)), np.float32), np.empty(n, np.int8), 0
    while frames:
        fr = frames.pop(0)
        X[i:i + fr.height] = fr.select(feats).to_numpy().astype(np.float32)
        y[i:i + fr.height] = fr["y"].to_numpy()
        i += fr.height
    return X, y


# --------------------------------------------------------------------------- oof
def cmd_oof(a):
    """Train stage-1 models A (folds 1,2) and B (folds 3,4) on ~6M sampled pairs each."""
    gt = pl.read_parquet(f"{W}/parquet/gt_pairs.parquet")
    folds = pl.read_parquet(f"{W}/norm/folds.parquet")
    feats = stage1_features()
    parts = parts_of("train")
    n_all = pl.scan_parquet(parts).select(pl.len()).collect().item()
    Xe, ye = load_sample(parts, folds, [0], 1_000_000 / (0.2 * n_all), feats, gt)
    for name, fs in (("A", [1, 2]), ("B", [3, 4])):
        X, y = load_sample(parts, folds, fs, a.rows / (0.4 * n_all), feats, gt)
        log(f"stage-1 model {name}: {X.shape}")
        m = train_model(X, y, Xe, ye, PARAMS, 3000, log_every=200)
        m.save_model(f"{W}/models/s1_{name}.txt")
        del X, y
    json.dump({"features": feats}, open(f"{W}/models/s1_meta.json", "w"))


# --------------------------------------------------------------------------- p1
def cmd_p1(a):
    """Stage-1 score p1 for every pair; train folds get the model that did not see them."""
    feats = json.load(open(f"{W}/models/s1_meta.json"))["features"]
    mA = lgb.Booster(model_file=f"{W}/models/s1_A.txt")
    mB = lgb.Booster(model_file=f"{W}/models/s1_B.txt")
    folds = pl.read_parquet(f"{W}/norm/folds.parquet")
    for split in SPLITS:
        os.makedirs(f"{W}/p1_{split}", exist_ok=True)
        for f in parts_of(split):
            out = f"{W}/p1_{split}/{os.path.basename(f)}"
            if os.path.exists(out):
                continue
            df = pl.read_parquet(f, columns=["s1_uid", "r_uid"] + feats)
            X = df.select(feats).to_numpy().astype(np.float32)
            pa, pb = mA.predict(X, num_threads=8), mB.predict(X, num_threads=8)
            ids = df.select("s1_uid", "r_uid")
            # validation / test pairs: one of the two models, picked per S1
            coin = ((ids["s1_uid"].to_numpy() * 7) % 2 == 0)
            p_single = np.where(coin, pa, pb)
            if split == "train":
                fold = ids.join(folds, on="s1_uid", how="left")["fold"].to_numpy()
                p1 = np.where(np.isin(fold, [1, 2]), pb, np.where(np.isin(fold, [3, 4]), pa, p_single))
            else:
                p1 = p_single
            ids.with_columns(pl.Series("p1", p1.astype(np.float32))).write_parquet(out)
        log(f"p1 done for {split}")


# --------------------------------------------------------------------------- ctx
def cmd_ctx(a):
    """Record-side context of p1 over ALL candidate pairs (per S2/S3 record), in 4 record buckets."""
    for split in SPLITS:
        out_dir = f"{W}/rctx_{split}"
        os.makedirs(out_dir, exist_ok=True)
        src = f"{W}/p1_{split}/*.parquet"
        for q in range(4):
            c = (pl.scan_parquet(src).filter((pl.col("r_uid") % 4) == q).collect(engine="streaming"))
            c = c.with_columns(
                pl.col("p1").rank("ordinal", descending=True).over("r_uid").cast(pl.Int16).alias("p1_rank_r"),
                pl.len().over("r_uid").cast(pl.Int16).alias("p1_ncand_r"),
                (pl.col("p1") >= 0.5).sum().over("r_uid").cast(pl.Int16).alias("p1_n50_r"),
                pl.col("p1").sum().over("r_uid").alias("p1_sum_r"),
                pl.col("p1").max().over("r_uid").alias("_max"),
                pl.col("p1").top_k(2).min().over("r_uid").alias("_second"),
            ).with_columns(
                # best score among the record's OTHER S1 options
                pl.when(pl.col("p1") >= pl.col("_max")).then(pl.col("_second")).otherwise(pl.col("_max"))
                .alias("p1_max_other_r")
            ).with_columns(
                pl.when(pl.col("p1_ncand_r") == 1).then(0.0).otherwise(pl.col("p1_max_other_r"))
                .alias("p1_max_other_r")
            ).drop("_max", "_second", "p1")
            c.write_parquet(f"{out_dir}/q{q}.parquet")
            del c
        log(f"record context done for {split}")


def group_features(p, strings):
    """S1 context of p1 and similarity of each pair's record to the S1's confident matches."""
    p = p.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("s1_uid").cast(pl.Int16).alias("p1_rank_s1"),
        pl.col("p1").max().over("s1_uid").alias("p1_max_s1"),
        pl.col("p1").sum().over("s1_uid").alias("p1_sum_s1"),
        (pl.col("p1") >= 0.5).sum().over("s1_uid").cast(pl.Int16).alias("p1_n50_s1"),
        (pl.col("p1") >= 0.9).sum().over("s1_uid").cast(pl.Int16).alias("p1_n90_s1"),
    ).with_columns((pl.col("p1_max_s1") - pl.col("p1")).alias("p1_gap_s1"))
    anchors = (p.filter(pl.col("p1") >= ANCHOR_P1)
               .sort("p1", descending=True).group_by("s1_uid").head(MAX_ANCHORS)
               .select("s1_uid", pl.col("r_uid").alias("a_uid"), pl.col("p1").alias("a_p1")))
    ev = p.filter(pl.col("p1") >= MIN_P1).select("s1_uid", "r_uid")
    pairs = ev.join(anchors, on="s1_uid").filter(pl.col("r_uid") != pl.col("a_uid"))
    pairs = (pairs.join(strings.rename({"uid": "r_uid", "n_clean": "rn", "a_clean": "ra", "src": "rs"}), on="r_uid")
                  .join(strings.rename({"uid": "a_uid", "n_clean": "an", "a_clean": "aa", "src": "as_"}), on="a_uid"))
    rn, an = pairs["rn"].to_list(), pairs["an"].to_list()
    ra, aa = pairs["ra"].to_list(), pairs["aa"].to_list()
    nt = process.cpdist(rn, an, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    nr = process.cpdist(rn, an, scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    at = process.cpdist(ra, aa, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    empty_a = np.fromiter(((not x) or (not y) for x, y in zip(ra, aa)), bool, len(ra))
    at[empty_a] = np.nan
    pairs = pairs.select("s1_uid", "r_uid", "a_p1", (pl.col("rs") == pl.col("as_")).alias("same_src")).with_columns(
        pl.Series("g_nt", nt), pl.Series("g_nr", nr), pl.Series("g_at", at))
    pairs = pairs.with_columns(
        pl.max_horizontal(pl.col("g_nt"), pl.col("g_at").fill_nan(0)).alias("g_best"),
        ((pl.col("g_nt") >= 90) | (pl.col("g_at").fill_nan(0) >= 90)).alias("g_twin"))
    g = pairs.group_by("s1_uid", "r_uid").agg(
        pl.col("g_nt").max().alias("g_nt_max"), pl.col("g_nr").max().alias("g_nr_max"),
        pl.col("g_at").fill_nan(None).max().alias("g_at_max"),
        pl.col("g_best").max().alias("g_best_max"),
        pl.col("g_twin").sum().cast(pl.Int16).alias("g_n_twins"),
        (pl.col("g_best") * pl.col("a_p1")).sum().alias("_wsum"), pl.col("a_p1").sum().alias("_w"),
        pl.col("a_p1").get(pl.col("g_best").arg_max()).alias("g_p1_of_most_similar"),
        pl.len().cast(pl.Int16).alias("g_n_anchors"),
        (pl.col("same_src") & pl.col("g_twin")).sum().cast(pl.Int16).alias("g_twins_same_src"),
    ).with_columns((pl.col("_wsum") / pl.col("_w")).alias("g_wsim")).drop("_wsum", "_w")
    return p.join(g, on=["s1_uid", "r_uid"], how="left")


def deviation_features(p, s1_strings, r_strings):
    """Do the record's differences from its S1 (extra name words, other numbers) recur among the S1's other candidates?"""
    tok = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    q = (p.select("s1_uid", "r_uid", "p1")
         .join(s1_strings.rename({"uid": "s1_uid", "n_clean": "n1", "a_nums": "u1"}), on="s1_uid")
         .join(r_strings.select(pl.col("uid").alias("r_uid"), pl.col("n_clean").alias("n2"),
                                pl.col("a_nums").alias("u2")), on="r_uid")
         .with_columns(pl.concat_list(
             tok("n2").list.set_difference(tok("n1")).list.eval("w:" + pl.element()),
             tok("u2").list.set_difference(tok("u1")).list.eval("n:" + pl.element())).alias("dev")))
    d = q.select("s1_uid", "r_uid", "p1", "dev").explode("dev").drop_nulls("dev")
    sup = (d.join(d.select("s1_uid", pl.col("r_uid").alias("o_uid"), "dev", pl.col("p1").alias("o_p1")),
                  on=["s1_uid", "dev"])
           .filter(pl.col("o_uid") != pl.col("r_uid"))
           .group_by("s1_uid", "r_uid", "dev").agg(pl.len().alias("n_share"), pl.col("o_p1").max().alias("share_p1")))
    agg = sup.group_by("s1_uid", "r_uid").agg(
        pl.col("n_share").max().cast(pl.Int16).alias("dev_support_max"),
        pl.col("share_p1").max().alias("dev_share_p1_max"),
        pl.col("share_p1").mean().alias("dev_share_p1_mean"),
        pl.len().cast(pl.Int16).alias("dev_n_shared"))
    q = q.select("s1_uid", "r_uid", pl.col("dev").list.len().cast(pl.Int16).alias("dev_n")).join(
        agg, on=["s1_uid", "r_uid"], how="left").with_columns(pl.col("dev_support_max", "dev_n_shared").fill_null(0))
    return p.join(q, on=["s1_uid", "r_uid"], how="left")


TE_SMOOTH = 20.0


def diff_words(pairs, s1_strings, r_strings):
    """Per pair: words the record ADDS to / DROPS from the S1 name and address (as tagged tokens)."""
    tok = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    q = (pairs.join(s1_strings.rename({"uid": "s1_uid", "n_clean": "n1", "a_clean": "a1"}), on="s1_uid")
         .join(r_strings.select(pl.col("uid").alias("r_uid"), pl.col("n_clean").alias("n2"), pl.col("a_clean").alias("a2")),
               on="r_uid"))
    alpha = lambda c: tok(c).list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$")))
    return q.select(
        "s1_uid", "r_uid",
        pl.concat_list(tok("n2").list.set_difference(tok("n1")).list.eval("na:" + pl.element()),
                       tok("n1").list.set_difference(tok("n2")).list.eval("nd:" + pl.element()),
                       alpha("a2").list.set_difference(alpha("a1")).list.eval("aa:" + pl.element()),
                       alpha("a1").list.set_difference(alpha("a2")).list.eval("ad:" + pl.element())).alias("dw"))


def load_strings(split, s1_ids, r_ids):
    cols = ["uid", "n_clean", "a_clean"]
    s1s = (pl.scan_parquet(f"{W}/norm/{split}_s1.parquet").select(cols)
           .join(s1_ids.lazy().rename({"s1_uid": "uid"}), on="uid", how="semi").collect(engine="streaming"))
    rs = (pl.concat([pl.scan_parquet(f"{W}/norm/{split}_s{s}.parquet").select(cols) for s in (2, 3)])
          .join(r_ids.lazy().rename({"r_uid": "uid"}), on="uid", how="semi").collect(engine="streaming"))
    return s1s, rs


def cmd_te_build(a):
    """Word track records: true / false counts of each added or dropped word, per fold half (A = 1-2, B = 3-4)."""
    gt = pl.read_parquet(f"{W}/parquet/gt_pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
    folds = pl.read_parquet(f"{W}/norm/folds.parquet")
    acc = []
    for f in parts_of("train"):
        p = (pl.read_parquet(f"{W}/s2_train/{os.path.basename(f)}", columns=["s1_uid", "r_uid"])
             .join(folds, on="s1_uid").filter(pl.col("fold") != 0))
        s1s, rs = load_strings("train", p.select("s1_uid").unique(), p.select("r_uid").unique())
        d = (diff_words(p.select("s1_uid", "r_uid"), s1s, rs).join(p.select("s1_uid", "r_uid", "fold"), on=["s1_uid", "r_uid"])
             .join(gt, on=["s1_uid", "r_uid"], how="left").with_columns(pl.col("y").fill_null(0))
             .with_columns(pl.when(pl.col("fold") <= 2).then(pl.lit("A")).otherwise(pl.lit("B")).alias("half"))
             .explode("dw").drop_nulls("dw"))
        acc.append(d.group_by("half", "dw").agg(pl.col("y").sum().alias("pos"), pl.len().alias("n")))
    t = pl.concat(acc).group_by("half", "dw").agg(pl.col("pos").sum(), pl.col("n").sum())
    t.write_parquet(f"{W}/te_words.parquet")
    log(f"word track-record table: {t.height:,} rows")


def te_features(p, s1_strings, r_strings, table, prior):
    """Smoothed log-odds of the record's added/dropped words -> min / mean / count per kind."""
    d = diff_words(p.select("s1_uid", "r_uid"), s1_strings, r_strings).explode("dw").drop_nulls("dw")
    d = d.join(table, on="dw", how="left").with_columns(
        (((pl.col("pos").fill_null(0) + TE_SMOOTH * prior) / (pl.col("n").fill_null(0) - pl.col("pos").fill_null(0)
          + TE_SMOOTH * (1 - prior))).log()).alias("te"),
        pl.col("dw").str.slice(0, 2).alias("kind"))
    out = p.select("s1_uid", "r_uid")
    for kind in ("na", "nd", "aa", "ad"):
        g = d.filter(pl.col("kind") == kind).group_by("s1_uid", "r_uid").agg(
            pl.col("te").min().alias(f"te_{kind}_min"), pl.col("te").mean().alias(f"te_{kind}_mean"),
            pl.len().cast(pl.Int16).alias(f"te_{kind}_n"))
        out = out.join(g, on=["s1_uid", "r_uid"], how="left")
    return p.join(out, on=["s1_uid", "r_uid"], how="left")


def cmd_te_apply(a):
    """Add the word track-record features (s2 -> s2te): folds 1-2 use half B, folds 3-4 half A, fold 0 and test both."""
    t = pl.read_parquet(f"{W}/te_words.parquet")
    tab = {"A": t.filter(pl.col("half") == "A").drop("half"), "B": t.filter(pl.col("half") == "B").drop("half"),
           "AB": t.group_by("dw").agg(pl.col("pos").sum(), pl.col("n").sum())}
    prior = t["pos"].sum() / t["n"].sum()
    folds = pl.read_parquet(f"{W}/norm/folds.parquet")
    for split in SPLITS:
        os.makedirs(f"{W}/s2te_{split}", exist_ok=True)
        for f in sorted(glob.glob(f"{W}/s2_{split}/part_*.parquet")):
            out = f"{W}/s2te_{split}/{os.path.basename(f)}"
            if os.path.exists(out):
                continue
            p = pl.read_parquet(f)
            s1s, rs = load_strings(split, p.select("s1_uid").unique(), p.select("r_uid").unique())
            if split == "train":
                p = p.join(folds, on="s1_uid")
                chunks = [te_features(p.filter(pl.col("fold").is_in(fs)), s1s, rs, tab[k], prior)
                          for fs, k in (([1, 2], "B"), ([3, 4], "A"), ([0], "AB"))]
                p = pl.concat([c.drop("fold") for c in chunks], how="diagonal_relaxed")
            else:
                p = te_features(p, s1s, rs, tab["AB"], prior)
            p.write_parquet(out, compression="zstd")
        log(f"word track-record features done for {split}")


def cmd_group(a):
    """Build stage-2 feature parts: stage-1 features + p1 + S1/record context + group."""
    for split in SPLITS:
        out_dir = f"{W}/s2_{split}"
        os.makedirs(out_dir, exist_ok=True)
        rctx = f"{W}/rctx_{split}/*.parquet"
        norm = [f"{W}/norm/{split}_s{s}.parquet" for s in (2, 3)]
        for f in parts_of(split):
            out = f"{out_dir}/{os.path.basename(f)}"
            if os.path.exists(out):
                continue
            base = pl.read_parquet(f).join(pl.read_parquet(f"{W}/p1_{split}/{os.path.basename(f)}"),
                                           on=["s1_uid", "r_uid"])
            ids = base.select("r_uid").unique()
            strings = (pl.concat([pl.scan_parquet(q).select("uid", "src", "n_clean", "a_clean", "a_nums")
                                  for q in norm])
                       .join(ids.lazy().rename({"r_uid": "uid"}), on="uid", how="semi").collect(engine="streaming"))
            s1_strings = (pl.scan_parquet(f"{W}/norm/{split}_s1.parquet").select("uid", "n_clean", "a_nums")
                          .join(base.lazy().select(pl.col("s1_uid").alias("uid")).unique(), on="uid", how="semi")
                          .collect(engine="streaming"))
            p = group_features(base, strings.drop("a_nums"))
            p = p.filter(pl.col("p1") >= MIN_P1)
            p = deviation_features(p, s1_strings, strings)
            p = p.join(pl.scan_parquet(rctx).join(p.lazy().select("s1_uid", "r_uid"), on=["s1_uid", "r_uid"],
                                                  how="semi").collect(engine="streaming"),
                       on=["s1_uid", "r_uid"], how="left")
            p.write_parquet(out, compression="zstd")
            del base, strings, p
        log(f"stage-2 features done for {split}")


def cmd_candidates(a):
    """candidate_pairs.tsv = input of the FIRST scoring model (stage 1) = every blocking candidate kept for features."""
    from decide import full_rows, id_lists, write_tsv
    s1_ids = pl.read_parquet(f"{W}/norm/test_s1.parquet", columns=["uid", "entity_id"]).rename({"uid": "s1_uid"})
    r_ids = pl.concat([pl.read_parquet(f"{W}/norm/test_s{s}.parquet", columns=["uid", "entity_id"])
                       for s in (2, 3)]).rename({"uid": "r_uid", "entity_id": "r_entity_id"})
    cand = [id_lists(pl.read_parquet(f, columns=["s1_uid", "r_uid"]), r_ids, "candidate_entity_ids")
            for f in parts_of("test")]
    os.makedirs(a.out_dir, exist_ok=True)
    write_tsv(full_rows(pl.concat(cand), s1_ids, "candidate_entity_ids"), f"{a.out_dir}/candidate_pairs.tsv")
    log(f"candidate_pairs.tsv written to {a.out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["oof", "p1", "ctx", "group", "te_build", "te_apply", "candidates"])
    ap.add_argument("--rows", type=int, default=6_000_000, help="training rows per stage-1 model (oof)")
    ap.add_argument("--out-dir", default="work/candidates", help="where candidates writes candidate_pairs.tsv")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
