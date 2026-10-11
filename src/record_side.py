"""Second-pass retrieval for addressed records left unassigned by the main selection.

Some true copies (typo in the name, cut-down address) are not among their entity's blocking candidates. Each unassigned
record is paired with the S1 entities that share a rare key with it: house number, house code ("A-45") or name word
combined with an address word, or two address words. The best TOP_K per record are scored by a LightGBM trained on the
training labels; picks above 0.85 (France 0.95) are added.

    python src/record_side.py build | train | apply --base <matching_results.tsv> --out <dir>
"""
import argparse
import glob
import json
import os
import re
import shutil

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from decide import exclusive, select_expected_f
from metric import score_pairs

W = "work"
DF_MAX, TOP_K, NPART = 20, 5, 16
SAMPLE = {"train": 0.35, "test": 1.0}   # share of unsure records processed (train: enough to fit / validate)
FEATS = ["n_tset", "n_ratio", "n_partial", "n_jw", "ns_ratio", "a_tset", "a_ratio", "hn_eq", "n_shared_num", "n_shared_word",
         "k_name", "len1", "len2", "src", "rank", "gap", "n_cand", "p1", "in_s2", "rec_p1max",
         "n_shared_code", "n_shared_name", "n_shared_wpair", "num_absdiff"]
STOPA = {"st", "rd", "av", "dr", "ln", "ct", "bd", "pl", "unit", "near", "opp", "flat", "no", "floor", "fl", "road", "street"}
# house codes such as "A-45" or "4/71" are rarer keys than the plain number
CODE_DROP = {"no", "h", "hno", "st", "nd", "rd", "th", "house", "flat", "plot", "door", "shop", "office", "unit", "apt", "ste",
             "suite", "bldg", "building", "block", "floor", "fl", "room", "sy", "survey", "gala", "ward", "sector", "sec", "phase",
             "ph", "road", "street"}
_TOK, _RUN = re.compile(r"[a-z0-9/\-\.#]+"), re.compile(r"[a-z]+|\d+")


def house_codes(addr):
    """Address tokens that keep letters / several numbers: 'A-45' -> 'a.45', '46/1106' -> '46.1106', '97 bis' stays plain."""
    out = set()
    for t in _TOK.findall((addr or "").lower()):
        if not any(ch.isdigit() for ch in t):
            continue
        runs = [(r.lstrip("0") or "0") if r.isdigit() else r for r in _RUN.findall(t)]
        runs = [r for r in runs if r not in CODE_DROP]
        if 2 <= len(runs) <= 6 and any(r.isdigit() for r in runs):
            out.add(".".join(runs))
    return sorted(out)


def keys(df, uid):
    """(country, number, address word) keys: first two numbers x address words of length >= 3 (not street types)."""
    return (df.select(pl.col(uid).alias("uid"), "country",
                      pl.col("a_nums").str.split(" ").list.head(2).alias("num"),
                      pl.col("a_clean").str.split(" ").list.eval(pl.element().filter((pl.element().str.len_chars() >= 3)
                                                                                     & ~pl.element().str.contains(r"^\d+$"))).list.unique().alias("w"))
            .explode("num").filter(pl.col("num").is_not_null() & (pl.col("num") != "")).explode("w").drop_nulls("w")
            .filter(~pl.col("w").is_in(list(STOPA))).unique())


def code_keys(df, uid):
    """(country, house code, address word) keys; the code takes the place of the number."""
    return (df.select(pl.col(uid).alias("uid"), "country",
                      pl.col("business_address").map_elements(house_codes, return_dtype=pl.List(pl.String)).alias("num"),
                      pl.col("a_clean").str.split(" ").list.eval(pl.element().filter((pl.element().str.len_chars() >= 3)
                                                                                     & ~pl.element().str.contains(r"^\d+$"))).list.unique().alias("w"))
            .filter(pl.col("num").list.len() > 0).explode("num").explode("w").drop_nulls("w")
            .filter(~pl.col("w").is_in(list(STOPA))).unique())


_AW = lambda: pl.col("a_clean").str.split(" ").list.eval(pl.element().filter((pl.element().str.len_chars() >= 3)
                                                                             & ~pl.element().str.contains(r"^\d+$") & ~pl.element().is_in(list(STOPA)))).list.unique()


def name_keys(df, uid):
    """(name token, address word) keys, the token prefixed 'N:' so it never meets a number."""
    return (df.select(pl.col(uid).alias("uid"), "country",
                      pl.col("n_clean").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.unique().alias("num"),
                      _AW().list.tail(4).alias("w"))   # truncated copies keep the town / state at the end
            .explode("num").explode("w").drop_nulls(["num", "w"]).with_columns(("N:" + pl.col("num")).alias("num")).unique())


def wpair_keys(df, uid):
    """(address word, address word) keys over the first 5 address words, prefixed 'W:'."""
    df = df.filter(pl.col("country") != "India")   # India: few misses reachable this way, key volume too large
    return (df.select(pl.col(uid).alias("uid"), "country", _AW().list.head(5).alias("num")).with_columns(pl.col("num").alias("w"))
            .explode("num").explode("w").drop_nulls(["num", "w"]).filter(pl.col("num") < pl.col("w"))
            .with_columns(("W:" + pl.col("num")).alias("num")).unique())


def families():
    return [keys, code_keys, name_keys, wpair_keys]


def s1_keys(s1):
    """Rare S1 keys (<= DF_MAX entities), built per country and key family to bound memory."""
    out = []
    for cty in s1["country"].unique().to_list():
        sc = s1.filter(pl.col("country") == cty)
        for fam in families():
            k = fam(sc, "uid")
            k = k.join(k.group_by("num", "w").len("df").filter(pl.col("df") <= DF_MAX).drop("df"), on=["num", "w"])
            out.append(k.select("country", "num", "w", pl.col("uid").alias("s1_uid")))
            del k
    return pl.concat(out)


def all_keys(df, uid):
    return pl.concat([f(df, uid) for f in families()]).unique()


def build_split(sp):
    lz = pl.scan_parquet(sorted(glob.glob(f"{W}/s2te_{sp}/part_*.parquet"))).select("s1_uid", "r_uid", "p1")
    rmax = lz.group_by("r_uid").agg(pl.col("p1").max().alias("rec_p1max")).collect(engine="streaming")
    cols = ["uid", "country", "n_clean", "a_clean", "a_nums", "a_null", "business_address"]
    rec = pl.concat([pl.read_parquet(f"{W}/norm/{sp}_s{s}.parquet", columns=cols) for s in (2, 3)]).filter(pl.col("a_null") == 0)
    rec = rec.join(rmax.rename({"r_uid": "uid"}), on="uid", how="left").with_columns(pl.col("rec_p1max").fill_null(0.0))
    rec = rec.filter(pl.col("rec_p1max") < 0.5)           # records the main pipeline is not confident about
    rec = rec.filter(((pl.col("uid") * 2654435761) % 1000) < int(SAMPLE[sp] * 1000))
    s1 = pl.read_parquet(f"{W}/norm/{sp}_s1.parquet", columns=["uid", "country", "n_clean", "a_clean", "a_nums", "business_address"])
    kname = s1.group_by("country", "n_clean").len("k_name")
    del rmax
    s2 = lz.join(rec.lazy().select(pl.col("uid").alias("r_uid")), on="r_uid", how="semi").select("s1_uid", "r_uid").collect(engine="streaming")
    s1t = s1.rename({"uid": "s1_uid", "n_clean": "n1", "a_clean": "a1", "a_nums": "u1"}).drop("country", "business_address")
    rt = rec.rename({"uid": "r_uid", "n_clean": "n2", "a_clean": "a2", "a_nums": "u2"}).drop("country", "a_null", "business_address")
    tmp = f"{W}/rside_{sp}_parts"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for cty, q in [(c, q) for c in sorted(s1["country"].unique().to_list()) for q in range(NPART)]:
        if q == 0:
            ks = None
            ks = s1_keys(s1.filter(pl.col("country") == cty))
        part = all_keys(rec.filter((pl.col("country") == cty) & ((pl.col("uid") % NPART) == q)), "uid").rename({"uid": "r_uid"})
        if part.height == 0:
            continue
        c = (part.join(ks, on=["country", "num", "w"]).group_by("r_uid", "s1_uid", "country")
             .agg(pl.col("num").n_unique().alias("n_shared_num"), pl.col("w").n_unique().alias("n_shared_word"),
                  pl.col("num").filter(pl.col("num").str.contains(".", literal=True)).n_unique().alias("n_shared_code"),
                  pl.col("num").filter(pl.col("num").str.starts_with("N:")).n_unique().alias("n_shared_name"),
                  pl.col("num").filter(pl.col("num").str.starts_with("W:")).n_unique().alias("n_shared_wpair")))
        c = c.join(s2.select("s1_uid", "r_uid"), on=["s1_uid", "r_uid"], how="anti")      # only pairs the pipeline never scored
        c = c.join(rt, on="r_uid").join(s1t, on="s1_uid")
        a1, a2 = c["a1"].fill_null("").to_list(), c["a2"].fill_null("").to_list()
        c = c.with_columns(pl.Series("a_tset", process.cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)))
        n1, n2 = c["n1"].fill_null("").to_list(), c["n2"].fill_null("").to_list()
        c = c.with_columns(pl.Series("n_tset", process.cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)))
        c = (c.with_columns((pl.col("a_tset") + pl.col("n_tset")).alias("_s"))
             .sort(["r_uid", "_s"], descending=[False, True]).group_by("r_uid", maintain_order=True).head(TOP_K).drop("_s"))
        n1, n2 = c["n1"].fill_null("").to_list(), c["n2"].fill_null("").to_list()
        a1, a2 = c["a1"].fill_null("").to_list(), c["a2"].fill_null("").to_list()
        ns1, ns2 = [x.replace(" ", "") for x in n1], [x.replace(" ", "") for x in n2]
        c = c.with_columns(
            pl.Series("n_ratio", process.cpdist(n1, n2, scorer=fuzz.ratio, workers=-1, dtype=np.float32)),
            pl.Series("n_partial", process.cpdist(n1, n2, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32)),
            pl.Series("n_jw", process.cpdist(n1, n2, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)),
            pl.Series("ns_ratio", process.cpdist(ns1, ns2, scorer=fuzz.ratio, workers=-1, dtype=np.float32)),
            pl.Series("a_ratio", process.cpdist(a1, a2, scorer=fuzz.ratio, workers=-1, dtype=np.float32)),
            (pl.col("u1").str.split(" ").list.first() == pl.col("u2").str.split(" ").list.first()).fill_null(False).cast(pl.Int8).alias("hn_eq"),
            (pl.col("u1").str.split(" ").list.first().cast(pl.Float64, strict=False)
             - pl.col("u2").str.split(" ").list.first().cast(pl.Float64, strict=False)).abs().cast(pl.Float32).alias("num_absdiff"),
            pl.col("n1").str.len_chars().alias("len1"), pl.col("n2").str.len_chars().alias("len2"),
            (pl.col("r_uid") // 10**10).cast(pl.Int8).alias("src"))
        c = c.with_columns((pl.col("a_tset") + pl.col("n_tset")).alias("_s")).with_columns(
            pl.col("_s").rank("ordinal", descending=True).over("r_uid").cast(pl.Float32).alias("rank"),
            (pl.col("_s") - pl.col("_s").top_k(2).min().over("r_uid")).alias("gap"),
            pl.len().over("r_uid").cast(pl.Float32).alias("n_cand")).drop("_s")
        c = (c.join(kname.rename({"n_clean": "n1"}), on=["country", "n1"], how="left")
             .with_columns(pl.lit(None, pl.Float32).alias("p1"), pl.lit(0, pl.Int8).alias("in_s2"), pl.col("k_name").cast(pl.Float32)))
        c.select("s1_uid", "r_uid", "country", *FEATS).write_parquet(f"{tmp}/{cty}_{q:02d}.parquet")   # parts on disk: bounded memory
        del n1, n2, a1, a2, ns1, ns2
        print(f"  {cty} part {q}: {c.height:,} pairs", flush=True)
    pl.scan_parquet(f"{tmp}/*.parquet").sink_parquet(f"{W}/rside_{sp}.parquet")
    shutil.rmtree(tmp)
    out = pl.scan_parquet(f"{W}/rside_{sp}.parquet").select(pl.len(), pl.col("r_uid").n_unique()).collect()
    print(f"{sp}: {out.item(0, 0):,} new pairs for {out.item(0, 1):,} unsure addressed records (of {rec.height:,})", flush=True)


def cmd_build(a):
    for sp in ("train", "test"):
        build_split(sp)


def cmd_train(a):
    d = pl.read_parquet(f"{W}/rside_train.parquet")
    gt = pl.read_parquet(f"{W}/parquet/gt_pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
    folds = pl.read_parquet(f"{W}/norm/folds.parquet")
    d = d.join(gt, on=["s1_uid", "r_uid"], how="left").with_columns(pl.col("y").fill_null(0)).join(folds, on="s1_uid")
    tr, va = d.filter(pl.col("fold") != 0), d.filter(pl.col("fold") == 0)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, metric="binary_logloss")
    dtr = lgb.Dataset(tr.select(FEATS).to_numpy().astype(np.float32), tr["y"].to_numpy())
    dva = lgb.Dataset(va.select(FEATS).to_numpy().astype(np.float32), va["y"].to_numpy(), reference=dtr)
    m = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    m.save_model(f"{W}/models/record_side.txt")
    from sklearn.metrics import roc_auc_score
    allp = d.select("s1_uid", "r_uid", "fold", "y").with_columns(pl.Series("q", m.predict(d.select(FEATS).to_numpy().astype(np.float32))))
    print(f"train {tr.height:,} / val {va.height:,} pairs; positive share {d['y'].mean():.4f}; val AUC "
          f"{roc_auc_score(va['y'], allp.filter(pl.col('fold') == 0)['q']):.4f}", flush=True)
    best = allp.sort("q", descending=True).group_by("r_uid").head(1)
    s1 = (pl.read_parquet(f"{W}/parquet/train_s1.parquet", columns=["uid", "country"]).rename({"uid": "s1_uid"})
          .join(folds.filter(pl.col("fold") == 0), on="s1_uid", how="semi").select("s1_uid", "country"))
    gv = gt.join(s1, on="s1_uid", how="semi").select("s1_uid", "r_uid")
    main = select_expected_f(exclusive(pl.read_parquet(f"{W}/models/valpred_s2_{a.tag}.parquet", columns=["s1_uid", "r_uid", "p"])), alpha=0.8)
    cand = best.filter(pl.col("fold") == 0).join(main.select("r_uid"), on="r_uid", how="anti")
    half = lambda df: df.filter(((pl.col("s1_uid") * 31) % 2) == 0)
    other = lambda df: df.filter(((pl.col("s1_uid") * 31) % 2) == 1)
    base_f = {h: score_pairs(f(main), gv.pipe(f), s1.pipe(f))[0] for h, f in (("A", half), ("B", other))}
    res = {}
    for tau in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98):
        add = cand.filter(pl.col("q") >= tau).select("s1_uid", "r_uid")
        sel = pl.concat([main.select("s1_uid", "r_uid"), add])
        res[tau] = {h: score_pairs(f(sel), gv.pipe(f), s1.pipe(f))[0] - base_f[h] for h, f in (("A", half), ("B", other))}
        res[tau]["n_add"] = add.height
        res[tau]["prec"] = float(add.join(gv, on=["s1_uid", "r_uid"], how="semi").height / max(add.height, 1))
        print(f"tau {tau}: added {add.height:,} (precision {res[tau]['prec']:.3f}); gain A {res[tau]['A']:+.5f}, B {res[tau]['B']:+.5f}", flush=True)
    tau_a = max(res, key=lambda t: res[t]["A"])
    tau_b = max(res, key=lambda t: res[t]["B"])
    honest = (res[tau_a]["B"] + res[tau_b]["A"]) / 2
    tau = max(res, key=lambda t: res[t]["A"] + res[t]["B"])
    print(f"honest gain estimate {honest:+.5f}; final threshold {tau}", flush=True)
    json.dump({"tau": tau, "honest_gain": honest, "res": {str(k): v for k, v in res.items()}}, open(f"{W}/models/record_side.json", "w"), indent=1)


def cmd_apply(a):
    """Add the record-side picks (records still unassigned in --base) above the threshold to the base selection."""
    m = lgb.Booster(model_file=f"{W}/models/record_side.txt")
    lz = pl.scan_parquet(f"{W}/rside_test.parquet")
    n, parts, B = lz.select(pl.len()).collect().item(), [], 2_000_000
    for off in range(0, n, B):   # batches: bounded memory; only pairs that could reach a threshold are kept
        d = lz.slice(off, B).collect()
        d = d.with_columns(pl.Series("q", m.predict(d.select(FEATS).to_numpy().astype(np.float32))))
        parts.append(d.filter(pl.col("q") >= 0.3).select("s1_uid", "r_uid", "country", "q", "num_absdiff"))
    d = pl.concat(parts)
    del parts
    # same changed-house-number odds as final_assemble.py
    odds = pl.DataFrame([{"country": c, "k": k, "m": v} for (c, k), v in
                         {("US", 1): 0.471, ("US", 2): 0.488, ("India", 1): 0.556, ("India", 2): 0.647,
                          ("France", 1): 0.05, ("France", 2): 0.326}.items()], schema={"country": pl.String, "k": pl.Int32, "m": pl.Float64})
    d = (d.with_columns(pl.when(pl.col("num_absdiff").is_between(1, 4)).then(1).when(pl.col("num_absdiff").is_between(5, 99)).then(2)
                        .otherwise(0).cast(pl.Int32).alias("k"))
         .join(odds, on=["country", "k"], how="left").with_columns(pl.col("m").fill_null(1.0))
         .with_columns((pl.col("q") * pl.col("m") / (pl.col("q") * pl.col("m") + 1 - pl.col("q"))).alias("q")).drop("k", "m"))
    best = d.sort("q", descending=True).group_by("r_uid").head(1)
    ids = pl.concat([pl.read_parquet(f"{W}/norm/test_s{s}.parquet", columns=["uid", "entity_id"]) for s in (1, 2, 3)])
    base = pl.read_csv(a.base, separator="\t", quote_char=None, infer_schema=False).with_columns(pl.col("matched_entity_ids").fill_null(""))
    assigned = (base.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
                .filter(pl.col("matched_entity_ids") != "").join(ids.rename({"uid": "r_uid", "entity_id": "matched_entity_ids"}), on="matched_entity_ids")
                .select("r_uid").unique())
    # France needs a higher score: a real-word name at the same address is often a neighbouring business
    add = (best.filter(pl.col("q") >= pl.when(pl.col("country") == "France").then(pl.lit(a.tau_fr)).otherwise(pl.lit(a.tau)))
           .join(assigned, on="r_uid", how="anti")
           .join(ids.rename({"uid": "s1_uid", "entity_id": "s1e"}), on="s1_uid").join(ids.rename({"uid": "r_uid", "entity_id": "re"}), on="r_uid"))
    print(f"threshold {a.tau}; added by country:", add.group_by("country").len().sort("country").rows(), flush=True)
    addl = add.group_by("s1e").agg(pl.col("re").alias("new"))
    out = (base.join(addl, left_on="source1_entity_id", right_on="s1e", how="left")
           .with_columns(pl.concat_list(pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element() != "")),
                                        pl.col("new").fill_null(pl.lit([], dtype=pl.List(pl.String)))).list.unique().list.sort().list.join(",")
                         .alias("matched_entity_ids")).select("source1_entity_id", "matched_entity_ids"))
    os.makedirs(a.out, exist_ok=True)
    out.write_csv(f"{a.out}/matching_results.tsv", separator="\t", quote_style="never")
    add.select("s1_uid", "r_uid", "q", "s1e", "re").write_parquet(f"{a.out}/added_pairs.parquet")
    print(f"written {a.out}/matching_results.tsv (+{add.height:,} pairs)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "train", "apply"])
    ap.add_argument("--base", default="work/final_b/matching_results.tsv", help="apply: selection to add picks to")
    ap.add_argument("--out", default="work/final", help="apply: output folder")
    ap.add_argument("--tau", type=float, default=0.85, help="apply: pick threshold (US / India)")
    ap.add_argument("--tau-fr", type=float, default=0.95, help="apply: pick threshold (France)")
    ap.add_argument("--tag", default="main", help="train: stacker whose validation selection the picks are added to")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
