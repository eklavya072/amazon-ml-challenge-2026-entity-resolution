"""Raw-spelling features (rw_*, lg_*).

Normalisation maps "Moque L.L.C.", "Moque LLC" and "Moque" to the same tokens, but a copy keeps the raw spelling of its
own S1. Names are folded (accents removed, lower case) with punctuation and legal-form spelling kept, then compared:
edit similarity, exact match, token overlap, rank and gap against the record's other candidates, and whether the raw
legal-form signature matches. Reads work/xside_cc_{split}, writes work/xside_rl_{split}.
"""
import glob
import re
import unicodedata

import numpy as np
import os

import polars as pl
from rapidfuzz import fuzz, process

W = "work"

# SPLITS=test rebuilds only the test side (after freq_fix.py)
SPLITS = os.environ.get("SPLITS", "train,test").split(",")
OUT = "xside_rl"   # side table read by stacker.py
_WS = re.compile(r"\s+")
# legal forms as written, punctuation kept: "l.l.c.", "llc", ", inc", "s.a.r.l."
_LEG = re.compile(r"(?<![a-z0-9])(,?\s*(?:p\.?\s?l\.?\s?l\.?\s?c\.?|l\.?\s?l\.?\s?c\.?|l\.?\s?l\.?\s?p\.?|l\.?\s?p\.?|p\.?\s?c\.?|"
                  r"p\.?\s?a\.?|p\.?\s?l\.?\s?c\.?|inc\.?|incorporated|corp\.?|corporation|co\.?|company|ltd\.?|limited|pvt\.?|"
                  r"private|s\.?\s?a\.?\s?r\.?\s?l\.?|s\.?\s?a\.?\s?s\.?|e\.?\s?u\.?\s?r\.?\s?l\.?|e\.?\s?i\.?|s\.?\s?a\.?|"
                  r"s\.?\s?c\.?\s?i\.?|gmbh))(?![a-z0-9])")


def fold(s):
    s = "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c))
    return _WS.sub(" ", s.lower()).strip()


def legal_sig(f):
    """Raw legal-form signature of a folded name, e.g. 'moque l.l.c.' -> 'l.l.c.'; ', inc' keeps its comma."""
    return "|".join(m.strip() for m in _LEG.findall(f))


def folded_names(split, uids):
    out = []
    for s in (1, 2, 3):
        d = (pl.scan_parquet(f"{W}/norm/{split}_s{s}.parquet").select("uid", "business_name")
             .join(uids.lazy(), on="uid", how="semi").collect())
        f = [fold(x) for x in d["business_name"].to_list()]
        out.append(d.select("uid", pl.Series("f", f), pl.Series("g", [legal_sig(x) for x in f])))
    return pl.concat(out)


def main():
    for sp in SPLITS:
        p = pl.scan_parquet(sorted(glob.glob(f"{W}/s2te_{sp}/part_*.parquet"))).select("s1_uid", "r_uid").collect(engine="streaming")
        uids = pl.concat([p.select(pl.col("s1_uid").alias("uid")), p.select(pl.col("r_uid").alias("uid"))]).unique()
        f = folded_names(sp, uids)
        q = (p.join(f.rename({"uid": "s1_uid", "f": "f1", "g": "g1"}), on="s1_uid", how="left")
              .join(f.rename({"uid": "r_uid", "f": "f2", "g": "g2"}), on="r_uid", how="left"))
        del f
        a, b = q["f1"].fill_null("").to_list(), q["f2"].fill_null("").to_list()
        ratio = process.cpdist(a, b, scorer=fuzz.ratio, workers=-1, dtype=np.float32)
        tok = process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        eq = np.fromiter((x == y and x != "" for x, y in zip(a, b)), bool, len(a))
        del a, b
        q = q.select("s1_uid", "r_uid",
                     (pl.col("g1").fill_null("") == pl.col("g2").fill_null("")).cast(pl.Int8).alias("lg_eq"),
                     (pl.col("g1").fill_null("") != "").cast(pl.Int8).alias("lg_s1_has"),
                     (pl.col("g2").fill_null("") != "").cast(pl.Int8).alias("lg_r_has")).with_columns(
            pl.Series("rw_ratio", ratio), pl.Series("rw_tok", tok), pl.Series("rw_eq", eq).cast(pl.Int8))
        q = q.with_columns(
            pl.col("rw_ratio").rank("min", descending=True).over("r_uid").cast(pl.Float32).alias("rw_rank_r"),
            pl.col("rw_ratio").max().over("r_uid").alias("_mx"),
            pl.col("rw_ratio").top_k(2).min().over("r_uid").alias("_sec"),
            pl.len().over("r_uid").alias("_n"),
            pl.col("rw_eq").sum().over("r_uid").cast(pl.Float32).alias("rw_neq_r"),
            pl.col("lg_eq").sum().over("r_uid").cast(pl.Float32).alias("lg_neq_r"),
        ).with_columns(
            pl.when(pl.col("_n") == 1).then(0.0).when(pl.col("rw_ratio") >= pl.col("_mx")).then(pl.col("_sec"))
            .otherwise(pl.col("_mx")).alias("rw_max_other_r")
        ).with_columns((pl.col("rw_ratio") - pl.col("rw_max_other_r")).alias("rw_gap_r")).drop("_mx", "_sec", "_n")
        x = pl.read_parquet(f"{W}/xside_cc_{sp}.parquet").join(q, on=["s1_uid", "r_uid"], how="left")
        x.write_parquet(f"{W}/{OUT}_{sp}.parquet")
        print(f"{sp}: {q.height:,} pairs; side table {x.height:,} rows; rw coverage {x['rw_ratio'].is_not_null().mean():.3f}", flush=True)
        del p, q, x


if __name__ == "__main__":
    main()
