"""Final decision on the test set.

1. Probabilities from the main stacker (US, India) and the France stacker (without the English-only cross-encoder).
2. Same-name pairs whose house number differs by 1-99 are more frequent in the test than in validation; their odds are
   scaled by the ratio of the two densities (ODDS).
3. France: records that add a decoy word (groupe, holding, ...) or swap one common name word for another are scaled
   down, records that only add a copy suffix (fils, associes, cie, ...) are scaled up.
4. Each record is kept for its best S1, then the expected-F0.5 selection.

    python src/final_assemble.py --out work/final_a
"""
import argparse
import glob
import os

import polars as pl
from rapidfuzz import fuzz

from decide import exclusive, full_rows, id_lists, select_expected_f, write_tsv

W = "work"
ODDS = {("US", 1): 0.471, ("US", 2): 0.488, ("India", 1): 0.556, ("India", 2): 0.647, ("France", 1): 0.05, ("France", 2): 0.326}
FR_DECOY = ["groupe", "developpement", "france", "holding", "international", "participations", "distribution"]
DECOY_SCALE = 0.1
VOCAB_MIN = 30          # a "common" France name word is used by >= 30 France S1 names
# French copy suffixes (centre, compagnie, service and sportive act as category words instead)
SAFE = {"fils", "cie", "associes", "services", "frs", "freres", "partners"}
SAFE_ODDS = 20.0


def tok(c):
    return pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()


def safe_suffix_pairs(pairs, s1, r_names):
    """France same-address pairs whose record only adds French copy suffixes (and drops at most one name word)."""
    f = (pl.scan_parquet(sorted(glob.glob(f"{W}/s2te_test/part_*.parquet"))).select("s1_uid", "r_uid", "hn_eq", "a_tset")
         .join(pairs.lazy(), on=["s1_uid", "r_uid"], how="semi").collect(engine="streaming")
         .filter((pl.col("hn_eq") == 1) & (pl.col("a_tset") >= 85)))
    d = (f.join(s1.select("s1_uid", pl.col("n_clean").alias("n1")), on="s1_uid").join(r_names, on="r_uid")
         .with_columns(tok("n2").list.set_difference(tok("n1")).alias("add"), tok("n1").list.set_difference(tok("n2")).alias("drop")))
    return (d.filter((pl.col("add").list.len() >= 1) & (pl.col("drop").list.len() <= 1)
                     & pl.col("add").list.eval(pl.element().is_in(list(SAFE))).list.all())
            .select("s1_uid", "r_uid"))


def vocab_swap_pairs(pairs, s1, r_names):
    """France pairs where the record replaces a common name word with another one (Club -> Amicale)."""
    fr = s1.filter(pl.col("country") == "France")
    vocab = (fr.select(pl.col("n_clean").str.split(" ").list.unique().alias("w")).explode("w")
             .filter(pl.col("w").str.len_chars() >= 3)
             .group_by("w").len().filter(pl.col("len") >= VOCAB_MIN)["w"].to_list())
    vocab = {w for w in vocab if w.isalpha() and len(w) >= 3}
    d = (pairs.join(fr.select("s1_uid", pl.col("n_clean").alias("n1")), on="s1_uid").join(r_names, on="r_uid")
         .with_columns(tok("n2").list.set_difference(tok("n1")).alias("add"), tok("n1").list.set_difference(tok("n2")).alias("drop"))
         .filter((pl.col("add").list.len() > 0) & (pl.col("drop").list.len() > 0)))
    flag = []
    for add, drop in zip(d["add"].to_list(), d["drop"].to_list()):
        flag.append(any(o in vocab for o in drop) and any(
            w in vocab and w not in SAFE and all(fuzz.ratio(w, o) < 60 and not w.startswith(o) and not o.startswith(w) for o in drop)
            for w in add))
    return d.filter(pl.Series(flag, dtype=pl.Boolean)).select("s1_uid", "r_uid")


def scale(p, pairs, factor=None, odds=None):
    """Multiply p (or its odds) by a factor on the given pairs."""
    p = p.join(pairs.with_columns(pl.lit(True).alias("_hit")), on=["s1_uid", "r_uid"], how="left")
    new = pl.col("p") * factor if factor is not None else pl.col("p") * odds / (pl.col("p") * odds + 1 - pl.col("p"))
    return p.with_columns(pl.when(pl.col("_hit").fill_null(False)).then(new).otherwise(pl.col("p")).alias("p")).drop("_hit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="main", help="stacker predictions for US / India (work/models/testpred_s2_<tag>)")
    ap.add_argument("--france", default="france", help="stacker predictions for France")
    ap.add_argument("--out", default="work/final_a")
    a = ap.parse_args()
    s1 = pl.read_parquet(f"{W}/norm/test_s1.parquet", columns=["uid", "entity_id", "country", "n_clean"]).rename({"uid": "s1_uid"})
    p = pl.read_parquet(f"{W}/models/testpred_s2_{a.base}.parquet", columns=["s1_uid", "r_uid", "p"]).join(s1.select("s1_uid", "country"), on="s1_uid")
    fr = pl.read_parquet(f"{W}/models/testpred_s2_{a.france}.parquet", columns=["s1_uid", "r_uid", "p"]).rename({"p": "pf"})
    p = p.join(fr, on=["s1_uid", "r_uid"], how="left").with_columns(
        pl.when(pl.col("country") == "France").then(pl.col("pf").fill_null(pl.col("p"))).otherwise(pl.col("p")).alias("p")).drop("pf")

    # 2. changed-house-number decoys: k = 1 (off by 1-4), 2 (off by 5-99)
    f = (pl.scan_parquet(sorted(glob.glob(f"{W}/s2te_test/part_*.parquet")))
         .select("s1_uid", "r_uid", "hn_eq", "a_null2", "num_n2", "num_mindiff", "hn_in_r").collect(engine="streaming"))
    num = (pl.col("a_null2") != 1) & (pl.col("hn_eq") != 1) & (pl.col("num_n2") > 0) & (pl.col("hn_in_r") != 1)
    f = f.select("s1_uid", "r_uid", pl.when(num & (pl.col("num_mindiff") <= 4)).then(1).when(num & (pl.col("num_mindiff") <= 99)).then(2)
                 .otherwise(0).alias("k"))
    m = pl.DataFrame([{"country": c, "k": k, "m": v} for (c, k), v in ODDS.items()], schema={"country": pl.String, "k": pl.Int32, "m": pl.Float64})
    p = (p.join(f, on=["s1_uid", "r_uid"], how="left").with_columns(pl.col("k").fill_null(0).cast(pl.Int32))
         .join(m, on=["country", "k"], how="left").with_columns(pl.col("m").fill_null(1.0))
         .with_columns((pl.col("p") * pl.col("m") / (pl.col("p") * pl.col("m") + 1 - pl.col("p"))).alias("p")).drop("k", "m"))

    # 3. France rules
    frp = p.filter(pl.col("country") == "France").select("s1_uid", "r_uid")
    r = pl.concat([pl.scan_parquet(f"{W}/norm/test_s{s}.parquet").select(pl.col("uid").alias("r_uid"), pl.col("n_clean").alias("n2"))
                   .join(frp.lazy().select("r_uid").unique(), on="r_uid", how="semi").collect() for s in (2, 3)])
    words = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    dec = (frp.join(s1.select("s1_uid", pl.col("n_clean").alias("n1")), on="s1_uid").join(r, on="r_uid")
           .filter(words("n2").list.set_difference(words("n1")).list.eval(pl.element().is_in(FR_DECOY)).list.any())
           .select("s1_uid", "r_uid"))
    p = scale(p, dec, factor=DECOY_SCALE)
    vs = vocab_swap_pairs(frp.join(p.filter(pl.col("p") > 0.02).select("s1_uid", "r_uid"), on=["s1_uid", "r_uid"], how="semi"), s1, r)
    p = scale(p, vs, factor=DECOY_SCALE)
    sf = safe_suffix_pairs(frp.join(p.filter(pl.col("p") > 0.05).select("s1_uid", "r_uid"), on=["s1_uid", "r_uid"], how="semi"), s1, r)
    p = scale(p, sf, odds=SAFE_ODDS)
    print(f"France: decoy-word pairs {dec.height:,}, word swaps {vs.height:,}, copy-suffix pairs {sf.height:,}", flush=True)

    # 4. selection
    sel = select_expected_f(exclusive(p.select("s1_uid", "r_uid", "p")), alpha=0.8)
    print("matches by country:", sel.join(s1.select("s1_uid", "country"), on="s1_uid").group_by("country").len().sort("country").rows(),
          " total", sel.height, flush=True)
    r_ids = pl.concat([pl.read_parquet(f"{W}/norm/test_s{s}.parquet", columns=["uid", "entity_id"]) for s in (2, 3)]).rename(
        {"uid": "r_uid", "entity_id": "r_entity_id"})
    os.makedirs(a.out, exist_ok=True)
    write_tsv(full_rows(id_lists(sel, r_ids, "matched_entity_ids"), s1.select("s1_uid", "entity_id"), "matched_entity_ids"),
              f"{a.out}/matching_results.tsv")
    print(f"written {a.out}/matching_results.tsv", flush=True)


if __name__ == "__main__":
    main()
