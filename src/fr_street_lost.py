"""Remove France matches with the same name and house number but a different street.

A match is removed when none of the S1's street words (from the address part holding its house number) appear in the
record, allowing typos. In France these pairs behave like planted decoys; US / India address formats differ, so the
rule is France only.

    python src/fr_street_lost.py --inp <matching_results.tsv> --out <matching_results.tsv>
"""
import argparse
import glob
import os
import re

import polars as pl
from anyascii import anyascii
from rapidfuzz import fuzz

W = "work"
TYPE = {"rue", "avenue", "ave", "boulevard", "place", "allee", "impasse", "chemin", "route", "quai", "cours", "des", "del", "appartement",
        "etage", "batiment"}


def toks(s):
    return re.sub(r"[^a-z0-9 ]", " ", anyascii(s or "").lower().replace("-", " ")).split()


def lost(a1, a2, n1):
    if not a1 or not a2 or not n1:
        return False
    num = n1.split()[0]
    part = next((p for p in a1.split(",") if num in toks(p)), None)
    if part is None:
        return False
    w = [t for t in toks(part) if len(t) >= 4 and not t.isdigit() and t not in TYPE and not re.match(r"^\d", t)]
    t2 = [t for t in toks(a2) if len(t) >= 3]
    return bool(w) and all(max((fuzz.ratio(x, y) for y in t2), default=0) < 75 for x in w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s1 = (pl.read_parquet(f"{W}/norm/test_s1.parquet", columns=["uid", "entity_id", "country", "business_address", "a_nums"])
          .rename({"uid": "s1_uid", "entity_id": "source1_entity_id", "business_address": "a1", "a_nums": "u1"}))
    ids = pl.concat([pl.read_parquet(f"{W}/norm/test_s{s}.parquet", columns=["uid", "entity_id", "business_address"]) for s in (2, 3)]).rename(
        {"uid": "r_uid", "entity_id": "matched_entity_ids", "business_address": "a2"})
    m = pl.read_csv(a.inp, separator="\t", quote_char=None, infer_schema=False).with_columns(pl.col("matched_entity_ids").fill_null(""))
    pr = (m.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
          .join(s1.filter(pl.col("country") == "France"), on="source1_entity_id").join(ids, on="matched_entity_ids"))
    hn = (pl.scan_parquet(sorted(glob.glob(f"{W}/s2te_test/part_*.parquet"))).select("s1_uid", "r_uid", "hn_eq")
          .join(pr.lazy().select("s1_uid", "r_uid"), on=["s1_uid", "r_uid"], how="semi").collect(engine="streaming"))
    pr = pr.join(hn, on=["s1_uid", "r_uid"], how="left").filter(pl.col("hn_eq") == 1)
    flag = [lost(x, y, u) for x, y, u in zip(pr["a1"].to_list(), pr["a2"].to_list(), pr["u1"].to_list())]
    drop = pr.filter(pl.Series(flag)).select("source1_entity_id", "matched_entity_ids")
    print(f"France 'other street' matches removed: {drop.height:,}", flush=True)
    dl = drop.group_by("source1_entity_id").agg(pl.col("matched_entity_ids").alias("rm"))
    out = (m.join(dl, on="source1_entity_id", how="left")
           .with_columns(pl.when(pl.col("rm").is_null()).then(pl.col("matched_entity_ids"))
                         .otherwise(pl.col("matched_entity_ids").str.split(",").list.set_difference(pl.col("rm")).list.sort().list.join(","))
                         .alias("matched_entity_ids")).select("source1_entity_id", "matched_entity_ids"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.write_csv(a.out, separator="\t", quote_style="never")
    print(f"written {a.out}", flush=True)


if __name__ == "__main__":
    main()
