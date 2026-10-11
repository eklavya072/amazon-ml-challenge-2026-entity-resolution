"""Pair features for every candidate pair.

Name: string scores on the cleaned name, full name, space-free name and dba alias, IDF-weighted token overlap and the
rarest token found on one side only; legal-form agreement. Address: the same scores plus house numbers and shared
numbers. Also name frequency, blocking scores and the rank of the pair among the S1's and the record's candidates.
Token IDF is computed once on train + test text, so both splits use the same scale.
"""

import argparse
import os
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

NORM_COLS = ["uid", "country", "n_full", "n_clean", "n_legal", "n_alt", "n_web", "n_script",
             "a_clean", "a_nums", "a_null"]


def idf_table(paths, col, name):
    """Token IDF ln(N / df) per country over the given files."""
    lf = pl.concat([pl.scan_parquet(p).select("country", col) for p in paths])
    n = lf.group_by("country").len("N")
    df = (lf.select("country", pl.col(col).str.split(" ").list.unique().alias("t")).explode("t")
          .filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
          .group_by("country", "t").len("df"))
    return (df.join(n, on="country")
            .select("country", "t", (pl.col("N") / pl.col("df")).log().cast(pl.Float32).alias(name))
            .collect(engine="streaming"))


def cp(a, b, scorer, empty_nan=True):
    """Element-wise rapidfuzz score of two aligned string lists (multithreaded C++)."""
    out = process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
    if empty_nan:
        bad = np.fromiter(((not x) or (not y) for x, y in zip(a, b)), bool, len(a))
        out[bad] = np.nan
    return out


def token_features(p, col1, col2, idf, prefix):
    """IDF-weighted overlap and 'rarest unmatched token' features for one text field."""
    q = p.select(
        "pid", "country",
        pl.col(col1).str.split(" ").list.unique().list.eval(pl.element().filter(pl.element() != "")).alias("A"),
        pl.col(col2).str.split(" ").list.unique().list.eval(pl.element().filter(pl.element() != "")).alias("B"),
    ).with_columns(
        pl.col("A").list.set_intersection("B").alias("I"),
        pl.col("A").list.set_difference("B").alias("OA"),
        pl.col("B").list.set_difference("A").alias("OB"),
    )
    res = q.select("pid", pl.col("A").list.len().alias(f"{prefix}_nA"),
                   pl.col("B").list.len().alias(f"{prefix}_nB"),
                   pl.col("I").list.len().alias(f"{prefix}_nI"))
    for lst, agg in (("A", "sum"), ("B", "sum"), ("I", "sum"), ("OA", "max"), ("OB", "max")):
        e = (q.select("pid", "country", pl.col(lst).alias("t")).explode("t")
             .join(idf, on=["country", "t"], how="left")
             .group_by("pid").agg(getattr(pl.col("idf"), agg)().alias(f"{prefix}_{lst}_{agg}")))
        res = res.join(e, on="pid", how="left")
    return res.select(
        "pid", f"{prefix}_nA", f"{prefix}_nB", f"{prefix}_nI",
        (pl.col(f"{prefix}_I_sum") / pl.col(f"{prefix}_A_sum")).alias(f"{prefix}_idfcov1"),
        (pl.col(f"{prefix}_I_sum") / pl.col(f"{prefix}_B_sum")).alias(f"{prefix}_idfcov2"),
        (pl.col(f"{prefix}_I_sum") / (pl.col(f"{prefix}_A_sum") + pl.col(f"{prefix}_B_sum")
                                       - pl.col(f"{prefix}_I_sum"))).alias(f"{prefix}_idfjac"),
        pl.col(f"{prefix}_I_sum").alias(f"{prefix}_idf_shared"),
        pl.col(f"{prefix}_OA_max").fill_null(0).alias(f"{prefix}_maxidf_only1"),
        pl.col(f"{prefix}_OB_max").fill_null(0).alias(f"{prefix}_maxidf_only2"),
    )


def pair_features(p, name_idf, addr_idf):
    """Compute all per-pair features for a joined chunk ``p`` (suffix _1 = S1, _2 = S2/S3)."""
    n1, n2 = p["n_clean_1"].to_list(), p["n_clean_2"].to_list()
    f1, f2 = p["n_full_1"].to_list(), p["n_full_2"].to_list()
    a1, a2 = p["a_clean_1"].to_list(), p["a_clean_2"].to_list()
    ns1 = [x.replace(" ", "") for x in n1]
    ns2 = [x.replace(" ", "") for x in n2]
    alt2 = p["n_alt_2"].to_list()
    feats = {
        "n_ratio": cp(n1, n2, fuzz.ratio), "n_tset": cp(n1, n2, fuzz.token_set_ratio),
        "n_tsort": cp(n1, n2, fuzz.token_sort_ratio), "n_partial": cp(n1, n2, fuzz.partial_ratio),
        "n_jw": cp(n1, n2, JaroWinkler.normalized_similarity),
        "nf_ratio": cp(f1, f2, fuzz.ratio), "nf_tset": cp(f1, f2, fuzz.token_set_ratio),
        "ns_ratio": cp(ns1, ns2, fuzz.ratio), "ns_partial": cp(ns1, ns2, fuzz.partial_ratio),
        "nalt_tset": cp(n1, alt2, fuzz.token_set_ratio),
        "a_ratio": cp(a1, a2, fuzz.ratio), "a_tset": cp(a1, a2, fuzz.token_set_ratio),
        "a_tsort": cp(a1, a2, fuzz.token_sort_ratio), "a_partial": cp(a1, a2, fuzz.partial_ratio),
    }
    feats["n_jw"] = feats["n_jw"] * 100
    # 0-100 similarity scores stored as 1-byte integers (null when a side is missing)
    name_cols = [c for c in ("nscore", "nrank") if c in p.columns]  # blocking b4 name-match score
    out = p.select(["pid", "s1_uid", "r_uid", "bscore", "nkeys", "brank", "rrank"] + name_cols).with_columns(
        [pl.Series(k, v).fill_nan(None).round(0).cast(pl.UInt8) for k, v in feats.items()])
    first = lambda c: pl.col(c).str.split(" ").list.first()
    last = lambda c: pl.col(c).str.split(" ").list.last()
    nums1 = pl.col("a_nums_1").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    nums2 = pl.col("a_nums_2").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    leg1 = pl.col("n_legal_1").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    leg2 = pl.col("n_legal_2").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    extra = p.select(
        "pid",
        (first("n_clean_1") == first("n_clean_2")).cast(pl.Int8).alias("n_first_eq"),
        (last("n_clean_1") == last("n_clean_2")).cast(pl.Int8).alias("n_last_eq"),
        (pl.col("n_clean_1") == pl.col("n_clean_2")).cast(pl.Int8).alias("n_exact"),
        (pl.col("n_clean_1").str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("")
         == pl.col("n_clean_2").str.replace_all(" ", "")).cast(pl.Int8).alias("n_acronym"),
        pl.col("n_clean_1").str.len_chars().alias("n_len1"),
        pl.col("n_clean_2").str.len_chars().alias("n_len2"),
        pl.col("n_web_2").alias("n_web2"), pl.col("n_script_2").alias("n_script2"),
        pl.col("n_script_1").alias("n_script1"),
        (pl.col("n_alt_2") != "").cast(pl.Int8).alias("n_has_alt2"),
        # legal form: 1 equal, 0 one side missing, -1 both present but disjoint
        pl.when((leg1.list.len() == 0) | (leg2.list.len() == 0)).then(0)
        .when(leg1.list.set_intersection(leg2).list.len() > 0).then(1).otherwise(-1)
        .cast(pl.Int8).alias("legal_agree"),
        pl.col("a_null_1").alias("a_null1"), pl.col("a_null_2").alias("a_null2"),
        pl.col("a_clean_1").str.len_chars().alias("a_len1"),
        pl.col("a_clean_2").str.len_chars().alias("a_len2"),
        nums1.list.len().alias("num_n1"), nums2.list.len().alias("num_n2"),
        nums1.list.set_intersection(nums2).list.len().alias("num_shared"),
        (nums1.list.first() == nums2.list.first()).cast(pl.Int8).alias("hn_eq"),
        (nums1.list.first().cast(pl.Int64, strict=False)
         - nums2.list.first().cast(pl.Int64, strict=False)).abs().alias("hn_absdiff"),
        nums2.list.set_difference(nums1).list.len().alias("num_only2"),
        nums1.list.set_difference(nums2).list.len().alias("num_only1"),
        (pl.col("r_uid") // 10**10).cast(pl.Int8).alias("src2"),
    )
    out = out.join(extra, on="pid")
    # smallest gap between numbers found on one side only (302 vs 306 -> 4), S1's first number found in R
    nq = p.select("pid", nums1.alias("A"), nums2.alias("B")).with_columns(
        pl.col("A").list.set_difference("B").alias("OA"), pl.col("B").list.set_difference("A").alias("OB"),
        pl.col("B").list.contains(pl.col("A").list.first()).cast(pl.Int8).alias("hn_in_r"))
    oa = nq.select("pid", pl.col("OA").alias("x")).explode("x").drop_nulls("x").with_columns(
        pl.col("x").cast(pl.Int64, strict=False))
    ob = nq.select("pid", pl.col("OB").alias("y")).explode("y").drop_nulls("y").with_columns(
        pl.col("y").cast(pl.Int64, strict=False))
    md = (oa.join(ob, on="pid").group_by("pid")
          .agg((pl.col("x") - pl.col("y")).abs().min().alias("num_mindiff")))
    out = out.join(nq.select("pid", "hn_in_r"), on="pid").join(md, on="pid", how="left")
    out = out.join(token_features(p, "n_clean_1", "n_clean_2", name_idf, "nt"), on="pid", how="left")
    out = out.join(token_features(p, "a_clean_1", "a_clean_2", addr_idf, "at"), on="pid", how="left")
    return out


def context_features(f):
    """Rank and gap of a pair among its S1's candidates."""
    q = f.with_columns(
        (0.5 * pl.col("n_tset").fill_null(0) + 0.5 * pl.col("a_tset").fill_null(pl.col("n_tset"))
         ).alias("simple"),
        # relative blocking scores only (the raw scale depends on the dataset size)
        (pl.col("bscore") / pl.col("bscore").max().over("s1_uid")).alias("bscore_rel"),
        (pl.col("bscore") / (pl.col("bscore") + pl.col("r_bgap"))).alias("r_brel"),
    ).drop("bscore", "r_bgap")
    if "nscore" in q.columns:
        q = q.with_columns((pl.col("nscore") / pl.col("nscore").max().over("s1_uid")).fill_nan(0).alias("nscore_rel")
                           ).drop("nscore")
    return q.with_columns(
        pl.len().over("s1_uid").alias("c_ncand_s1"),
        pl.col("simple").rank("ordinal", descending=True).over("s1_uid").alias("c_rank_s1"),
        (pl.col("simple").max().over("s1_uid") - pl.col("simple")).alias("c_gap_s1"),
        (pl.col("simple") >= 90).sum().over("s1_uid").alias("c_n90_s1"),
        pl.col("n_tset").rank("ordinal", descending=True).over("s1_uid").alias("c_nrank_s1"),
        pl.col("a_tset").rank("ordinal", descending=True).over("s1_uid").alias("c_arank_s1"),
        (pl.col("n_tset").max().over("s1_uid") - pl.col("n_tset")).alias("c_ngap_s1"),
        (pl.col("a_tset").max().over("s1_uid") - pl.col("a_tset")).alias("c_agap_s1"),
    )


def build(split, cand_path, norm_dir, out_dir, n_chunks=12, top_k=50, sub=8):
    """Compute features for all candidate pairs of a split; one Parquet part per S1 bucket."""
    t = time.time()
    os.makedirs(out_dir, exist_ok=True)
    s1_path = f"{norm_dir}/{split}_s1.parquet"
    r_paths = [f"{norm_dir}/{split}_s{s}.parquet" for s in (2, 3)]
    # IDF on train + test text together, so both splits share one scale
    all_paths = [f"{norm_dir}/{sp}_s{s}.parquet" for sp in ("train", "test") for s in (1, 2, 3)]
    name_idf = idf_table(all_paths, "n_clean", "idf")
    addr_idf = idf_table(all_paths, "a_clean", "idf")
    # name frequency per 100K records of the country (the test is rescaled by freq_fix.py)
    n_s1 = pl.scan_parquet(s1_path).group_by("country").len("N").collect()
    n_r = pl.concat([pl.scan_parquet(p).select("country") for p in r_paths]).group_by("country").len("N").collect()
    s1_freq = (pl.scan_parquet(s1_path).group_by("country", "n_clean").len("c")
               .join(n_s1.lazy(), on="country")
               .select("country", "n_clean", (pl.col("c") * 1e5 / pl.col("N")).cast(pl.Float32).alias("freq_s1_name"))
               .collect(engine="streaming"))
    r_freq = (pl.concat([pl.scan_parquet(p).select("country", "n_clean") for p in r_paths])
              .group_by("country", "n_clean").len("c").join(n_r.lazy(), on="country")
              .select("country", "n_clean", (pl.col("c") * 1e5 / pl.col("N")).cast(pl.Float32).alias("freq_r_name"))
              .collect(engine="streaming"))
    print(f"[{split}] idf/frequency tables ready ({time.time() - t:.0f}s)", flush=True)
    # candidates are read chunk by chunk
    keep = pl.col("brank") <= top_k
    if "nrank" in pl.read_parquet_schema(cand_path):  # b4: reserved slots for the best name matches
        from blocking import NAME_QUOTA
        keep = keep | ((pl.col("nrank") <= NAME_QUOTA) & (pl.col("nscore") > 0))
    cand = pl.scan_parquet(cand_path).filter(keep)
    r_freq = r_freq.join(s1_freq.rename({"freq_s1_name": "freq_s1_name_of_r"}),
                         on=["country", "n_clean"], how="left")
    bucket = (pl.col("s1_uid") * 7919) % n_chunks
    sub_bucket = ((pl.col("s1_uid") * 7919) // n_chunks) % sub  # independent of ``bucket``
    for b in range(n_chunks):
        if os.path.exists(f"{out_dir}/part_{b:02d}.parquet"):  # resume: bucket already done
            continue
        for j in range(sub):  # optional sub-split of a bucket (all pairs of an S1 stay together)
            out = f"{out_dir}/part_{b:02d}.parquet" if sub == 1 else f"{out_dir}/part_{b:02d}_{j}.parquet"
            if os.path.exists(out):
                continue
            c = (cand.filter((bucket == b) & (sub_bucket == j)).collect(engine="streaming")
                 .with_row_index("pid"))
            # load only the records this chunk needs (keeps memory low)
            s1j = (pl.scan_parquet(s1_path).select(NORM_COLS)
                   .join(c.lazy().select(pl.col("s1_uid").alias("uid")).unique(), on="uid", how="semi")
                   .collect(engine="streaming").join(s1_freq, on=["country", "n_clean"], how="left"))
            rj = (pl.concat([pl.scan_parquet(q).select(NORM_COLS) for q in r_paths])
                  .join(c.lazy().select(pl.col("r_uid").alias("uid")).unique(), on="uid", how="semi")
                  .collect(engine="streaming").join(r_freq, on=["country", "n_clean"], how="left"))
            s1j = s1j.rename({x: f"{x}_1" for x in s1j.columns if x not in ("uid", "country")}).rename(
                {"uid": "s1_uid"})
            rj = rj.rename({x: f"{x}_2" for x in rj.columns if x not in ("uid", "country")}).rename(
                {"uid": "r_uid", "country": "country_r"})
            p = c.join(s1j, on="s1_uid").join(rj, on="r_uid")
            del s1j, rj
            f = pair_features(p, name_idf, addr_idf)
            f = f.join(p.select("pid", "r_ncand", "r_bgap", "freq_s1_name_1", "freq_r_name_2",
                                "freq_s1_name_of_r_2"), on="pid")
            f = context_features(f).drop("pid", "simple")
            f.write_parquet(out, compression="zstd")
            print(f"  part {b}.{j}: {f.height:,} rows, {f.width} cols ({time.time() - t:.0f}s)", flush=True)
            del c, p, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--norm-dir", default="work/norm")
    ap.add_argument("--n-chunks", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=50, help="candidates per S1 fed to the model")
    ap.add_argument("--sub", type=int, default=8, help="split each bucket into this many parts")
    a = ap.parse_args()
    cand = f"{a.work_dir}/blocking/cand_{a.split}.parquet"
    build(a.split, cand, a.norm_dir, f"{a.work_dir}/features_{a.split}", a.n_chunks, a.top_k, a.sub)


if __name__ == "__main__":
    main()
