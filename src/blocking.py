"""Candidate generation (blocking).

Each record gets short keys from its name and address. Two records become a candidate pair when they share a rare key;
every shared key adds an IDF-like weight to the pair score and each Source-1 entity keeps its top pairs. Keys include
the country, so records are only compared within a country. The work is split by key hash and S1 bucket to fit in
8 GB of RAM.
"""

import argparse
import os
import shutil
import time
from multiprocessing import Pool

import polars as pl

from normalize import ADDR_ABBR

STREET_TYPES = {v for v in ADDR_ABBR.values() if v} | {"unit", "pobox", "st", "near", "opp"}
VOWELS = set("aeiou")
MAX_NAME_TOK = 6
MAX_ADDR_TOK = 8


def skeleton(tok):
    """Consonant skeleton with repeats collapsed: 'maassge' -> 'msg' (typo-tolerant key)."""
    out = []
    for ch in tok:
        if ch in VOWELS:
            continue
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out) or tok


# selectivity caps: max S1 records with the key, max S1 x S2/S3 pairs it creates
STRICT_CAPS = (15, 500)
LOOSE_CAPS = (150, 5000)
LOOSE_FAMILIES = (11, 12, 15)
N_FAMILIES = 15
# name families also give a name score (nscore); the best NAME_QUOTA pairs by nscore are always kept
NAME_FAMILIES = (1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 14, 15)
NAME_QUOTA = 15


def family_cap(f, max_s1_df, max_pair_df):
    return LOOSE_CAPS if f in LOOSE_FAMILIES else (max_s1_df, max_pair_df)


def record_keys(country, n_clean, n_alt, a_clean, a_nums, emit_loose=False):
    """Blocking keys of one record.

    Families (prefix digit):
      1 name-token pair        2 single name token      3 name token + house number
      4 number + address word  5 adjacent address words 6 name without spaces
      7 first 7 chars of name without spaces            8 skeleton name-token pair
      9 name token + address word  10 number + two adjacent address words
     11 name-token pair (loose)  12 single name token (loose)
     13 full name + house number 14 full name + address word   15 full name (loose)
    Loose families are used by S1 records and by S2/S3 records without an address.
    """
    T = list(dict.fromkeys(n_clean.split()))[:MAX_NAME_TOK]
    Talt = list(dict.fromkeys(n_alt.split()))[:2] if n_alt else []
    all_nums = a_nums.split() if a_nums else []
    # first two numbers + up to two further multi-digit numbers (house numbers in long
    # Indian addresses are often the 3rd/4th number)
    nums = all_nums[:2] + [n for n in all_nums[2:] if len(n) >= 2][:2]
    A = [t for t in dict.fromkeys(a_clean.split()) if len(t) >= 2 and not t.isdigit()
         and t not in STREET_TYPES][:MAX_ADDR_TOK]
    c = country
    keys = []
    for i in range(len(T)):
        keys.append(f"2|{c}|{T[i]}")
        for j in range(i + 1, len(T)):
            a, b = sorted((T[i], T[j]))
            keys.append(f"1|{c}|{a}|{b}")
    if nums:
        for t in T[:4] + Talt:
            for n in nums[:3]:
                keys.append(f"3|{c}|{t}|{n}")
        for n in nums:
            for a in A:
                keys.append(f"4|{c}|{n}|{a}")
        for n in nums[:3]:  # 3-part: number + two adjacent address words (very selective)
            for i in range(min(len(A), 6) - 1):
                keys.append(f"10|{c}|{n}|{A[i]}|{A[i + 1]}")
    for i in range(len(A) - 1):
        keys.append(f"5|{c}|{A[i]}|{A[i + 1]}")
    ns = "".join(T)
    if len(ns) >= 4:
        keys.append(f"6|{c}|{ns}")
        for n in nums[:3]:
            keys.append(f"13|{c}|{ns}|{n}")
        for a in A[:6]:
            keys.append(f"14|{c}|{ns}|{a}")
    if len(ns) >= 7:
        keys.append(f"7|{c}|{ns[:7]}")
    sk = [skeleton(t) for t in T]
    for i in range(len(sk) - 1):
        a, b = sorted((sk[i], sk[i + 1]))
        keys.append(f"8|{c}|{a}|{b}")
    for t in T[:3] + Talt:
        for a in A[:6]:
            keys.append(f"9|{c}|{t}|{a}")
    if emit_loose:
        if len(ns) >= 4:
            keys.append(f"15|{c}|{ns}")
        for i in range(len(T)):
            keys.append(f"12|{c}|{T[i]}")
            for j in range(i + 1, len(T)):
                a, b = sorted((T[i], T[j]))
                keys.append(f"11|{c}|{a}|{b}")
    return keys


def _write_keys(args):
    """Worker: build the keys of one slice of a source file and write them to Parquet."""
    path, offset, length, out_path, is_s1 = args
    df = pl.scan_parquet(path).slice(offset, length).select(
        "uid", "country", "n_clean", "n_alt", "a_clean", "a_nums").collect()
    uids, keys = [], []
    for uid, country, n_clean, n_alt, a_clean, a_nums in df.iter_rows():
        ks = record_keys(country or "", n_clean or "", n_alt or "", a_clean or "", a_nums or "",
                         emit_loose=is_s1 or not a_clean)
        uids.extend([uid] * len(ks))
        keys.extend(ks)
    part = pl.DataFrame({"uid": uids, "k": keys}).select(
        pl.col("k").hash(seed=17).alias("key"), "uid",
        pl.col("k").str.split_exact("|", 1).struct.field("field_0").cast(pl.UInt8).alias("fam"))
    part.write_parquet(out_path)
    return part.height


def build_key_files(norm_dir, split, out_dir, workers=8, chunk=60_000):
    """Write key Parquet parts for S1 (keys/s1_*.parquet) and S2+S3 (keys/r_*.parquet)."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    jobs = []
    for s in (1, 2, 3):
        path = f"{norm_dir}/{split}_s{s}.parquet"
        n = pl.scan_parquet(path).select(pl.len()).collect().item()
        side = "s1" if s == 1 else "r"
        jobs += [(path, i, chunk, f"{out_dir}/{side}_{s}_{i // chunk:04d}.parquet", s == 1)
                 for i in range(0, n, chunk)]
    with Pool(workers) as p:
        n = sum(p.imap_unordered(_write_keys, jobs))
    return n


def selective_key_rows(key_dir, tmp_dir, max_s1_df, max_pair_df, n_part=16):
    """Keep the selective keys of one hash slice: <= cap1 S1 records and n1 * n2 <= cap2.

    Key weight w = ln(1 + 1000 / (n1 * n2)); the caps depend on the key family.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    fams = list(range(1, N_FAMILIES + 1))
    caps = pl.DataFrame({"fam": fams,
                         "cap1": [family_cap(f, max_s1_df, max_pair_df)[0] for f in fams],
                         "cap2": [family_cap(f, max_s1_df, max_pair_df)[1] for f in fams]},
                        schema={"fam": pl.UInt8, "cap1": pl.Int64, "cap2": pl.Int64})
    tot_s, tot_r = 0, 0
    for p in range(n_part):
        s = (pl.scan_parquet(f"{key_dir}/s1_*.parquet").filter(pl.col("key") % n_part == p)
             .select("key", "uid", "fam").unique().collect())
        n1 = (s.group_by("key").agg(pl.len().alias("n1"), pl.col("fam").first())
              .join(caps, on="fam").filter(pl.col("n1") <= pl.col("cap1")))
        r = (pl.scan_parquet(f"{key_dir}/r_*.parquet").filter(pl.col("key") % n_part == p)
             .select("key", "uid").unique()
             .join(n1.lazy().select("key"), on="key", how="semi").collect())
        n2 = r.group_by("key").len("n2")
        good = (n1.join(n2, on="key").filter(pl.col("n1") * pl.col("n2") <= pl.col("cap2"))
                .select("key", (1 + 1000.0 / (pl.col("n1") * pl.col("n2"))).log().cast(pl.Float32).alias("w"),
                        pl.col("fam").is_in(list(NAME_FAMILIES)).alias("nm")))
        sg = s.select("key", "uid").join(good, on="key")
        rg = r.join(good.select("key"), on="key", how="semi")
        sg.write_parquet(f"{tmp_dir}/s1_{p:02d}.parquet")
        rg.write_parquet(f"{tmp_dir}/r_{p:02d}.parquet")
        tot_s += sg.height
        tot_r += rg.height
        del s, r, n1, n2, good, sg, rg
    return tot_s, tot_r


def generate_candidates(key_dir, out_path, top_k=50, max_s1_df=STRICT_CAPS[0], max_pair_df=STRICT_CAPS[1],
                        n_part=16, n_buckets=12, log=print, name_quota=NAME_QUOTA):
    """Score pairs that share a selective key (sum of key weights) and keep the top-K per S1.

    Runs per bucket of S1 entities; record-side context (rrank, r_ncand, r_bgap) is added at the end.
    """
    t = time.time()
    tmp = os.path.join(os.path.dirname(out_path), "tmp_" + os.path.basename(out_path).split(".")[0])
    ns, nr = selective_key_rows(key_dir, tmp, max_s1_df, max_pair_df, n_part)
    log(f"  selective key rows: S1 {ns:,}, S2/S3 {nr:,} ({time.time() - t:.0f}s)")
    for b in range(n_buckets):
        partial = []
        for p in range(n_part):
            sb = pl.read_parquet(f"{tmp}/s1_{p:02d}.parquet").filter((pl.col("uid") * 7919) % n_buckets == b)
            rp = pl.read_parquet(f"{tmp}/r_{p:02d}.parquet")
            partial.append(sb.join(rp, on="key", suffix="_r").select("uid", "uid_r", "w", "nm"))
        pairs = (pl.concat(partial).group_by("uid", "uid_r")
                 .agg(pl.col("w").sum().alias("bscore"), pl.len().cast(pl.Int16).alias("nkeys"),
                      pl.col("w").filter(pl.col("nm")).sum().alias("nscore"))
                 .rename({"uid": "s1_uid", "uid_r": "r_uid"})
                 .with_columns(pl.col("bscore").rank("ordinal", descending=True)
                               .over("s1_uid").cast(pl.Int16).alias("brank"),
                               pl.col("nscore").rank("ordinal", descending=True)
                               .over("s1_uid").cast(pl.Int16).alias("nrank"))
                 .filter((pl.col("brank") <= top_k) | ((pl.col("nrank") <= name_quota) & (pl.col("nscore") > 0))))
        pairs.write_parquet(f"{tmp}/pairs_b{b:02d}.parquet")
        log(f"  bucket {b}: {pairs.height:,} pairs kept ({time.time() - t:.0f}s)")
        del partial, pairs
    parts = []
    for q in range(4):  # record-side context, partitioned by r_uid
        c = (pl.scan_parquet(f"{tmp}/pairs_b*.parquet").filter(pl.col("r_uid") % 4 == q).collect()
             .with_columns(
                 pl.col("bscore").rank("ordinal", descending=True).over("r_uid").cast(pl.Int16).alias("rrank"),
                 pl.len().over("r_uid").cast(pl.Int16).alias("r_ncand"),
                 (pl.col("bscore").max().over("r_uid") - pl.col("bscore")).alias("r_bgap")))
        c.write_parquet(f"{tmp}/ctx_{q}.parquet")
        parts.append(c.height)
    pl.scan_parquet(f"{tmp}/ctx_*.parquet").sink_parquet(out_path, compression="zstd")
    shutil.rmtree(tmp, ignore_errors=True)
    return sum(parts)


def main():
    ap = argparse.ArgumentParser(description="blocking / candidate generation")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--norm-dir", default="work/norm")
    ap.add_argument("--work-dir", default="work/blocking")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-s1-df", type=int, default=STRICT_CAPS[0])
    ap.add_argument("--max-pair-df", type=int, default=STRICT_CAPS[1])
    a = ap.parse_args()
    t = time.time()
    key_dir = f"{a.work_dir}/keys_{a.split}"
    n = build_key_files(a.norm_dir, a.split, key_dir)
    print(f"keys written: {n:,} ({time.time() - t:.0f}s)", flush=True)
    out = f"{a.work_dir}/cand_{a.split}.parquet"
    n = generate_candidates(key_dir, out, a.top_k, a.max_s1_df, a.max_pair_df,
                            log=lambda m: print(m, flush=True))
    print(f"candidate pairs: {n:,} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
