"""Normalise names and addresses to lower-case ASCII tokens.

Columns added per record:
    n_full    name tokens including the legal form
    n_clean   name tokens without legal form, titles and stop words
    n_legal   legal-form tokens ("llc", "pvt ltd", ...)
    n_alt     second name of "X dba Y" / "X a/k/a Y" ('' if none)
    n_web     1 if the name is a website
    n_script  0 ascii, 1 accented latin, 2 indic, 3 other
    a_clean   address tokens with standard street-type abbreviations
    a_nums    numbers in the address, in order
    a_null    1 if the address is missing
Indic words are mapped to English with a dictionary learned from the training pairs (learn_translit_dict).
"""

import json
import os
import re
import unicodedata
from multiprocessing import Pool

import polars as pl
from anyascii import anyascii

LEGAL = {
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp",
    "corporation": "corp", "co": "co", "company": "co", "ltd": "ltd",
    "limited": "ltd", "pvt": "pvt", "private": "pvt", "llp": "llp", "lp": "lp",
    "pc": "pc", "pllc": "pllc", "plc": "plc", "pa": "pa", "sa": "sa", "sas": "sas",
    "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "sci": "sci", "snc": "snc",
    "ei": "ei", "selarl": "selarl", "gmbh": "gmbh", "ltda": "ltda",
}
TITLES = {"dr", "smt", "shri", "sri", "shree", "ms", "mr", "mrs", "prof", "messrs"}
STOP = {"the", "of", "and", "de", "du", "la", "le", "les", "des", "et", "a", "an", "d", "l"}
# long/variant form -> one short canonical form (applied to address tokens only)
ADDR_ABBR = {
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "av", "ave": "av",
    "drive": "dr", "drv": "dr", "lane": "ln", "court": "ct", "boulevard": "bd",
    "blvd": "bd", "blv": "bd", "place": "pl", "circle": "cir", "trail": "trl",
    "highway": "hwy", "parkway": "pkwy", "pky": "pkwy", "terrace": "ter", "square": "sq",
    "rue": "r", "allee": "all", "impasse": "imp", "chemin": "ch", "chem": "ch",
    "route": "rte", "quai": "q", "esplanade": "esp", "faubourg": "fbg", "residence": "res",
    "batiment": "bat", "appartement": "apt", "apartment": "apt", "appt": "apt",
    "building": "bldg", "floor": "fl", "flr": "fl", "north": "n", "south": "s",
    "east": "e", "west": "w", "nr": "near", "opposite": "opp", "sector": "sec",
    "phase": "ph", "extension": "ext", "extn": "ext", "colony": "col", "mount": "mt",
    "point": "pt", "heights": "hts", "junction": "jct", "jn": "jct", "center": "ctr",
    "centre": "ctr", "suite": "ste", "sainte": "ste", "township": "twp", "unit": "unit",
    "no": "", "number": "", "nos": "", "null": "", "none": "", "nan": "",
}

_INDIC = re.compile(r"[ऀ-෿]")
_NUM_SIGN = re.compile(r"\b[nN]\s?[°º]\s?")  # "N°", "Nº" (French "numéro")
_WEB = re.compile(
    r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|in|net|org|co|fr|biz|info|us|io)\b")
_AKA = re.compile(r"\b(?:a\s*/\s*k\s*/\s*a|aka|d\s*/\s*b\s*/\s*a|dba|t\s*/\s*a|"
                  r"doing business as|also known as)\b\s*:?")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")
_NULLISH = re.compile(r"<\s*null\s*>|\bnull\b|\bnone\b|\bnan\b")
_PUNCT_PAD = re.compile(r"[,()\[\]|.:;]")
_POBOX = re.compile(r"\b(?:p\s*\.?\s*o\s*\.?\s*box|post\s*box)\b")

TRANSLIT = {}  # native token -> english token, filled by load_translit_dict()


def load_translit_dict(path):
    """Load the learned native->English token dictionary into this module."""
    global TRANSLIT
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            TRANSLIT = json.load(f)


def script_class(s):
    """0 = pure ASCII, 1 = accented Latin, 2 = Indic script, 3 = other non-ASCII."""
    if s.isascii():
        return 0
    if _INDIC.search(s):
        return 2
    for ch in s:
        if ord(ch) > 127 and unicodedata.name(ch, "").startswith("LATIN"):
            return 1
    return 3


def to_ascii(s):
    """NFKC-normalise, map known native tokens via TRANSLIT, transliterate the rest."""
    s = unicodedata.normalize("NFKC", s)
    if s.isascii():
        return s.lower()
    s = _NUM_SIGN.sub(" ", s)
    if TRANSLIT and _INDIC.search(s):
        s = _PUNCT_PAD.sub(r" \g<0> ", s)  # detach punctuation so tokens hit the dictionary
        s = " ".join(TRANSLIT.get(t, t) for t in s.split())
    return anyascii(s).lower()


_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})


def _deleet(tok):
    """Undo digit-for-letter typos inside words ('fu1ler' -> 'fuller'); pure numbers are kept."""
    if tok.isalpha() or tok.isdigit():
        return tok
    return tok.translate(_LEET)


def _tokens(text, deleet=False):
    """Split on non-alphanumerics, then glue runs of single letters (p c -> pc)."""
    toks = [t for t in _NON_ALNUM.split(text) if t]
    if deleet:
        toks = [_deleet(t) for t in toks]
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    dedup = [t for i, t in enumerate(out) if i == 0 or t != out[i - 1]]  # "x x" -> "x"
    return dedup


def norm_name(raw):
    """Normalise one business name. Returns (n_full, n_clean, n_legal, n_alt, n_web, n_script)."""
    if raw is None:
        return "", "", "", "", 0, 0
    script = script_class(raw)
    s = to_ascii(raw)
    parts = [p for p in s.split("|") if p.strip()]
    s = parts[0] if parts else ""
    web = 0
    m = _WEB.search(s)
    if m and len(m.group(0)) >= 0.6 * len(s.strip()):  # the name IS a website
        s, web = m.group(1), 1
    alt = ""
    pieces = [p for p in _AKA.split(s) if p.strip()]
    if len(pieces) >= 2:  # "Alias dba Real Name": keep right side as main
        alt, s = pieces[0], pieces[-1]
    s = s.replace("&", " and ").replace("'", "").replace("`", "")
    toks = _tokens(s, deleet=True)
    legal = sorted({LEGAL[t] for t in toks if t in LEGAL})
    clean = [t for t in toks if t not in LEGAL and t not in TITLES and t not in STOP]
    if not clean:  # name made only of legal/stop words: keep what there is
        clean = [t for t in toks if t not in TITLES] or toks
    full = [LEGAL.get(t, t) for t in toks if t not in TITLES]
    alt_clean = " ".join(t for t in _tokens(alt.replace("'", ""), deleet=True)
                         if t not in LEGAL and t not in STOP)
    return " ".join(full), " ".join(clean), " ".join(legal), alt_clean, web, script


def norm_addr(raw):
    """Normalise one address. Returns (a_clean, a_nums, a_null)."""
    if raw is None or not raw.strip():
        return "", "", 1
    s = _NULLISH.sub(" ", to_ascii(raw))
    s = _POBOX.sub(" pobox ", s)
    nums = []
    for d in _DIGITS.findall(s):
        d = d.lstrip("0") or "0"
        if d not in nums:
            nums.append(d)
    toks = []
    for t in _NON_ALNUM.split(s):
        if not t:
            continue
        t = ADDR_ABBR.get(t, t)
        if t:
            if t.isdigit():
                t = t.lstrip("0") or "0"
            toks.append(t)
    return " ".join(toks), " ".join(nums), int(not toks)


OUT_COLS = ["n_full", "n_clean", "n_legal", "n_alt", "n_web", "n_script", "a_clean", "a_nums", "a_null"]
INT_COLS = {"n_web", "n_script", "a_null"}


def _norm_chunk(args):
    """Worker: normalise a chunk of (name, address) pairs."""
    names, addrs, translit_path = args
    load_translit_dict(translit_path)
    rows = [norm_name(x) + norm_addr(y) for x, y in zip(names, addrs)]
    cols = list(zip(*rows)) if rows else [[] for _ in OUT_COLS]
    return pl.DataFrame({c: pl.Series(c, v, dtype=pl.Int8 if c in INT_COLS else pl.String)
                         for c, v in zip(OUT_COLS, cols)})


def normalize_frame(df, translit_path=None, workers=8, chunk=200_000):
    """Add the normalised columns (see module docstring) to a source DataFrame."""
    jobs = [(df["business_name"].slice(i, chunk).to_list(), df["business_address"].slice(i, chunk).to_list(),
             translit_path) for i in range(0, df.height, chunk)]
    with Pool(workers) as p:
        parts = p.map(_norm_chunk, jobs, chunksize=1)
    del jobs
    return pl.concat([df, pl.concat(parts)], how="horizontal")


def learn_translit_dict(s1, r, pairs, min_count=3, min_share=0.3):
    """Map Indic-script tokens to English tokens using the training pairs.

    For true pairs whose record has Indic tokens, count co-occurrences with the English tokens of the S1. A native token
    n maps to the English token e with the best P(e | n) * spelling similarity (P(e | n) >= min_share, seen at least
    min_count times).
    """
    from collections import Counter, defaultdict
    from rapidfuzz import fuzz

    ri = r.filter(pl.col("business_name").str.contains(r"[\u0900-\u0DFF]")
                  | pl.col("business_address").fill_null("").str.contains(r"[\u0900-\u0DFF]"))
    j = (pairs.join(ri.select(pl.col("uid").alias("r_uid"), pl.col("business_name").alias("rn"),
                              pl.col("business_address").alias("ra")), on="r_uid")
         .join(s1.select(pl.col("uid").alias("s1_uid"), pl.col("business_name").alias("sn"),
                         pl.col("business_address").alias("sa")), on="s1_uid"))
    co = defaultdict(Counter)
    n_occ = Counter()
    for rn, ra, sn, sa in j.select("rn", "ra", "sn", "sa").iter_rows():
        for native, eng in ((rn, sn), (ra or "", sa or "")):
            if not _INDIC.search(native):
                continue
            etoks = {t for t in _NON_ALNUM.split(eng.lower()) if t and not t.isdigit()}
            ntoks = {t for t in _PUNCT_PAD.sub(" ", unicodedata.normalize("NFKC", native)).split()
                     if _INDIC.search(t)}
            for n in ntoks:
                n_occ[n] += 1
                co[n].update(etoks)
    out = {}
    for n, c in co.items():
        if n_occ[n] < min_count:
            continue
        guess = anyascii(n).lower()
        cands = [(k / n_occ[n] * fuzz.ratio(guess, e) / 100, k, e) for e, k in c.items()
                 if k / n_occ[n] >= min_share and k >= min_count]
        if cands:
            out[n] = max(cands)[2]
    return out


def fold_of(uid_col="uid", k=5):
    """Deterministic fold id (0..k-1) of a Source-1 entity from its integer uid."""
    return (((pl.col(uid_col) % 10**10) * 2654435761) % 2**32 % k).cast(pl.Int8)


def main():
    """Learn the transliteration dictionary (train folds 1-4 only) and normalise all sources."""
    import argparse
    import time
    ap = argparse.ArgumentParser(description="normalise all source files")
    ap.add_argument("--parquet-dir", default="work/parquet")
    ap.add_argument("--out-dir", default="work/norm")
    ap.add_argument("--val-fold", type=int, default=0,
                    help="fold whose labels are hidden when learning the dictionary")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t = time.time()
    s1 = pl.read_parquet(f"{a.parquet_dir}/train_s1.parquet")
    folds = s1.select(pl.col("uid").alias("s1_uid"), fold_of().alias("fold"))
    folds.write_parquet(f"{a.out_dir}/folds.parquet")
    pairs = pl.read_parquet(f"{a.parquet_dir}/gt_pairs.parquet").join(folds, on="s1_uid")
    pairs = pairs.filter(pl.col("fold") != a.val_fold)
    r = pl.concat([pl.read_parquet(f"{a.parquet_dir}/train_s{s}.parquet") for s in (2, 3)])
    d = learn_translit_dict(s1, r, pairs)
    dict_path = f"{a.out_dir}/translit_dict.json"
    with open(dict_path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    print(f"translit dict: {len(d):,} tokens ({time.time() - t:.0f}s)")
    del r
    for split in ("train", "test"):
        for s in (1, 2, 3):
            df = pl.read_parquet(f"{a.parquet_dir}/{split}_s{s}.parquet")
            df = normalize_frame(df, dict_path)
            df.write_parquet(f"{a.out_dir}/{split}_s{s}.parquet", compression="zstd")
            print(f"{split} S{s} normalised ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
