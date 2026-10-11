"""Rescale the test name-frequency features to the training scale.

features.py stores name frequency per 100K entities of the split. The test split is smaller (663K US entities against
1.32M in train), so a unique test name got the value of a name shared by two entities in training. Multiplying by
N_test / N_train keeps the meaning "how many entities share this name" (France uses the mean of US and India).
Rewrites work/features_test in place, once (work/freq_fix.json marks it as done).
"""
import glob
import json
import os
import sys

import polars as pl

W = "work"
S1_COLS, R_COLS = ["freq_s1_name_1", "freq_s1_name_of_r_2"], ["freq_r_name_2"]


def counts(sp):
    s = pl.read_parquet(f"{W}/norm/{sp}_s1.parquet", columns=["country"]).group_by("country").len("n1")
    r = pl.concat([pl.read_parquet(f"{W}/norm/{sp}_s{i}.parquet", columns=["country"]) for i in (2, 3)]).group_by("country").len("nr")
    return s.join(r, on="country")


def main():
    if os.path.exists(f"{W}/freq_fix.json"):
        sys.exit("work/freq_fix.json exists: features_test is already rescaled")
    tr, te = counts("train"), counts("test")
    ref = tr.select(pl.col("n1").mean(), pl.col("nr").mean()).row(0)
    tr = pl.concat([tr, pl.DataFrame({"country": ["France"], "n1": [int(ref[0])], "nr": [int(ref[1])]}, schema=tr.schema)])
    fac = (te.join(tr, on="country", suffix="_tr")
           .select("country", (pl.col("n1") / pl.col("n1_tr")).alias("f1"), (pl.col("nr") / pl.col("nr_tr")).alias("fr")))
    print(fac)
    s1c = pl.read_parquet(f"{W}/norm/test_s1.parquet", columns=["uid", "country"]).rename({"uid": "s1_uid"})
    for f in sorted(glob.glob(f"{W}/features_test/part_*.parquet")):
        d = pl.read_parquet(f)
        cols = d.columns
        d = (d.join(s1c, on="s1_uid", how="left").join(fac, on="country", how="left")
             .with_columns([(pl.col(c) * pl.col("f1")).cast(pl.Float32).alias(c) for c in S1_COLS]
                           + [(pl.col(c) * pl.col("fr")).cast(pl.Float32).alias(c) for c in R_COLS]).select(cols))
        d.write_parquet(f + ".tmp")
        os.replace(f + ".tmp", f)
    json.dump(fac.to_dict(as_series=False), open(f"{W}/freq_fix.json", "w"))   # marker: the rescale is done once
    print("features_test rescaled", flush=True)


if __name__ == "__main__":
    main()
