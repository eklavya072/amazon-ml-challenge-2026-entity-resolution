"""Add the record-side pairs to the candidate file, so every match is also a candidate.

    python src/extend_candidates.py --cand work/candidates/candidate_pairs.tsv --added work/final/added_pairs.parquet \
        --out output/candidate_pairs.tsv
"""
import argparse

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--added", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    addl = pl.read_parquet(a.added).select("s1e", "re").group_by("s1e").agg(pl.col("re").alias("new"))
    c = pl.scan_csv(a.cand, separator="\t", quote_char=None, infer_schema=False)
    merged = (pl.concat_list(pl.col("candidate_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")),
                             pl.col("new")).list.unique(maintain_order=True).list.join(","))
    c = (c.join(addl.lazy(), left_on="source1_entity_id", right_on="s1e", how="left")
         .with_columns(pl.when(pl.col("new").is_null()).then(pl.col("candidate_entity_ids").fill_null("")).otherwise(merged)
                       .alias("candidate_entity_ids"))
         .select("source1_entity_id", "candidate_entity_ids"))
    c.sink_csv(a.out, separator="\t", quote_style="never")
    print(f"written {a.out}; entities with added candidates: {addl.height:,}", flush=True)


if __name__ == "__main__":
    main()
