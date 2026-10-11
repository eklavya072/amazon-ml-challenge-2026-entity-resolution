"""Check candidate_pairs.tsv with the validator's rules, in slices so it fits in memory.

Header, one row per test S1, no duplicate or unknown ids, every matched id is a candidate.

    python src/check_candidates.py <test_dir> <candidate_pairs.tsv> <matching_results.tsv>
"""
import sys

import polars as pl

K = 8
test_dir, cand_path, match_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(cand_path, encoding="utf-8") as f:
    header = f.readline().rstrip("\n").split("\t")
assert header == ["source1_entity_id", "candidate_entity_ids"], header
rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)
scan = lambda p: pl.scan_csv(p, separator="\t", quote_char=None, infer_schema=False)
s1 = rd(f"{test_dir}/test_source1.tsv").select("entity_id")
ids = scan(cand_path).select("source1_entity_id").collect()
assert ids.height == s1.height and ids["source1_entity_id"].n_unique() == ids.height, "row count / duplicate rows"
assert ids.join(s1, left_on="source1_entity_id", right_on="entity_id", how="anti").height == 0, "unknown S1 id"
valid = pl.concat([rd(f"{test_dir}/test_source{s}.tsv").select("entity_id") for s in (2, 3)])
part = lambda k: (pl.col("source1_entity_id").hash(seed=3) % K) == k
n_ids = n_match = n_empty = 0
for k in range(K):
    c = scan(cand_path).filter(part(k)).collect()
    n_empty += (c["candidate_entity_ids"].fill_null("") == "").sum()
    ex = (c.with_columns(pl.col("candidate_entity_ids").fill_null("").str.split(","))
          .explode("candidate_entity_ids").filter(pl.col("candidate_entity_ids") != ""))
    del c
    assert ex.select("source1_entity_id", "candidate_entity_ids").is_duplicated().sum() == 0, "duplicate id in a list"
    assert ex["candidate_entity_ids"].str.contains(r"^S[23]-\d+$").all(), "bad prefix"
    assert ex.join(valid, left_on="candidate_entity_ids", right_on="entity_id", how="anti").height == 0, "unknown id"
    m = (scan(match_path).filter(part(k)).collect().with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
         .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != ""))
    missing = m.join(ex, left_on=["source1_entity_id", "matched_entity_ids"],
                     right_on=["source1_entity_id", "candidate_entity_ids"], how="anti").height
    assert missing == 0, f"{missing} matched ids not in candidates"
    n_ids += ex.height
    n_match += m.height
    del ex, m
print(f"candidate_pairs.tsv OK: {ids.height:,} rows, {n_ids:,} candidate ids ({n_ids / ids.height:.1f} per S1), "
      f"{n_empty:,} empty rows; all {n_match:,} matches are candidates")
