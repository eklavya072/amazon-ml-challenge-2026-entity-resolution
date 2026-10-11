# ARCA: Business Entity Resolution at Scale

**Amazon ML Challenge 2026** | Team **ARCA**: Eklavya Singh, Kamran Ansari, Ayush Singh

Matching the same real-world business across three noisy data sources: ~10 million records, three countries and
two languages (one of them with no training labels at all), solved on a single 8 GB laptop.

| | |
|---|---|
| **Public leaderboard** | **Rank 346** among **18,000+ teams** (~90,000 registrations) |
| **Score** | macro F0.5 **0.9869** on the public leaderboard, **0.9891** on our held-out validation |
| **Scale** | 1.7-2.2M reference businesses and ~10M records per split, 84M candidate pairs scored |
| **Hardware** | one 8 GB MacBook + free Kaggle T4 GPUs |
| **Models** | LightGBM + three open cross-encoders (MIT / Apache-2.0), all trained only on the challenge data |

---

## Contents

1. [The problem: entity resolution](#1-the-problem-entity-resolution)
2. [Why this dataset is hard](#2-why-this-dataset-is-hard)
3. [How we planned it](#3-how-we-planned-it)
4. [Architecture](#4-architecture)
5. [Candidate generation (blocking)](#5-candidate-generation-blocking)
6. [Matching models and features](#6-matching-models-and-features)
7. [Decision layer](#7-decision-layer)
8. [Key insights](#8-key-insights)
9. [Engineering under constraints](#9-engineering-under-constraints)
10. [Results](#10-results)
11. [Output](#11-output)
12. [Pipeline, step by step](#12-pipeline-step-by-step)
13. [Reproducing the results](#13-reproducing-the-results)
14. [Team](#14-team)

---

## 1. The problem: entity resolution

Large marketplaces receive information about the same business from many independent sources: registration forms,
partner feeds, public listings. None of them share an identifier. The same shop can appear as
"Sunrise Foods Pvt Ltd, 12 MG Road, Pune", as "SUNRISE FOODS PRIVATE LIMITED, MG ROAD, PUNE" and as a Hindi-script
listing with a cut-down address. Deciding which records describe the same real-world business is **entity
resolution**. It underpins catalogue quality, seller verification, fraud detection and every analysis that counts
businesses rather than records.

In this challenge:

- **Source 1** is a clean reference list of businesses (name, address, country).
- **Sources 2 and 3** are noisy copies of them, plus look-alikes that belong to no one.
- For every Source-1 business we must return all Source-2/3 records of that business, or nothing if it has none.

The score is **F0.5 computed per business and averaged over all businesses**. Precision weighs twice as much as
recall, every business counts equally, and a business with no copies (5.6% of them) only scores if we predict
nothing for it.

## 2. Why this dataset is hard

| challenge | what it looks like |
|---|---|
| **Heavy noise** | typos, digit-for-letter swaps ("fu1ler"), abbreviations, reordered words, legal forms written ten ways (LLC, L.L.C., Pvt Ltd, Private Limited, SARL...), websites used as names, missing addresses |
| **Many scripts** | names and addresses in ten Indic scripts next to English |
| **Planted look-alikes** | records that copy a business's building with one name word changed ("Group", "Holdings"), or its name with a neighbouring house number |
| **Shared names** | 39% of the reference businesses share their cleaned name with another business |
| **A country with no labels** | France (15% of the test, French text) never appears in the training data |
| **A shifted test set** | more look-alikes per business than training (5.8 records per business against 4.7) and smaller splits |
| **Scale** | ~2 x 10^13 possible pairs per split; comparing everything is impossible |

## 3. How we planned it

We treated the challenge as an engineering project with a clear order of priorities, and measured every decision on a
validation set built to behave like the test.

| phase | goal | what we built |
|---|---|---|
| **1. Understand** | know the data and the metric exactly | full data audit (scripts, noise types, one-owner rule, singletons), an exact re-implementation of the metric, checked against the organisers' worked example and an all-empty submission ([data analysis](docs/EDA.md)) |
| **2. Validate honestly** | a validation score we can trust | held-out *businesses* (20%), candidates generated against the full record universe, an out-of-fold chain so no model ever scores data it trained on, test-weighted validation |
| **3. Recall first** | never lose a true pair early | normalisation and a 15-family blocking system: 98.3% recall with 48 candidates per business |
| **4. Precision** | separate copies from look-alikes | two-stage LightGBM, three fine-tuned cross-encoders, a stacker and an F0.5-optimal decision rule |
| **5. Close the test gap** | make the leaderboard follow validation | label-free test forensics: decoy-density correction, a split-size bug fix, rules for the unlabelled language |
| **6. Ship** | a clean, reproducible package | one run script, verified retraining, documented reproducibility |

Every component was measured on the same validation set and kept only if it held up there. On the public leaderboard
the final pipeline scores within 0.2 points of its validation score.

## 4. Architecture

```
raw TSV -> normalise -> blocking (15 key families) -> ~78 pair features
   |
   v
stage 1: LightGBM, two out-of-fold models -> p1
   |
   v
stage 2: context, group evidence, deviation support, word track records
   |
   +-- 3 cross-encoders (MiniLM, multilingual MiniLM, multilingual-e5), fine-tuned on training folds
   v
final stacker: LightGBM on 148 features (+ copy counts, raw spelling)
   |
   v
test-shift corrections -> exclusive assignment -> expected-F0.5 choice per business
   |
   v
second pass for unassigned records -> matching_results.tsv, candidate_pairs.tsv
```

## 5. Candidate generation (blocking)

**Normalisation first.** Legal forms are mapped to canonical tokens and kept aside, digit-for-letter typos are undone,
websites are reduced to their stem, "X dba Y" gives a second name, and address numbers are extracted in order. Indic
scripts are mapped to English with a dictionary **learned from the training pairs**: for true pairs written in, say,
Devanagari, we count which English words of the matching business co-occur with each native word. No external
transliteration resource is used.

**Fifteen key families.** Every record emits short keys: pairs of name tokens, name token + house number, house number
+ address word, the space-free name, consonant skeletons (typo-tolerant), the full name + number for very common names,
and loose name keys for records without an address. Every key includes the country, which is treated as an open set.

**Only rare keys count.** A key is used only if at most 15 businesses carry it and it creates at most 500 pairs (wider
limits for address-less records, whose name is all we have). Each shared key adds a weight `ln(1 + 1000 / (n1 * n2))`;
each business keeps its 50 best pairs plus its 15 best name matches.

**A second pass from the record's side.** Some true copies keep only the first chunk of the address ("A-45, CHENNAI")
and a mangled name, so they never reach their business's top 50. Starting from records that end up unassigned, we
search with keys built for exactly this case (house codes such as "A-45", name word + town, pairs of address words)
and score the pairs with a dedicated model. This recovered 11,383 matches on the test.

| | |
|---|---|
| candidates per business | 48.4 |
| recall of true pairs (validation) | 98.3% |
| scaling | only records sharing a rare key are compared, so cost grows roughly linearly with the data |

## 6. Matching models and features

**Pair features (~78).** String similarities on several views of the name (cleaned, with legal form, without spaces,
dba alias), IDF-weighted token overlap, the rarest token found on one side only, legal-form agreement, address
similarities, house-number agreement and the smallest gap between unmatched numbers (302 vs 306 is the signature of a
neighbour-unit decoy), name frequency, and blocking ranks.

**Stage 1.** LightGBM trained twice on disjoint halves of the training data, so every training pair is scored by a
model that never saw it.

**Stage 2: what a single pair cannot see.**
- *Context*: how the pair ranks among the business's candidates and among the record's possible owners.
- *Group evidence*: is the record a near twin of the business's confident copies?
- *Deviation support*: do the record's differences (an extra word, another number) recur among those copies?
- *Word track records*: how often adding or dropping a given word meant a true match in training. This is how the
  model learns that adding "Holdings" is a planted decoy while adding "Services" is a harmless variation.

**Three cross-encoders**, fine-tuned only on training pairs and used out of sample:

| model | licence | size | input |
|---|---|---|---|
| cross-encoder/ms-marco-MiniLM-L-12-v2 | Apache-2.0 | 33M | normalised text |
| cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | Apache-2.0 | 118M | raw text, any script |
| intfloat/multilingual-e5-base | MIT | 278M | raw text, uncertain pairs only |

**Final stacker.** LightGBM on 148 features: everything above plus each cross-encoder's view of the business's group,
per-source copy counts, and raw-spelling features that keep the punctuation and legal-form spelling normalisation
throws away ("Moque L.L.C." vs "Moque LLC").

## 7. Decision layer

1. **Exclusive assignment**: each record is kept only for its most probable business.
2. **Expected-F0.5 selection**: for each business, output the top k candidates that maximise the expected F0.5,

   `E[F | k] = 1.25 * sum_{i<=k} p_i / (0.25 * sum_all p + k)`, and `E[F | 0] = prod(1 - p_i)`,

   so the model outputs nothing when "this business has no copies" is the better bet.

## 8. Key insights

**A bug that only existed on the test.** The leaderboard trailed validation and no validation experiment explained
why. Comparing, without labels, how many matches the model accepted per confidence level in each country, we found
the US and France full of uncertain matches that India did not have. The cause: a name-popularity feature defined
per 100,000 businesses *of the split*. The US test split is half the size of the training split (663K against 1.32M
businesses), so a unique test name looked like a name shared by two businesses (by four in France). Rescaling it restored the model's confidence on these records from 0.58 to
0.975 (US) and from 0.34 to 0.966 (France), exactly the validation level.

**Learning a language with no labels.** France had no training data, so we measured the generator's habits on the
labelled countries and looked for the same habits in French. Adding "Group" or "Holdings" is always a decoy in
training, so "Groupe" and "Holding" are treated the same way; same-address copies that swap a word for "Services" or
"Partners" are 99.4-99.9% true, so "& Fils" and "& Associés" are trusted. We checked these rules without labels: planted decoys land on
random businesses, so they are far more often a business's *only* match (6-10% against 2.4% for normal matches).

**Correcting for more decoys in the test.** Pairs with the same name and a house number changed by 1-99 are 1.4x to
2.3x more frequent in the test than in validation, while true pairs of that kind are not. A Bayes correction of the
odds, computed from counts alone, removes most of these extra look-alikes.

**Training for the unknown.** French words have no track record, so during training we hid the word evidence on 30%
of the rows. The model learned to fall back on the multilingual cross-encoders exactly when it meets words it has
never seen.

## 9. Engineering under constraints

- **8 GB of RAM for ~10M records per split.** Every heavy step streams Parquet in parts, keys are partitioned by hash
  and businesses by bucket, and a small guard script stops any step before it exhausts memory or disk.
- **Free GPUs.** The cross-encoders ran as Kaggle notebooks on compact data packages; the largest model scores only
  the 3M uncertain pairs instead of all 84M.
- **Compliance.** Only the provided data; open-weight models under 8B parameters; no external data, APIs or test
  labels.
- **Reproducibility.** Retraining the final stacker and the second-pass model with this code reproduces the
  submitted model files byte for byte.

## 10. Results

| | |
|---|---|
| public leaderboard F0.5 | **0.9869** (rank 346 among 18,000+ teams) |
| validation F0.5 (20% held-out businesses, exact metric) | **0.9891** |
| candidate recall (validation) | 98.3% of true pairs (US 98.6%, India 97.8%) with 48.4 candidates per business |
| second pass (test) | 11,383 extra matches recovered for records the first pass left unassigned |

Where the remaining 1.1 validation points go:

| source of loss | F0.5 points |
|---|---|
| records without an address whose name several businesses share (largely undecidable from the data) | 0.64 |
| true copies that blocking never proposes | 0.31 |
| wrong matches | 0.10 |
| true pairs just below the decision rule | 0.07 |

## 11. Output

The pipeline writes two tab-separated files in the challenge format, one row per Source-1 business
(the ids below are placeholders):

```
output/matching_results.tsv
source1_entity_id   matched_entity_ids
S1-1001             S2-2001,S3-3001,S3-3002
S1-1002                                          <- no copies predicted

output/candidate_pairs.tsv
source1_entity_id   candidate_entity_ids
S1-1001             S2-2001,S2-2417,S3-3001,S3-3002,S3-3550,...
```

The submitted test files:

| | |
|---|---|
| Source-1 businesses | 1,732,544 |
| predicted matches | 5,793,569 (3.55 per business that has any) |
| businesses with no predicted copies | 101,358 (5.9%) |
| candidate ids | 83,879,604 (48.4 per business); every predicted match is a candidate |

Both files pass the organisers' format validator and `src/check_candidates.py`. The outputs themselves are derived
from the challenge data and are not part of this repository.

## 12. Pipeline, step by step

```
run_pipeline.sh                 the whole pipeline, steps 1-11
requirements.txt                pinned packages
src/                            one script per step (table below); src/kaggle/ holds the three GPU notebooks
docs/APPROACH.md                detailed write-up (also as PDF)
docs/EDA.md                     data analysis
```

`run_pipeline.sh` runs these steps in order. Each step reads the files of the previous ones from `work/`.

| step | scripts | what it does | writes |
|---|---|---|---|
| 1 | prepare_data.py, normalize.py | TSV to Parquet with integer ids; normalised names and addresses; folds (fold 0 = validation) | work/parquet, work/norm |
| 2 | blocking.py | 15 key families, rare keys only, top 50 per business plus 15 name matches | work/blocking |
| 3 | features.py | ~78 features per candidate pair | work/features_* |
| 4 | stage2.py | stage-1 LightGBM (out-of-fold), stage-2 features, candidate file | work/s2te_*, work/candidates |
| 5 (GPU) | kaggle_pack.py, kaggle/xenc_kaggle.py | cross-encoder 1: ms-marco-MiniLM-L-12-v2, normalised text | work/kaggle_out |
| 6 (GPU) | kaggle_pack2.py, kaggle/xenc2_kaggle.py | cross-encoder 2: mmarco-mMiniLMv2-L12-H384-v1, raw text | work/kaggle_out2 |
| 7 (GPU) | stacker.py prep, kaggle_pack3.py, kaggle/xenc3_kaggle.py | cross-encoder 3: multilingual-e5-base, uncertain pairs | work/kaggle_out3 |
| 8 | stacker.py, cc_features.py, rw_features.py | final stacker, plus a France model without the English-only cross-encoder | work/models |
| 9 | record_side.py build, train | second-pass retrieval model | work/rside_*, work/models/record_side.txt |
| 10 | freq_fix.py, stage2.py, cc_features.py, rw_features.py, stacker.py predict | test name frequency on the training scale, test side rebuilt and scored | work/models/testpred_* |
| 11 | final_assemble.py, fr_street_lost.py, record_side.py apply, extend_candidates.py, check_candidates.py | test-shift corrections, final decision, second pass, candidate file, checks | output/ |

Helpers: `decide.py` (exclusive assignment, expected-F0.5 selection, TSV writers), `metric.py` (challenge metric),
`train.py` (LightGBM settings), `guard.sh` (stops a step before it exhausts memory or disk).

## 13. Reproducing the results

**Data.** The challenge data belongs to the organisers and is not included in this repository. With access to it,
place the files as:

```
Dataset/Train files tsv/   train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
Dataset/Test files tsv/    test_source1.tsv   test_source2.tsv   test_source3.tsv
```

**Run.**

```bash
pip install -r requirements.txt     # Python 3.13
bash run_pipeline.sh                # from step 1
bash run_pipeline.sh 6              # resume at step 6 after a GPU step
```

The script stops at the three GPU steps (5-7), prints which notebook to run and where to put its output, and resumes
from the step you give it. Intermediate files go to `work/` (about 40 GB), results to `output/`.

**GPU steps on Kaggle** (T4, free tier):

1. Upload each package folder as a private dataset: `kaggle datasets init -p work/kaggle_pack`, set a title and id
   in the generated `dataset-metadata.json`, then `kaggle datasets create -p work/kaggle_pack` (the step-7 package
   already contains a metadata file; change its id to your account). Step 5 runs offline, so first add the model to
   its package:
   `huggingface-cli download cross-encoder/ms-marco-MiniLM-L-12-v2 --local-dir work/kaggle_pack`
2. Create a notebook from the script in `src/kaggle/`, turn the GPU on and attach the datasets:
   step 5: kaggle_pack; step 6: kaggle_pack and kaggle_pack2 (internet on); step 7: kaggle_pack2 and kaggle_pack3
   (internet on).
3. Download the notebook output (`*_scores.parquet`) into the folder shown in the table above and resume the script.

The CPU steps run on an 8 GB laptop; the GPU notebooks take a few hours each (2.6 h and 2.8 h for the first two).
GPU training is not bit-exact, so a rebuild reproduces the method and the score level rather than the exact submitted
file; the CPU-trained models (final stacker, second-pass model) retrain byte for byte. See
[`docs/APPROACH.md`](docs/APPROACH.md) for the full method and [`docs/EDA.md`](docs/EDA.md) for the data analysis.

## 14. Team

**Team ARCA**: Eklavya Singh, Kamran Ansari, Ayush Singh.

Released under the MIT licence (see `LICENSE`).
