# Data analysis

First look at the challenge data (25 Sep 2026), computed on the Parquet files written by `src/prepare_data.py`.
These findings shaped the blocking keys, the features and the validation design.

## Size and structure
| | S1 | S2 | S3 |
|---|---|---|---|
| train | 2,206,821 (US 60%, India 40%) | 5,034,616 | 5,285,603 |
| test | 1,732,544 (India 47%, US 38%, France 15%) | 4,887,273 | 5,082,316 |

- Singletons 5.6%; mean 3.46 matches per S1 (0-11); S2 0-5, S3 0-6 per S1.
- Every matched S2/S3 record belongs to exactly one S1 (7,638,365 unique pairs).
- ~26% of train S2/S3 records match nothing (distractors).
- (S2+S3)/S1 ratio: train 4.68; test 5.5-5.8 in every country -> test likely has more distractors.

## Matching signal
- Country label identical in 100% of true pairs.
- Normalised names identical in only 25.8% of true pairs.
- Name token-set similarity < 50 in 3.8% of true pairs; address < 50 in ~4.6%; both < 50 in 0.01%.
- 90% of true pairs share a name token; 92% share an address number (when both have numbers); 1.6% share neither.
- Postal codes nearly absent (<1% of addresses).
- 39% of S1 names are shared by at least one other S1 (e.g. "primary care group" x253).

## Noise catalogue (share of S2+S3 rows)
ALL-CAPS names 17.5%, brackets 7.9%, accented Latin 6.0%, website-like names 4.0%, leading junk (`--`, `<<`, `#`) 2.2%, DBA/a.k.a. 0.7%, pipe suffix 0.3%, null address 3.3%, literal "null" in address 2.5%, "near/opp" landmarks 3.5%, PO Box 1.0%.

Positive-pair noise also includes random alias names at the same address, acronyms, generic-word swaps (Shop->Center), titles (Dr, Smt, M/s), house-number perturbation (809->807, 705->705-C), zero padding, city typos/suffixes, state code  /  full name  /  native script, dropped address components.

## Hard negatives
Distractors deliberately copy an S1's building with a mutated name token and/or unit number ("Es Clinic, Flat 302" vs "Es Clinic Overseas Co, Flat 306"), or sit at an identical address with a different name. 21.8% of distractors with a house-number+street key share it with some S1.

## Scripts
S2/S3 names in 10 Indic scripts (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya, Gurmukhi) + accented Latin; 752,869 train records. anyascii alone reaches median token-set similarity 75 against the English S1 name ("klasik kmsltemsi elelpi" vs "Classic Consultancy LLP") -> learn a native->English token dictionary from training pairs.

## France (test only)
~15 cities in 3 regions -> dense; one name can repeat at dozens of addresses in one town, so house number + street decide. S2: upper case, R/AV/ALL/IMP/BLVD/BIS abbreviations. S3: No./Nº/#, zero-padded numbers, département instead of region. Accent injection in names (Sàrl, Çlub).
