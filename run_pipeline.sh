#!/bin/bash
# Full pipeline (team ARCA). Writes output/matching_results.tsv and output/candidate_pairs.tsv.
# Run from the repository root, with the challenge data in Dataset/ (see README). Intermediate files go to work/.
# Steps 5-7 run on a GPU (Kaggle notebooks, see README); the script stops there and says where to resume:
#   bash run_pipeline.sh <step>
set -e
S=src
PY=${PY:-python3}
FROM=${1:-1}
run() { local mem=$1; shift; MAX_MEM_MB=$mem bash $S/guard.sh $PY "$@"; }   # memory / disk guard
step() { [ "$1" -ge "$FROM" ]; }
gpu() { echo; echo ">>> [GPU] run $1 on Kaggle with $2, put its output in $3,"
        echo ">>> then continue with: bash run_pipeline.sh $4"; exit 0; }

# 1. data and normalisation
if step 1; then
  run 4000 $S/prepare_data.py --train-dir "Dataset/Train files tsv" --test-dir "Dataset/Test files tsv"
  run 6500 $S/normalize.py
fi

# 2. blocking
if step 2; then
  run 6500 $S/blocking.py --split train
  run 6500 $S/blocking.py --split test
fi

# 3. pair features
if step 3; then
  run 6000 $S/features.py --split train
  run 6000 $S/features.py --split test
fi

# 4. stage 1 + stage-2 features, candidate file
if step 4; then
  for c in oof p1 ctx group te_build te_apply candidates; do run 6500 $S/stage2.py $c; done
fi

# 5-7. cross-encoders (GPU)
if step 5; then
  run 2500 $S/kaggle_pack.py
  echo ">>> add the model to the package first (the notebook runs offline):"
  echo ">>>   huggingface-cli download cross-encoder/ms-marco-MiniLM-L-12-v2 --local-dir work/kaggle_pack"
  gpu src/kaggle/xenc_kaggle.py work/kaggle_pack work/kaggle_out 6
fi
if step 6; then run 2500 $S/kaggle_pack2.py; gpu src/kaggle/xenc2_kaggle.py "work/kaggle_pack + work/kaggle_pack2" work/kaggle_out2 7; fi
if step 7; then
  run 5000 $S/stacker.py prep
  run 2500 $S/kaggle_pack3.py
  gpu src/kaggle/xenc3_kaggle.py "work/kaggle_pack2 + work/kaggle_pack3" work/kaggle_out3 8
fi

# 8. final stacker, plus a France model without the English-only cross-encoder
if step 8; then
  run 5000 $S/stacker.py prep
  run 6500 $S/cc_features.py
  run 6500 $S/rw_features.py
  run 7000 $S/stacker.py train --tag main
  run 7000 $S/stacker.py train --tag france --drop "xenc,xenc_*"
fi

# 9. second-pass retrieval model
if step 9; then
  run 6800 $S/record_side.py build
  run 6500 $S/record_side.py train
fi

# 10. test side: name frequency on the training scale, rebuild, predict
if step 10; then
  run 4000 $S/freq_fix.py
  export SPLITS=test
  for c in p1 ctx group te_apply; do run 6500 $S/stage2.py $c; done
  run 6500 $S/cc_features.py
  run 6500 $S/rw_features.py
  unset SPLITS
  run 6000 $S/stacker.py predict --tag main
  run 6000 $S/stacker.py predict --tag france
fi

# 11. final decision, candidate file, checks
if step 11; then
  run 5000 $S/final_assemble.py --out work/final_a
  run 5000 $S/fr_street_lost.py --inp work/final_a/matching_results.tsv --out work/final_b/matching_results.tsv
  run 6500 $S/record_side.py apply --base work/final_b/matching_results.tsv --out work/final
  mkdir -p output
  cp work/final/matching_results.tsv output/matching_results.tsv
  run 6000 $S/extend_candidates.py --cand work/candidates/candidate_pairs.tsv --added work/final/added_pairs.parquet \
      --out output/candidate_pairs.tsv
  run 7000 $S/check_candidates.py "Dataset/Test files tsv" output/candidate_pairs.tsv output/matching_results.tsv
  echo "done: output/matching_results.tsv, output/candidate_pairs.tsv"
fi
