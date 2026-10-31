#!/usr/bin/env bash
# One-shot MAPPO district training for a time-boxed cloud notebook (Kaggle / Colab).
# See docs/training/kaggle.md. Run from the repository root:
#
#     bash scripts/kaggle_train.sh
#
# Tunables (environment variables):
#     OUT           where checkpoints + the final model go   (default /kaggle/working/atlas_out)
#     ITERATIONS    upper bound; the time budget normally ends the run first   (default 3000)
#     NUM_STEPS     decisions collected per iteration                          (default 256)
#     MAX_HOURS     stop cleanly after this wall-clock budget (default 8.5: Kaggle's session
#                   limit is ~9-12 h; leave headroom for the final save)
#
# Re-running the same command in a new session RESUMES from $OUT/checkpoints (copy that folder
# back in first -- notebook disks are wiped between sessions).
set -euo pipefail

OUT="${OUT:-/kaggle/working/atlas_out}"
mkdir -p "$OUT/checkpoints"

pip install -q -r docs/training/requirements-kaggle.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# The raw OSM extract is committed; the network and demand are rebuilt here.
python scripts/build_district_network.py
python scripts/generate_demand.py

python scripts/train_mappo_district.py \
    --iterations "${ITERATIONS:-3000}" \
    --num-steps "${NUM_STEPS:-256}" \
    --max-hours "${MAX_HOURS:-8.5}" \
    --checkpoint-dir "$OUT/checkpoints" \
    --model-path "$OUT/mappo_district.pt" \
    --no-mlflow

echo
echo "Done. Download from $OUT:"
ls -la "$OUT" "$OUT/checkpoints"
echo "Then locally:  ATLAS_RL_CHECKPOINT=<path to mappo_district.pt> make demo"
echo "and evaluate:  python scripts/run_evaluation_pilot.py --rl-checkpoint <path>"
