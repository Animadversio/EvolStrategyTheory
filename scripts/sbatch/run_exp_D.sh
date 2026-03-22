#!/bin/bash
#SBATCH --job-name=expD_es_gd
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_binxuwang_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=holygpu8a19205
#SBATCH --output=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation_ddof0_v2/logs/expD_%j.out
#SBATCH --error=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation_ddof0_v2/logs/expD_%j.err

# Usage:
#   sbatch run_exp_D.sh                         # default settings
#   sbatch --export=ALL,TAG=sigma05 run_exp_D.sh # with custom tag
#   sbatch --array=0-4 run_exp_D.sh             # sweep (see SIGMA_VALS below)

set -e
source /n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh
source /n/sw/Miniforge3-24.11.3-0/etc/profile.d/mamba.sh
export CONDA_ENVS_PATH=/n/home12/binxuwang/.conda/envs
conda activate torch2

REPO=/n/home12/binxuwang/Github/EvolStrategyTheory
OUT=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation_ddof0_v2
mkdir -p $OUT/logs $OUT/figures $OUT/data

cd $REPO

# ── Sweep over sigma if running as array job ──
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    SIGMA_VALS=(0.05 0.1 0.2 0.5 1.0)
    SIGMA=${SIGMA_VALS[$SLURM_ARRAY_TASK_ID]}
    TAG="sigma${SIGMA//./_}"
    echo "Array task $SLURM_ARRAY_TASK_ID → sigma=$SIGMA tag=$TAG"
else
    SIGMA=${SIGMA:-0.1}
    TAG=${TAG:-""}
fi

python scripts/exp_D_multistep_gd_comparison.py \
    --d 5000 \
    --k 2500 \
    --flat 0.0 \
    --N 30 \
    --sigma $SIGMA \
    --xi 0.1 \
    --T 500 \
    --n_trials 300 \
    --theta0_norm 10.0 \
    --spectrum powerlaw \
    --beta 1.0 \
    --lam_max 5.0 \
    --lam_min 0.1 \
    --ddof 0 \
    --out_dir $OUT \
    --tag "$TAG" \
    --exps D1,D2,D3,D4

echo "Done: $TAG"
