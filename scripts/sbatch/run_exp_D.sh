#!/bin/bash
#SBATCH --job-name=expD_es_gd
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_fellow_binxuwang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation/logs/expD_%j.out
#SBATCH --error=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation/logs/expD_%j.err

# Usage:
#   sbatch run_exp_D.sh                         # default settings
#   sbatch --export=ALL,TAG=sigma05 run_exp_D.sh # with custom tag
#   sbatch --array=0-4 run_exp_D.sh             # sweep (see SIGMA_VALS below)

set -e
module load python/3.10.9-fasrc01 cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01
source activate base  # or your conda env

REPO=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Github/EvolStrategyTheory
OUT=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation
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
    --d 1000 \
    --k 20 \
    --N 50 \
    --sigma $SIGMA \
    --xi 0.0 \
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
