#!/bin/bash
#SBATCH --job-name=expE_ou_var
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_fellow_binxuwang
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1
#SBATCH --mem=32G --time=03:00:00
#SBATCH --output=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation/logs/expE_%j.out
#SBATCH --error=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation/logs/expE_%j.err

# Usage:
#   sbatch run_exp_E.sh                         # default σ=0.1
#   sbatch --array=0-5 run_exp_E.sh             # sweep σ ∈ {0.01,0.05,0.1,0.2,0.5,1.0}
#   sbatch --export=ALL,SIGMA=0.2 run_exp_E.sh  # custom σ

set -e
module load python/3.10.9-fasrc01 cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01
source activate base

REPO=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Github/EvolStrategyTheory
OUT=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation
mkdir -p $OUT/logs $OUT/figures $OUT/data

cd $REPO

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    SIGMA_VALS=(0.01 0.05 0.1 0.2 0.5 1.0)
    SIGMA=${SIGMA_VALS[$SLURM_ARRAY_TASK_ID]}
    TAG="sigma${SIGMA//./_}"
else
    SIGMA=${SIGMA:-0.1}
    TAG=${TAG:-""}
fi

echo "Running Exp E: sigma=$SIGMA tag=$TAG"

python scripts/exp_E_ou_variance_trajectory.py \
    --d 1000 --k 20 \
    --lam_max 5.0 --lam_min 0.1 \
    --spectrum powerlaw --beta 1.0 \
    --N 50 --sigma $SIGMA --xi 0.0 --ddof 0 \
    --T 600 --n_trials 300 --theta0_norm 10.0 \
    --out_dir $OUT --tag "$TAG" \
    --sigma_list "0.01,0.05,0.1,0.2,0.5,1.0" \
    --exps E1,E2,E3,E4

echo "Done: $TAG"
