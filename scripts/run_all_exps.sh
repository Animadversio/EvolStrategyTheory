#!/bin/bash
# Run all ZScoreES validation experiments sequentially, with timing.
set -e

REPO="/n/home12/binxuwang/Github/EvolStrategyTheory"
source /n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh
source /n/sw/Miniforge3-24.11.3-0/etc/profile.d/mamba.sh
export CONDA_ENVS_PATH=/n/home12/binxuwang/.conda/envs
conda activate torch2

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $REPO

echo "=============================="
echo "Starting Exp 1: Flat landscape"
echo "=============================="
START1=$SECONDS
python scripts/exp1_flat.py
echo "Exp 1 wall time: $((SECONDS - START1))s"

echo ""
echo "================================"
echo "Starting Exp 2: Linear landscape"
echo "================================"
START2=$SECONDS
python scripts/exp2_linear.py
echo "Exp 2 wall time: $((SECONDS - START2))s"

echo ""
echo "==================================="
echo "Starting Exp 3: Quadratic landscape"
echo "==================================="
START3=$SECONDS
python scripts/exp3_quadratic.py
echo "Exp 3 wall time: $((SECONDS - START3))s"

echo ""
echo "========================="
echo "Generating report"
echo "========================="
python scripts/gen_report.py

echo ""
echo "ALL DONE. Total wall time: ${SECONDS}s"
