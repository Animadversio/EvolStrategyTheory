#!/bin/bash
#SBATCH --job-name=zscore_exps
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH --exclude=holygpu8a19205
#SBATCH -o /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation_ddof0_v2/%j.log
#SBATCH -e /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation_ddof0_v2/%j.err

bash /n/home12/binxuwang/Github/EvolStrategyTheory/scripts/run_all_exps.sh
