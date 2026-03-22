# Scripts — ZScoreES Theory Validation

All scripts validate propositions from `theory.tex` (Qiu et al. 2025).

## Environment

```bash
# Activate the research conda env (has torch, numpy, matplotlib, pandas)
conda activate research   # or: ~/miniforge3/envs/research/bin/python

# Works on local GPU or FASRC Kempner H100 cluster
```

---

## Experiments Overview

| Script | Validates | Args | Cluster |
|--------|-----------|------|---------|
| `validate_theory.py` | Props 1–3 + multi-step breakdown (Exps 1–4) | hardcoded | ❌ |
| `exp_A_spectrum.py`  | Spectrum effects on σ_R (Exp A) | hardcoded | ❌ |
| `exp_B_snr_multistep.py` | ρ heatmap, N* scaling (Exp B) | hardcoded | ❌ |
| `exp_C_optimizer_comparison.py` | ZScoreES vs SeparableES vs SimpleES (Exp C) | hardcoded | ❌ |
| **`exp_D_multistep_gd_comparison.py`** | Multi-step theory + ES vs GD (Exp D) | `argparse` | ✅ |

---

## exp_D_multistep_gd_comparison.py — Full CLI

The most complete script. Parameterized for cluster sweeps.

### Sub-experiments

| Sub-exp | What | Theory tested |
|---------|------|---------------|
| **D1** | Mean trajectory E[θ_t] per eigenmode vs (I-γ₀Q)^t θ₀ | Frozen-σ_R approx |
| **D2** | Per-mode log decay slope vs log(1-γ₀λ_i) | Eigenmode decay rates |
| **D3** | Stationary Var[c_i(∞)] vs OU theory | Noise floor |
| **D4** | ‖θ_t‖ vs steps & function evals: ES vs GD | Sample efficiency |

### Quick local run

```bash
python scripts/exp_D_multistep_gd_comparison.py \
    --d 500 --k 20 --N 50 --sigma 0.1 \
    --T 400 --n_trials 200
```

### Full cluster-scale run

```bash
python scripts/exp_D_multistep_gd_comparison.py \
    --d 1000 --k 20 --N 50 --sigma 0.1 --xi 0.0 \
    --T 500 --n_trials 300 --theta0_norm 10.0 \
    --spectrum powerlaw --beta 1.0 \
    --lam_max 5.0 --lam_min 0.1 --ddof 0 \
    --out_dir /path/to/output --tag run1
```

### Sigma sweep (sbatch array)

```bash
sbatch --array=0-4 scripts/sbatch/run_exp_D.sh
# Runs sigma ∈ {0.05, 0.1, 0.2, 0.5, 1.0} in parallel
```

### Run only specific sub-experiments

```bash
python scripts/exp_D_multistep_gd_comparison.py --exps D1,D2
```

### Key arguments

```
Landscape:
  --d          Dimension                         [default: 500]
  --k          Active (curved) eigenmodes        [default: 20]
  --lam_max    Largest eigenvalue                [default: 5.0]
  --lam_min    Smallest active eigenvalue        [default: 0.5]
  --spectrum   uniform | powerlaw | range        [default: powerlaw]
  --beta       Power-law exponent                [default: 1.0]

ZScoreES:
  --N          Population size                   [default: 50]
  --sigma      Exploration std σ                 [default: 0.1]
  --xi         Observation noise ξ               [default: 0.0]
  --ddof       Std denominator: 0=N, 1=N-1       [default: 0]

Simulation:
  --T          Steps per trial                   [default: 400]
  --n_trials   Independent trajectories          [default: 200]
  --theta0_norm  ‖θ_0‖                           [default: 10.0]

Output:
  --out_dir    Output base directory             [default: auto]
  --tag        Tag appended to filenames         [default: ""]
  --exps       Sub-experiments: D1,D2,D3,D4     [default: all]
```

---

## Output Structure

```
DL_Projects/EvolStrategyTheory_validation/
├── figures/
│   ├── exp1_flat.png
│   ├── expA1_sigmaR_k_sweep.png
│   ├── expB1_rho_heatmap.png
│   ├── expC12_optimizer_comparison.png
│   ├── expD1_mean_trajectory[_tag].png
│   ├── expD2_eigenmode_decay[_tag].png
│   ├── expD3_noise_floor[_tag].png
│   └── expD4_es_vs_gd[_tag].png
└── data/                          ← intermediate compute results (pkl)
    ├── expD_simulation[_tag].pkl  ← raw traj_coords (n_trials × T+1 × d)
    ├── expD1_mean_trajectory[_tag].pkl
    ├── expD2_eigenmode_decay[_tag].pkl
    ├── expD3_noise_floor[_tag].pkl
    └── expD4_es_vs_gd[_tag].pkl
```

**Convention:** compute scripts save to `data/*.pkl`; plotting loads from pkl.
This separates expensive computation from visualization — replot freely without re-running.

---

## Cluster (FASRC Kempner)

```bash
# Single run
sbatch scripts/sbatch/run_exp_D.sh

# Sigma sweep
sbatch --array=0-4 scripts/sbatch/run_exp_D.sh

# Logs
tail -f /n/holylfs06/.../EvolStrategyTheory_validation/logs/expD_<jobid>.out
```

Output dir on cluster:
```
/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/EvolStrategyTheory_validation/
```
