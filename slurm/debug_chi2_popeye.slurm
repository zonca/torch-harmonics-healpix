#!/bin/bash
#SBATCH --job-name=thh-debug-chi2
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/logs/debug_chi2_%j.out

source ~/torch-hh-venv/bin/activate
cd ~/torch-harmonics-healpix
python3 scripts/debug_mcmc_chi2.py
