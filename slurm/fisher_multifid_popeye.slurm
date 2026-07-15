#!/bin/bash
#SBATCH --job-name=thh-fisher-mfid
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/azonca/torch-harmonics-healpix/logs/fisher_multifid_%j.out

# Fisher (Cramer-Rao) bounds at off-center fiducial points, fixed lmax=500,
# matching the multi-fiducial CNN evaluation on Expanse
# (slurm/eval_fiducial_v3_expanse.slurm). CPU only.

set -e
export PYTHONUNBUFFERED=1

date
hostname

module load python/3.11.11
source ~/torch-hh-venv/bin/activate

cd ~/torch-harmonics-healpix
git pull

FIDS="0.001:0.054 0.006:0.054 0.003:0.040 0.003:0.070"

for FID in $FIDS; do
  R=${FID%%:*}; TAU=${FID##*:}
  OUTDIR=/mnt/home/azonca/torch-harmonics-healpix/results_v3/fisher_multifid/r${R}_tau${TAU}
  mkdir -p $OUTDIR
  for NS in 16 32 128; do
    for CFG in "1.0 0" "1.0 6" "0.1 0" "0.1 6"; do
      set -- $CFG
      FSKY=$1; NOISE=$2
      echo "=== Fisher NSIDE=$NS fsky=$FSKY noise=$NOISE r=$R tau=$TAU ==="
      python3 scripts/fisher_forecast.py \
        --nside $NS --f_sky $FSKY --noise_std $NOISE \
        --r $R --tau $TAU \
        --output $OUTDIR/test4_fisher_nside${NS}_fsky${FSKY}_noise${NOISE}.json
    done
  done
done

echo "=== fisher_multifid complete ==="
date
