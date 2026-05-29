#!/bin/bash
#SBATCH --job-name=thh-restripe
#SBATCH --partition=shared
#SBATCH --account=sds275
#SBATCH --qos=shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/restripe_%j.out

# Re-stripe HDF5 files for better Lustre I/O performance
# Original files have stripe_count=1 (single OST, ~80 MB/s)
# Copied to stripe_count=16 directory (16 OSTs, ~1+ GB/s expected)

set -e
export PYTHONUNBUFFERED=1

SRC=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_data
DST=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/hdf5_striped

echo "=== Re-striping HDF5 files ==="
date
hostname

for f in $SRC/*.h5; do
    fname=$(basename $f)
    echo "Copying $fname..."
    t0=$(date +%s)
    cp $f $DST/$fname
    t1=$(date +%s)
    size=$(stat -c%s $DST/$fname)
    echo "  Done: $(($size/1073741824)) GB in $(($t1-$t0))s ($(($size/($t1-$t0)/1048576)) MB/s)"
    echo "  Stripe info:"
    lfs getstripe $DST/$fname | head -4
done

echo "=== Re-striping complete ==="
date
