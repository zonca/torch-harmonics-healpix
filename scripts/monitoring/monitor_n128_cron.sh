#!/bin/bash
# Cron script: check N128 hc=32 training on Expanse
# Outputs summary to stdout; silent if nothing to report
set -e

EXPATH=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix

# Check queue
JOBS=$(ssh -o ConnectTimeout=15 -o ControlPath=~/.ssh/sockets/%r@%h-%p -o ControlMaster=no expanse \
  'squeue -u zonca -o "%.10i %.8T %.10M %.20j" 2>/dev/null || echo "SSH_FAILED"' 2>&1)

if echo "$JOBS" | grep -q "SSH_FAILED"; then
  echo "SSH to Expanse failed"
  exit 0
fi

RUNNING=$(echo "$JOBS" | grep -c "RUNNING" || true)
PENDING=$(echo "$JOBS" | grep -c "PENDING" || true)

if [ "$RUNNING" -eq 0 ] && [ "$PENDING" -eq 0 ]; then
  # No jobs — check results
  RESULTS=$(ssh -o ConnectTimeout=15 expanse \
    "ls $EXPATH/results_v3/test4_nside128_hc32_*.json 2>/dev/null | wc -l" 2>&1)
  
  if [ "$RESULTS" -ge 4 ]; then
    echo "All 4 N128 hc=32 configs complete!"
  else
    echo "WARNING: No jobs queued but only $RESULTS/4 configs done. May need resubmit."
  fi
  exit 0
fi

# Jobs still running — report progress
echo "$RUNNING running, $PENDING pending"
ssh -o ConnectTimeout=15 expanse \
  "for f in $EXPATH/logs/train_n128_v3_hc32_5*.out; do
    BASE=\$(basename \$f .out)
    JOBID=\${BASE##*_}
    STATE=\$(squeue -j \$JOBID -ho '%T' 2>/dev/null || echo 'DONE')
    if [ \"\$STATE\" = 'DONE' ] || [ \"\$STATE\" = 'CANCELLED' ]; then continue; fi
    LAST_EPOCH=\$(grep -E '^\s+[0-9]+\s+\|' \$f 2>/dev/null | tail -1 | awk '{print \$1, \$3, \$5}')
    echo \"  Job \$JOBID (\$STATE): epoch \$LAST_EPOCH\"
  done" 2>&1
