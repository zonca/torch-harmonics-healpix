#!/bin/bash
# Monitor N128 hc=32 training jobs on Expanse
# Checks job status, logs progress, and flags errors
# Run via cron every 1-2 hours

REPO=~/zonca/p/software/project_work/torch-harmonics-healpix
LOG=$REPO/results_v3/n128_monitor.log
EXPATH=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix

echo "=== $(date) ===" >> "$LOG"

# Check job queue
JOBS=$(ssh -o ConnectTimeout=15 -o ControlPath=~/.ssh/sockets/%r@%h-%p -o ControlMaster=no expanse \
  'squeue -u zonca -o "%.10i %.8T %.10M %.20j"' 2>&1)

echo "$JOBS" >> "$LOG"

# If no jobs running/pending, check if results exist
RUNNING=$(echo "$JOBS" | grep -c "RUNNING" || true)
PENDING=$(echo "$JOBS" | grep -c "PENDING" || true)

if [ "$RUNNING" -eq 0 ] && [ "$PENDING" -eq 0 ]; then
  echo "No jobs in queue. Checking results..." >> "$LOG"
  RESULTS=$(ssh -o ConnectTimeout=15 expanse \
    "ls $EXPATH/results_v3/test4_nside128_hc32_*.json 2>/dev/null | wc -l" 2>&1)
  echo "N128 hc=32 result files: $RESULTS" >> "$LOG"
  
  if [ "$RESULTS" -lt 4 ]; then
    echo "WARNING: Only $RESULTS/4 configs complete. May need to resubmit." >> "$LOG"
  else
    echo "All 4 configs complete!" >> "$LOG"
  fi
fi

# Check for errors in running job logs
ssh -o ConnectTimeout=15 expanse \
  "for f in $EXPATH/logs/train_n128_v3_hc32_*.out; do
    if grep -qi 'error\|killed\|oom\|cuda error' \"\$f\" 2>/dev/null; then
      echo \"ERROR in \$(basename \$f):\"
      grep -i 'error\|killed\|oom\|cuda error' \"\$f\" | tail -3
    fi
  done" 2>&1 >> "$LOG"

echo "" >> "$LOG"
