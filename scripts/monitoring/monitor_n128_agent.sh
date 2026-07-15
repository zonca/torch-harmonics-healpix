#!/bin/bash
# Monitor N128 hc=32 training on Expanse, report status
set -e

EXPATH=/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix
REPO=~/zonca/p/software/project_work/torch-harmonics-healpix

# Check queue
JOBS=$(ssh -o ConnectTimeout=15 -o ControlPath=~/.ssh/sockets/%r@%h-%p -o ControlMaster=no expanse \
  'squeue -u zonca -o "%.10i %.8T %.10M %.20j" 2>/dev/null || echo "SSH_FAILED"' 2>&1)

if echo "$JOBS" | grep -q "SSH_FAILED"; then
  echo "❌ SSH to Expanse failed"
  exit 1
fi

RUNNING=$(echo "$JOBS" | grep -c "RUNNING" || true)
PENDING=$(echo "$JOBS" | grep -c "PENDING" || true)

if [ "$RUNNING" -eq 0 ] && [ "$PENDING" -eq 0 ]; then
  # No jobs — check results
  RESULTS=$(ssh -o ConnectTimeout=15 expanse \
    "ls $EXPATH/results_v3/test4_nside128_hc32_*.json 2>/dev/null | wc -l" 2>&1)
  
  if [ "$RESULTS" -ge 4 ]; then
    echo "✅ All 4 N128 hc=32 configs complete! Results in $EXPATH/results_v3/"
  else
    echo "⚠️ No jobs queued but only $RESULTS/4 configs done. Need to resubmit."
  fi
else
  # Jobs exist — check progress
  echo "📊 $RUNNING running, $PENDING pending"
  
  # Get latest epoch from running job log
  LATEST=$(ssh -o ConnectTimeout=15 expanse \
    "for f in $EXPATH/logs/train_n128_v3_hc32_*.out; do
      echo \"--- \$(basename \$f) ---\"
      grep -E '^\s+[0-9]+\s+\|' \"\$f\" 2>/dev/null | tail -1
      # Check for errors
      if grep -qi 'error\|killed\|oom\|cuda error' \"\$f\" 2>/dev/null; then
        echo '  ❌ ERRORS FOUND'
      fi
    done" 2>&1)
  echo "$LATEST"
fi
