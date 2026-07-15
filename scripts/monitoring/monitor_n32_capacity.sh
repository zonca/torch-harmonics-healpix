#!/bin/bash
# Poll Expanse for hc=64 configs 3-4, pull JSONs, regenerate figures when all 4 are local.
REPO="/home/zonca/zonca/p/software/project_work/torch-harmonics-healpix"
RDIR="$REPO/results_v3"
PYTHON="$REPO/.venv/bin/python"
REMOTE="/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v3"
LOG="$REPO/results_v3/monitor.log"

CONFIGS="fsky1.0_noise0 fsky1.0_noise6 fsky0.1_noise0 fsky0.1_noise6"
MAX_WAIT=28800  # 8h
START=$(date +%s)

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    if [ $ELAPSED -gt $MAX_WAIT ]; then
        echo "[$(date)] Timeout ($((ELAPSED/3600))h)." >> "$LOG"
        break
    fi

    DONE_COUNT=0
    for CFG in $CONFIGS; do
        LOCAL="$RDIR/test4_nside32_hc64_${CFG}.json"
        if [ ! -f "$LOCAL" ]; then
            scp "expanse:$REMOTE/test4_nside32_hc64_${CFG}.json" "$LOCAL" 2>/dev/null
            if [ -f "$LOCAL" ]; then
                echo "[$(date)] Pulled hc64 $CFG" >> "$LOG"
            fi
        else
            DONE_COUNT=$((DONE_COUNT + 1))
        fi
    done

    if [ $DONE_COUNT -eq 4 ]; then
        echo "[$(date)] All 4 hc=64 configs local! Regenerating figures + summary..." >> "$LOG"
        cd "$REPO"
        $PYTHON scripts/generate_n32_publication_figures.py >> "$LOG" 2>&1
        $PYTHON scripts/compile_n32_summary.py >> "$LOG" 2>&1
        echo "[$(date)] Figures and summary regenerated." >> "$LOG"
        break
    fi

    sleep 300
done
