#!/bin/bash
# Poll Expanse for config 4 (fsky=0.1, noise=6) completion.
# When done, pull all 4 hc=64 JSONs and regenerate figures.

REPO="/home/zonca/zonca/p/software/project_work/torch-harmonics-healpix"
RDIR="$REPO/results_v3"
REMOTE="expanse:/expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v3"
CONFIG4="test4_nside32_hc64_fsky0.1_noise6.json"
MAX_WAIT=28800  # 8 hours

echo "=== Monitoring config 4 (fsky=0.1, noise=6) ==="
echo "Started: $(date)"

for i in $(seq 1 32); do  # 32 x 15min = 8 hours
    sleep 900  # 15 min
    
    # Check if config 4 JSON exists on Expanse
    if ssh expanse "test -f /expanse/lustre/scratch/zonca/temp_project/torch-harmonics-healpix/results_v3/$CONFIG4" 2>/dev/null; then
        echo "$(date): Config 4 complete! Pulling results..."
        
        # Pull all hc=64 JSONs
        for f in fsky1.0_noise0 fsky1.0_noise6 fsky0.1_noise0 fsky0.1_noise6; do
            scp "$REMOTE/test4_nside32_hc64_${f}.json" "$RDIR/" 2>/dev/null
            echo "  Pulled $f"
        done
        
        # Regenerate figures
        echo "$(date): Regenerating figures..."
        cd "$REPO"
        .venv/bin/python scripts/generate_n32_publication_figures.py 2>&1
        
        echo "$(date): Done! All 4 hc=64 configs available."
        exit 0
    fi
    
    echo "$(date): Config 4 still training..."
done

echo "$(date): Timeout reached. Check training status manually."
