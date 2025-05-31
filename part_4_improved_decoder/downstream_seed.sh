#!/usr/bin/env bash

# List of the first half of your simclr_* directories:
DIRS=(
  "simclr_decoder_larger_specific_decoder_train_flip_l1_decay"
  "autoencoder_l1_group_norm"
  "simclr_decoder_larger_specific_decoder_first_idea"  
)

MAX_JOBS=3
job_count=0

for dir in "${DIRS[@]}"; do
  if [ -d "$dir" ]; then
    (
      echo "====> [GPU 0] Starting  $dir"
      cd "$dir" || { echo "Failed to cd into $dir"; exit 1; }

      # Run downstream_seed.py and redirect stdout/stderr into downstream_seed.log
      python3 -u downstream_seed.py > downstream_seed.log 2>&1

      echo "====> [GPU 0] Finished $dir"
    ) &

    ((job_count++))
    if (( job_count >= MAX_JOBS )); then
      # Wait for any one job to finish, then decrement counter
      wait -n
      ((job_count--))
    fi
  else
    echo "[GPU 0] Directory $dir not found; skipping."
  fi
done

# Wait for any remaining jobs to finish
wait

echo "====> [GPU 0] All jobs complete. <===="
