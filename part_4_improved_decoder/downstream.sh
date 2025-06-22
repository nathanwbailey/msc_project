#!/bin/bash

# --- Configuration ---

# An array of directories to process.
# Add your directory paths here. For example:
# DIRS=("/path/to/your/first/folder" "/path/to/your/second/folder" "/another/path")
DIRS=(
  "simclr_decoder_improved_mse_loss"
  "simclr_decoder_improved_mse_loss_decoded"
  "simclr_decoder_improved_mse_loss_weighted_losses"
  "simclr_decoder_weight_decay"
)

# The names of the two Python files you want to execute.
FILE1="main.py"
FILE2="downstream_seed.py"

LOG_FILE="downstream_out.log"
LOG_FILE_GEN="out.log"



# Get the current directory to return to it after processing each subdirectory.
ORIGINAL_DIR=$(pwd)

log_message() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE_GEN"
}

log_message "--- Starting script execution ---"

# Loop through each directory specified in the DIRS array.
for dir in "${DIRS[@]}"; do
  log_message "Processing directory: $dir"

  # Check if the directory exists and is actually a directory.
  if [ -d "$dir" ]; then
    # Change into the directory.
    cd "$dir" || { log_message "Error: Could not cd into $dir. Skipping."; continue; }


    log_message "Successfully changed to directory: $(pwd)"

    # --- Process File 1 ---
    if [ -f "$FILE1" ]; then
      if [ -r "$FILE1" ]; then
        log_message "Executing $FILE1..."
        # Execute the python script with unbuffered output (-u) and append stdout/stderr to the log.
        python3 -u "$FILE1" > "$LOG_FILE" 2>&1
        log_message "$FILE1 finished."
      else
        log_message "Warning: $FILE1 exists but is not readable. Skipping."
      fi
    else
      log_message "Info: $FILE1 does not exist in this directory."
    fi

    # --- Process File 2 ---
    if [ -f "$FILE2" ]; then
       # For Python scripts, we check for read permission.
      if [ -r "$FILE2" ]; then
        log_message "Executing $FILE2..."
        # Execute the python script with unbuffered output (-u) and append stdout/stderr.
        python3 -u "$FILE2" >> "$LOG_FILE" 2>&1
        log_message "$FILE2 finished."
      else
        log_message "Warning: $FILE2 exists but is not readable. Skipping."
      fi
    else
      log_message "Info: $FILE2 does not exist in this directory."
    fi

    # Return to the original directory.
    cd "$ORIGINAL_DIR" || { log_message "FATAL: Could not return to original directory $ORIGINAL_DIR. Exiting."; exit 1; }
  else
    log_message "Warning: Directory '$dir' does not exist. Skipping."
  fi
done

log_message "--- Script execution finished ---"

echo "All directories processed. Check the log file at $LOG_FILE for details."
