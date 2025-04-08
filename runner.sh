#!/bin/bash

# --- Configuration ---
PYTHON_SCRIPT="dpo2.py" # The script to run
LOG_FILE="large_dpo_run.log"  # File to store output (stdout & stderr)
SLEEP_DURATION=10       # Seconds to wait before restarting after a failure

# --- Important Setting ---
# Ensures that the exit status of a pipeline (like 'python | tee')
# is the status of the last command to exit with non-zero status,
# or zero if all commands exit successfully. This makes sure we capture
# the python script's exit status correctly, not just tee's.
set -o pipefail

# --- Initialization ---
echo "Starting DPO training wrapper script."
echo "Output and errors will be mirrored to console and logged to: $LOG_FILE"
echo "-------------------------------------" | tee -a "$LOG_FILE" # Use tee to print to console AND log file
echo "Wrapper started at $(date)" | tee -a "$LOG_FILE"
echo "-------------------------------------" | tee -a "$LOG_FILE"

# --- Main Loop ---
while true; do
    echo "[$(date)] Launching $PYTHON_SCRIPT..." | tee -a "$LOG_FILE"

    # Run the python script:
    # 1. 'python "$PYTHON_SCRIPT"': Executes the script.
    # 2. '2>&1': Redirects standard error (stderr, file descriptor 2)
    #            to standard output (stdout, file descriptor 1).
    #            Now both normal output and errors go to stdout.
    # 3. '|': Pipes the combined stdout stream to the next command.
    # 4. 'tee -a "$LOG_FILE"': Reads the piped input.
    #    - Prints it to standard output (the terminal).
    #    - '-a' appends it to the specified log file.
    python -u "$PYTHON_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
    exit_status=$? # Capture the exit status of the pipeline (respecting pipefail)

    # Check the exit status
    if [ $exit_status -eq 0 ]; then
        # Exit code 0 means success
        echo "[$(date)] $PYTHON_SCRIPT finished successfully (Exit Code: 0)." | tee -a "$LOG_FILE"
        break # Exit the loop
    else
        # Non-zero exit code means failure
        echo "[$(date)] $PYTHON_SCRIPT failed with Exit Code: $exit_status." | tee -a "$LOG_FILE"
        echo "[$(date)] Waiting $SLEEP_DURATION seconds before restarting..." | tee -a "$LOG_FILE"
        sleep $SLEEP_DURATION
        echo "[$(date)] Restarting..." | tee -a "$LOG_FILE"
        # Loop continues automatically
    fi
done

# --- Cleanup ---
echo "-------------------------------------" | tee -a "$LOG_FILE"
echo "Wrapper script finished at $(date)." | tee -a "$LOG_FILE"
echo "-------------------------------------" | tee -a "$LOG_FILE"
echo "Wrapper script finished."

# Reset pipefail if desired (optional, script is ending anyway)
# set +o pipefail
exit 0
