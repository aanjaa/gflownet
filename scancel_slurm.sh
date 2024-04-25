#!/bin/bash

# The job ID to keep
keep_job_id=4006564

# Check if a job ID has been provided
if [[ -z "$keep_job_id" ]]; then
  echo "Please provide a job ID to keep."
  exit 1
fi

# Loop over all jobs belonging to the user
for job in $(squeue -u "$USER" -h -o "%i"); do
  if [[ "$job" != "$keep_job_id" ]]; then
    scancel "$job"
  fi
done
