#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/validate_prime_optimized_results_low_freq_1000batch_size
mkdir -p "$JOB_RESULTS_DIR"

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit 1
fi

JOB_NAME=$1
echo "$JOB_NAME"

sbatch --job-name "$JOB_NAME" "$SRC_DIR"/test_low_freq_dataset.sbatch --load-dataset "./$JOB_NAME"


echo "Testing 8000 lines of low freq dataset"