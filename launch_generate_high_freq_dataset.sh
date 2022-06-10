#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/results_high_freq
mkdir -p "$JOB_RESULTS_DIR"

for counter in {1..70..1}
do
    JOB_NAME=high_freq_dataset_${counter}_out_of_80
    echo "$JOB_NAME"
    sbatch --job-name "$JOB_NAME" "$SRC_DIR"/generate_high_freq_dataset.sbatch --save-dataset "./$JOB_NAME.xlsx"
done

echo "Generating 7000 lines of high freq dataset"