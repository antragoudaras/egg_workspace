#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/ground_truth_contextual_prime_BCI_2a_EEG_low_freq_leave_one_out_logs
mkdir -p "$JOB_RESULTS_DIR"

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit 1
fi

JOB_NAME=$1

DATASET_PATH="$SRC_DIR"/contextual_EEG_low_freq_leave_one_out_optimized_params_december/"$JOB_NAME"
echo "$DATASET_PATH"

sbatch --job-name "$JOB_NAME" "$SRC_DIR"/test_contextual_prime_leave_one_out_low_freq_BCI_2a_EEG.sbatch --load-dataset "$DATASET_PATH"


echo "Testing of 25 contextual PRIME generated archs. trained with leave-one-patient-out technique of BCI EEG low-freq"