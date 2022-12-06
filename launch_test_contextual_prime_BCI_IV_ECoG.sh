#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/ground_truth_contextual_prime_BCI_4_ECoG_all_patients_logs
mkdir -p "$JOB_RESULTS_DIR"

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit 1
fi

JOB_NAME=$1

DATASET_PATH="$SRC_DIR"/contextual_ECoG_positive_COM_optimized_params_november/"$JOB_NAME"
echo "$DATASET_PATH"

sbatch --job-name "$JOB_NAME" "$SRC_DIR"/test_contextual_prime_BCI_IV_ECoG.sbatch --load-dataset "$DATASET_PATH"


echo "Testing of 25 contextual PRIME generated archs. of BCI ECoG all subjects/multi-context optimization. After .xlsx deletion"