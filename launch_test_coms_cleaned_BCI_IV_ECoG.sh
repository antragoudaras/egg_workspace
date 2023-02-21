#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_LOGS_DIR="$PROJECT_DIR"/ground_truth_BCI_IV_ECoG_coms_cleaned_logs
mkdir -p "$JOB_LOGS_DIR"

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit 1
fi

DATASET_NAME=$1

DATASET_PATH="$SRC_DIR"/COMs_candidate_archs_discovered/"$DATASET_NAME"
echo "$DATASET_PATH"

sbatch --job-name "$DATASET_NAME" "$SRC_DIR"/test_coms_cleaned_BCI_IV_ECoG.sbatch --load-dataset "$DATASET_PATH"


echo "Testing discovered archs. by coms_cleaned on BCI IV ECoG challenge ..."