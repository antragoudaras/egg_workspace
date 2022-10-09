#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/ECoG_recovering_actual_accuracy_r2_scores_positive_COM_logs
mkdir -p "$JOB_RESULTS_DIR"

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit 1
fi

JOB_NAME=$1
echo "$JOB_NAME"

sbatch --job-name "$JOB_NAME" "$SRC_DIR"/test_ECoG_actual_positive_COM.sbatch --load-dataset "$JOB_NAME"


echo "Testing 25 Shallow ConvNet arch. of BCI IV 4 positive COM"