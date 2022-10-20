#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/manual_testing_BCI_IV_2a_low_freq_logs
mkdir -p "$JOB_RESULTS_DIR"

JOB_NAME=manual_testing_BCI_IV_2a_low_freq

sbatch --job-name "$JOB_NAME" "$SRC_DIR"/manual_test_BCI_2a_low_freq.sbatch

echo "Manually testing BCI IV 2a low_freq given architectures"