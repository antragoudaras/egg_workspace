#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/building_baseline_bci_iv_2a_low_freq_log_outputs
mkdir -p "$JOB_RESULTS_DIR"

JOB_NAME=building_testing_bci_iv_2a_low_freq
sbatch --job-name "$JOB_NAME" "$SRC_DIR"/build_bci_2a_EEG_dataset_ibex_low_freq.sbatch


echo "Testing the baseline ShallowConvnet BCI 2a low freq..."