#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/results_generate_bci_iv_2a_high_freq_dataset_contextual
mkdir -p "$JOB_RESULTS_DIR"

JOB_NAME=bci_iv_2a_low_freq_initial_baselines_high_freq
sbatch --job-name "$JOB_NAME" "$SRC_DIR"/build_bci_2a_EEG_dataset_ibex_high_freq.sbatch

echo "Generating baselines architectures per subject/context of BCI 2a EEG high frequency dataset"