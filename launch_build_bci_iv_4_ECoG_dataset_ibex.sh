#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/building_folder_mne_data_bci_iv_4_ECoG_log_outputs
mkdir -p "$JOB_RESULTS_DIR"

JOB_NAME=building_testing_bci_iv_4_ECoG
sbatch --job-name "$JOB_NAME" "$SRC_DIR"/build_bci_iv_4_ECoG_dataset_ibex.sbatch


echo "Downloading BCI IV ECoG Dataset & Testing the baseline ShallowConvnet..."