#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/results_generate_bci_iv_2a_low_freq_dataset_contextual
mkdir -p "$JOB_RESULTS_DIR"

for counter in {1..80..1}
do
    JOB_NAME=bci_iv_2a_low_freq_${counter}_out_of_80_contextual_november
    echo "$JOB_NAME"
    sbatch --job-name "$JOB_NAME" "$SRC_DIR"/build_bci_2a_EEG_dataset_ibex_low_freq.sbatch --save-dataset "${JOB_NAME}.xlsx"
done

echo "Generating 8000 architectures per subject/context of BCI 2a EEG low frequency dataset"