#!/bin/bash --login
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=4 
#SBATCH --mem=16G 
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --output=./building_baseline_bci_iv_2a_low_freq_log_outputs/%x-%j-slurm.out
#SBATCH --error=./building_baseline_bci_iv_2a_low_freq_log_outputs/%x-%j-slurm.err

set -e

conda activate ../env

python extend_BCI_2a_EEG_cropped_context_individual_low_freq.py