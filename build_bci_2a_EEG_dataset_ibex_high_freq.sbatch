#!/bin/bash --login
#SBATCH --time=32:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6 
#SBATCH --mem=32G 
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --mail-user=antonios.tragoudaras@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --output=./results_generate_bci_iv_2a_high_freq_dataset_contextual/%x-%j-slurm.out
#SBATCH --error=./results_generate_bci_iv_2a_high_freq_dataset_contextual/%x-%j-slurm.err

set -e

conda activate ../env

python extend_BCI_2a_EEG_cropped_context_individual_high_freq.py "$@"