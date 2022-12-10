#!/bin/bash --login
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=8 
#SBATCH --mem=64G 
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --output=./ground_truth_contextual_prime_high_freq_BCI_2a_EEG_all_patients_logs/%x-%j-slurm.out
#SBATCH --error=./ground_truth_contextual_prime_high_freq_BCI_2a_EEG_all_patients_logs/%x-%j-slurm.err

set -e

conda activate ../env

python test_contextual_prime_high_freq_BCI_2a_EEG.py "$@"