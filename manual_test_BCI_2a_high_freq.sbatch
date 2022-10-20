#!/bin/bash --login
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6 
#SBATCH --mem=32G 
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --output=./manual_testing_BCI_IV_2a_high_freq_logs/%x-%j-slurm.out
#SBATCH --error=./manual_testing_BCI_IV_2a_high_freq_logs/%x-%j-slurm.err

set -e

conda activate ../env

python test_prime_BCI_2a_high_freq_manually.py 