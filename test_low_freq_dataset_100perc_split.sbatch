#!/bin/bash --login
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6 
#SBATCH --mem=64G 
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --output=./validate_prime_optimized_results_low_freq_1000batch_size_100perc_split/%x-%j-slurm.out
#SBATCH --error=./validate_prime_optimized_results_low_freq_1000batch_size_100perc_split/%x-%j-slurm.err

set -e

conda activate ../env

python test_prime_low_freq_100perc.py "$1" "$2"