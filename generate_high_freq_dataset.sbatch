#!/bin/bash --login
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6 
#SBATCH --mem=32G 
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --output=./results_high_freq/%x-%j-slurm.out
#SBATCH --error=./results_high_freq/%x-%j-slurm.err

set -e

conda activate ../env

python extend_high_freq.py "$1" "$2"