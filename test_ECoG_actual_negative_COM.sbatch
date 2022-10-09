#!/bin/bash --login
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6 
#SBATCH --mem=32G 
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --output=./ECoG_recovering_actual_accuracy_r2_scores_negative_COM_logs/%x-%j-slurm.out
#SBATCH --error=./ECoG_recovering_actual_accuracy_r2_scores_negative_COM_logs/%x-%j-slurm.err

set -e

conda activate ../env

python test_prime_BCI_IV_ECoG_negative_COM.py "$1" "$2"